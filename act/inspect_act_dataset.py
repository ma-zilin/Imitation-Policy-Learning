"""ACT A2：复用固定 Push-T 数据，检查窗口、batch、归一化和最小 CUDA 运算。

不创建策略网络，不执行 backward、optimizer、checkpoint 或环境 rollout。
"""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# 所有数据访问都使用本地缓存，不允许隐式下载。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import av
import torch
from torch.utils.data import DataLoader

from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.datasets.video_utils import decode_video_frames_torchvision
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_KEYS = ("observation.image", "observation.state", "action", "action_is_pad")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def json_value(value):
    """用于配置冻结与 JSON 比较；不保留 tuple/Enum 的 Python 表示。"""
    return json.loads(json.dumps(value))


def load_config(path):
    spec = read_json(path)
    require(spec["schema_version"] == 1, "不支持的配置 schema")
    fields = dict(spec["policy"])
    for group in ("input_features", "output_features"):
        fields[group] = {
            key: PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
            for key, value in fields[group].items()
        }
    fields["normalization_mapping"] = {
        FeatureType(key): NormalizationMode(value)
        for key, value in fields["normalization_mapping"].items()
    }
    policy = ACTConfig(**fields)
    policy.validate_features()
    require(json_value(dataclasses.asdict(policy)) == spec["policy"],
            "配置未完整冻结当前 ACTConfig 字段或默认值发生漂移")
    require(policy.pretrained_backbone_weights is None, "A2 禁止引入未核验的骨干权重")
    require(spec["batch_size"] == 2 and spec["inspection"]["batches_per_split"] == 2,
            "本轮只检查每个 split 的两个 batch，每 batch 两个样本")
    require(spec["inspection"]["num_workers"] == 0, "本轮只用主进程解码")
    return spec, policy


def build_delta_timestamps(config, fps):
    require(fps > 0, "fps 必须为正")
    require(config.observation_delta_indices is None, "ACT 只使用当前观测")
    return {"action": [index / fps for index in config.action_delta_indices]}


def validate_split(split, episodes):
    require(split.get("schema_version") == 1 and split.get("dataset") == "lerobot/pusht",
            "划分 schema 或数据集错误")
    groups = [split["train_episodes"], split["val_episodes"]]
    require(all(group and len(group) == len(set(group)) for group in groups), "划分为空或有重复")
    require(all(type(item) is int for group in groups for item in group), "episode ID 必须为整数")
    train, val = map(set, groups)
    require(not train & val, "训练与验证 episode 泄漏")
    require(train | val == set(episodes), "划分没有恰好覆盖全部 episode")


def validate_stats_source(source, split, lengths, fps):
    require(source["train_episodes"] == split["train_episodes"], "统计量包含错误的训练 episode")
    require(source["raw_frame_count"] == sum(lengths[e] for e in split["train_episodes"]),
            "统计量原始帧数不匹配")
    require(source["fps"] == fps, "统计量 fps 不匹配")
    require(source["method"] == "LeRobot compute_stats; all raw training frames; no temporal windows or augmentation",
            "统计方法不符合训练原始帧约定")


def validate_stats(stats, features):
    for key, feature in features.items():
        shape = (feature.shape[0], 1, 1) if feature.type is FeatureType.VISUAL else tuple(feature.shape)
        for name in ("mean", "std", "min", "max"):
            value = stats[key][name]
            require(tuple(value.shape) == shape and bool(torch.isfinite(value).all()),
                    f"统计量形状或数值错误：{key}.{name}")
        require(bool((stats[key]["std"] > 0).all()), f"标准差必须为正：{key}")
        require(bool((stats[key]["max"] >= stats[key]["min"]).all()), f"min/max 错误：{key}")


def git_info(root):
    def run(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).rstrip()
    return {"head": run("rev-parse", "HEAD"), "status": run("status", "--short")}


def verify_provenance(spec):
    """核验记录、全部数据文件及统计来源；不重新计算统计量。"""
    data = spec["dataset"]
    root = Path(data["root"])
    paths = {}
    for name in ("split", "stats", "manifest"):
        paths[name] = PROJECT_ROOT / data[f"{name}_path"]
        require(sha256(paths[name]) == data[f"{name}_sha256"], f"冻结文件指纹改变：{name}")
    manifest, split, saved_stats = (read_json(paths[key]) for key in ("manifest", "split", "stats"))
    require(manifest["status"] == "PASS" and manifest["dataset"] == data["repo_id"], "下载清单不匹配")
    require(manifest["revision"] == data["revision"], "数据 revision 不匹配")
    require(Path(manifest["dataset_root"]).resolve() == root.resolve(), "下载清单根目录不匹配")
    info = read_json(root / "meta/info.json")
    records = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line]
    lengths = {entry["episode_index"]: entry["length"] for entry in records}
    require(len(lengths) == info["total_episodes"] and sum(lengths.values()) == info["total_frames"],
            "episode 元数据数量不匹配")
    require(info["fps"] == data["fps"], "数据 fps 改变")
    validate_split(split, lengths)
    expected_files = {f"meta/{name}" for name in ("info.json", "episodes.jsonl", "tasks.jsonl", "stats.json")}
    training_files = set()
    for episode in lengths:
        fields = {"episode_index": episode, "episode_chunk": episode // info["chunks_size"]}
        names = {info["data_path"].format(**fields)}
        names.update(info["video_path"].format(video_key=key, **fields)
                     for key, value in info["features"].items() if value["dtype"] == "video")
        expected_files.update(names)
        if episode in split["train_episodes"]:
            training_files.update(names)
    require(set(manifest["files"]) == expected_files and manifest["file_count"] == len(expected_files),
            "下载清单没有恰好覆盖所需数据文件")
    file_hashes = {}
    for rel, record in manifest["files"].items():
        path = root / rel
        require(path.is_file() and path.stat().st_size == record["size"], f"数据缺失或大小变化：{rel}")
        file_hashes[rel] = sha256(path)
        require(file_hashes[rel] == record["sha256"], f"数据指纹改变：{rel}")
    source = saved_stats["source"]
    require(source["dataset"] == data["repo_id"] and source["dataset_revision"] == data["revision"],
            "训练统计量的数据来源错误")
    require(Path(source["dataset_root"]).resolve() == root.resolve(), "统计量根目录不匹配")
    validate_stats_source(source, split, lengths, info["fps"])
    require(set(source["training_files_sha256"]) == training_files, "统计量文件集合不等于训练 episode")
    for rel, expected in source["training_files_sha256"].items():
        require(file_hashes[rel] == expected, f"训练统计量来源数据改变：{rel}")
    for name, expected in source["metadata_sha256"].items():
        require(file_hashes[f"meta/{name}"] == expected, f"统计量来源元数据改变：{name}")
    components = {"statistics": compute_stats, "dataset": LeRobotDataset,
                  "video_decoder": decode_video_frames_torchvision,
                  "dataset_utils": dataset_to_policy_features, "act_config": ACTConfig,
                  "normalization": Normalize}
    fingerprints = {key: {"path": inspect.getfile(value), "sha256": sha256(inspect.getfile(value))}
                    for key, value in components.items()}
    for key, expected in source["implementation_sha256"].items():
        require(fingerprints[key]["sha256"] == expected, f"统计实现发生漂移：{key}")
    for key, expected in source["packages"].items():
        require(importlib.metadata.version(key) == expected, f"统计环境版本发生漂移：{key}")
    upstream = Path(inspect.getfile(ACTConfig)).resolve().parents[4]
    upstream_git = git_info(upstream)
    require(upstream_git["head"] == spec["lerobot_revision"], "LeRobot revision 改变")
    for line in upstream_git["status"].splitlines():
        rel = line[3:]
        path = upstream / rel
        if path.is_file():
            fingerprints[f"dirty:{rel}"] = {"path": str(path), "sha256": sha256(path)}
    stats = {key: {name: torch.tensor(value, dtype=torch.float32) for name, value in entry.items()}
             for key, entry in saved_stats["dataset_stats"].items()}
    summary = {
        "revision": data["revision"], "verified_files": len(file_hashes),
        "verified_bytes": sum(value["size"] for value in manifest["files"].values()),
        "split_counts": {key: len(split[f"{key}_episodes"]) for key in ("train", "val")},
        "statistics_raw_training_frames": source["raw_frame_count"],
        "statistics_training_files": len(training_files), "statistics_recomputed": False,
        "statistics_source": source, "implementation_fingerprints": fingerprints,
        "lerobot_git": upstream_git,
        "padding_windows_from_metadata_all_raw_anchors": {
            str(k): {"with_padding": sum(min(k - 1, lengths[e]) for e in split["train_episodes"]),
                     "total": source["raw_frame_count"]} for k in (16, 100)
        },
    }
    # 报告保留完整来源的摘要和已有文件指纹，不重复几百条文件哈希。
    summary["statistics_source"] = {key: value for key, value in source.items()
                                    if key not in ("training_files_sha256", "metadata_sha256", "implementation_sha256")}
    return split, stats, lengths, info, summary


def expected_window(anchor, length, chunk):
    """独立按 episode 局部帧号构造预期，不调用 LeRobot 私有窗口函数。"""
    require(0 <= anchor < length and chunk > 0, "窗口锚点或长度错误")
    requested = list(range(anchor, anchor + chunk))
    return requested, [min(frame, length - 1) for frame in requested], [frame >= length for frame in requested]


def reference_video(path, timestamps):
    """顺序 PyAV 解码作为独立参考，不复用 LeRobot 的 seek/选帧与通道转换代码。"""
    targets = sorted(set(timestamps))
    best = {target: (float("inf"), None) for target in targets}
    decoded = 0
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            decoded += 1
            ts = float(frame.pts * frame.time_base)
            for target in targets:
                distance = abs(ts - target)
                if distance < best[target][0]:
                    pixels = frame.to_ndarray(format="rgb24").copy()
                    tensor = torch.from_numpy(pixels).movedim(-1, 0).float() / 255
                    best[target] = (distance, tensor)
    for target in targets:
        require(best[target][0] <= 1e-4, f"参考视频时间戳不匹配：{target}")
    return {target: best[target][1] for target in targets}, decoded


def check_sample(dataset, position, episode, anchor, length, start, config, reference=None):
    sample = dataset[position]
    raw = dataset.hf_dataset[position]
    require(int(sample["episode_index"]) == episode and int(sample["frame_index"]) == anchor,
            "样本 episode/局部帧号不匹配")
    requested, actual, padding = expected_window(anchor, length, config.chunk_size)
    rows = [dataset.hf_dataset[start + frame] for frame in actual]
    require(all(int(row["episode_index"]) == episode for row in rows), "动作跨越了 episode")
    expected_actions = torch.stack([row["action"] for row in rows])
    torch.testing.assert_close(sample["action"], expected_actions, rtol=0, atol=0)
    torch.testing.assert_close(sample["observation.state"], raw["observation.state"], rtol=0, atol=0)
    torch.testing.assert_close(sample["timestamp"], raw["timestamp"], rtol=0, atol=0)
    require(sample["action_is_pad"].dtype == torch.bool, "padding 标记必须为 bool")
    require(sample["action_is_pad"].tolist() == padding, "padding 标记不匹配")
    expected_shapes = {"observation.image": (3, 96, 96), "observation.state": (2,),
                       "action": (config.chunk_size, 2), "action_is_pad": (config.chunk_size,)}
    for key, shape in expected_shapes.items():
        require(tuple(sample[key].shape) == shape, f"样本形状错误：{key}")
        require(bool(torch.isfinite(sample[key]).all()), f"非有限输入：{key}")
        if key != "action_is_pad":
            require(sample[key].dtype == torch.float32, f"数据 dtype 错误：{key}")
    image = sample["observation.image"]
    require(float(image.min()) >= 0 and float(image.max()) <= 1, "图像不在 [0,1]")
    if reference is not None:
        torch.testing.assert_close(image, reference, rtol=0, atol=0)
    return sample, {
        "episode": episode, "frame": anchor, "dataset_position": position,
        "original_global_index": int(sample["index"]), "timestamp": float(sample["timestamp"]),
        "requested_action_frames": requested, "actual_action_frames": actual,
        "action_is_pad": padding, "padded_positions": sum(padding),
        "shapes": {key: list(sample[key].shape) for key in DATA_KEYS},
        "current_state": sample["observation.state"].tolist(),
        "first_action": sample["action"][0].tolist(), "last_action": sample["action"][-1].tolist(),
        "independent_video_match": reference is not None,
    }


def check_normalization(batch, policy, stats, tolerances, device="cpu"):
    features = {**policy.input_features, **policy.output_features}
    normalizer = Normalize(features, policy.normalization_mapping, stats).to(device)
    inverse = Unnormalize(features, policy.normalization_mapping, stats).to(device)
    values = {key: value.to(device) for key, value in batch.items() if key in DATA_KEYS}
    original = {key: value.clone() for key, value in values.items()}
    normalized = normalizer(values)
    restored = inverse(normalized)
    require(all(not parameter.requires_grad for parameter in normalizer.parameters()), "统计量不应参与参数学习")
    result = {}
    for key in features:
        mean, std = stats[key]["mean"].to(device), stats[key]["std"].to(device)
        # 独立复算，避免正反两个函数以同一种错误互相抵消。
        manual = (values[key] - mean) / (std + 1e-8)
        torch.testing.assert_close(normalized[key], manual, rtol=1e-6, atol=1e-6)
        require(bool(torch.isfinite(normalized[key]).all()), f"归一化出现 NaN/Inf：{key}")
        error = float((restored[key] - values[key]).abs().max())
        require(error <= tolerances[key], f"归一化往返误差超限：{key}: {error}")
        torch.testing.assert_close(values[key], original[key], rtol=0, atol=0)
        result[key] = {"round_trip_max_abs_error": error, "tolerance": tolerances[key],
                       "raw_min": float(values[key].min()), "raw_max": float(values[key].max()),
                       "normalized_min": float(normalized[key].min()),
                       "normalized_max": float(normalized[key].max()), "manual_formula_match": True}
    require(torch.equal(normalized["action_is_pad"], values["action_is_pad"]), "归一化不应改变 padding")
    return result


def inspect_split(name, selected, metadata, spec, policy, stats, lengths):
    dataset = LeRobotDataset(spec["dataset"]["repo_id"], root=metadata.root, episodes=selected,
                            delta_timestamps=build_delta_timestamps(policy, metadata.fps),
                            local_files_only=True, video_backend="pyav")
    require(dataset.image_transforms is None, "A2 不使用图像增强")
    result = {"selected_episodes": selected, "windows": [], "batches": [],
              "reference_video_decoded_frames": 0}
    batches = []
    for offset, episode in enumerate(selected):
        start = int(dataset.episode_data_index["from"][offset])
        end = int(dataset.episode_data_index["to"][offset])
        length = lengths[episode]
        require(end - start == length, "子集 episode 边界不匹配")
        anchors = sorted({0, length // 2, max(0, length - policy.chunk_size),
                          max(0, length - policy.chunk_size + 1), length - 1})
        times = {anchor: float(dataset.hf_dataset[start + anchor]["timestamp"]) for anchor in anchors}
        video = metadata.root / metadata.get_video_file_path(episode, "observation.image")
        references, decoded = reference_video(video, list(times.values()))
        result["reference_video_decoded_frames"] += decoded
        for anchor in anchors:
            _, details = check_sample(dataset, start + anchor, episode, anchor, length, start,
                                      policy, references[times[anchor]])
            result["windows"].append(details)
        # 两个 batch 分别混合完整中间窗和末尾补位窗。
        batches.append([start + length // 2, end - 1])
    require(len(batches) == spec["inspection"]["batches_per_split"], "batch 数量不匹配")
    loader = DataLoader(dataset, batch_sampler=batches, num_workers=0,
                        generator=torch.Generator().manual_seed(spec["seed"]))
    saved_batches = []
    for number, batch in enumerate(loader):
        positions = batches[number]
        for row, position in enumerate(positions):
            item = dataset.hf_dataset[position]
            episode, anchor = int(item["episode_index"]), int(item["frame_index"])
            ep_offset = selected.index(episode)
            start = int(dataset.episode_data_index["from"][ep_offset])
            checked, _ = check_sample(dataset, position, episode, anchor, lengths[episode], start, policy)
            for key in DATA_KEYS:
                torch.testing.assert_close(batch[key][row], checked[key], rtol=0, atol=0)
        shapes = {key: list(batch[key].shape) for key in DATA_KEYS}
        require(shapes == {"observation.image": [2, 3, 96, 96], "observation.state": [2, 2],
                           "action": [2, policy.chunk_size, 2], "action_is_pad": [2, policy.chunk_size]},
                "真实 DataLoader batch 形状错误")
        norm = check_normalization(batch, policy, stats, spec["inspection"]["round_trip_atol"])
        images = torch.stack([batch[key] for key in policy.image_features], dim=-4)
        require(tuple(images.shape) == (2, 1, 3, 96, 96), "ACT 多相机堆叠形状错误")
        result["batches"].append({"number": number, "dataset_positions": positions,
                                  "episode_ids": batch["episode_index"].tolist(), "shapes": shapes,
                                  "camera_stack_shape": list(images.shape),
                                  "padding_positions_per_sample": batch["action_is_pad"].sum(-1).tolist(),
                                  "normalization_cpu": norm})
        saved_batches.append({key: batch[key] for key in DATA_KEYS})
    return result, saved_batches


def check_cuda(batch, spec, policy, stats):
    require(torch.cuda.is_available(), "CUDA 不可用；不自动改用 CPU 掩盖 GPU 检查")
    # 一次极小矩阵运算和一个真实 batch 的归一化，不构造 ACT 网络。
    with torch.inference_mode():
        cpu = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        gpu = cpu.to("cuda")
        result = gpu @ gpu.T
        torch.cuda.synchronize()
        torch.testing.assert_close(result.cpu(), cpu @ cpu.T, rtol=0, atol=0)
        norm = check_normalization(batch, policy, stats, spec["inspection"]["round_trip_atol"], "cuda")
        torch.cuda.synchronize()
    props = torch.cuda.get_device_properties(0)
    return {"status": "PASS", "name": props.name, "total_memory_bytes": props.total_memory,
            "torch_cuda_version": torch.version.cuda, "matrix_shape": [4, 4],
            "matrix_matches_cpu": True, "real_batch_normalization": norm,
            "act_model_memory_or_training_feasibility_tested": False}


def execute(spec, policy, report, device):
    torch.set_num_threads(spec["num_threads"])
    torch.manual_seed(spec["seed"])
    split, stats, lengths, info, provenance = verify_provenance(spec)
    report["provenance"] = provenance
    print(f"provenance PASS: {provenance['verified_files']} files; split 185/21", flush=True)
    metadata = LeRobotDatasetMetadata(spec["dataset"]["repo_id"], root=spec["dataset"]["root"],
                                     local_files_only=True)
    require(metadata.fps == spec["dataset"]["fps"], "加载元数据后的 fps 不匹配")
    features = dataset_to_policy_features(metadata.features)
    for key, value in {**policy.input_features, **policy.output_features}.items():
        require(value.type == features[key].type and tuple(value.shape) == tuple(features[key].shape),
                f"策略字段与真实元数据不匹配：{key}")
    validate_stats(stats, {**policy.input_features, **policy.output_features})
    report["normalization_source"] = {"path": spec["dataset"]["stats_path"],
                                      "sha256": spec["dataset"]["stats_sha256"],
                                      "repository_meta_stats_used_for_normalization": False,
                                      "same_training_stats_for_both_splits": True}
    report["delta_timestamps"] = build_delta_timestamps(policy, metadata.fps)
    report["splits"] = {}
    first_batch = None
    for name in ("train", "val"):
        selected = spec["inspection"][f"{name}_episodes"]
        require(len(selected) == 2 and len(set(selected)) == 2, "每组固定两个不同 episode")
        require(set(selected) <= set(split[f"{name}_episodes"]), "检查 episode 不属于当前 split")
        details, batches = inspect_split(name, selected, metadata, spec, policy, stats, lengths)
        report["splits"][name] = details
        first_batch = batches[0] if first_batch is None else first_batch
        print(f"{name} PASS: {len(details['windows'])} windows, {len(details['batches'])} batches", flush=True)
    if device == "cuda":
        report["gpu"] = check_cuda(first_batch, spec, policy, stats)
        report["status"] = "PASS"
    else:
        report["gpu"] = {"status": "NOT_RUN", "reason": "显式 --device cpu；不能声明 GPU 通过"}
        report["status"] = "DATA_PASS_GPU_NOT_RUN"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/act/pusht_a2.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output_dir or PROJECT_ROOT / "outputs/act" / f"a2_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "RUNNING", "started_utc": stamp,
              "scope": "A2 data inspection and minimal compute only",
              "config_path": str(args.config.resolve()), "config_sha256": sha256(args.config),
              "command": [sys.executable, *sys.argv], "project_git_before_run": git_info(PROJECT_ROOT),
              "packages": {key: importlib.metadata.version(key) for key in
                           ("torch", "torchvision", "lerobot", "datasets", "av", "gym-pusht")},
              "python": platform.python_version(), "python_executable": sys.executable,
              "script_sha256": sha256(__file__),
              "not_executed": ["ACTPolicy construction", "policy forward/backward", "optimizer updates",
                               "checkpoint save/reload", "environment rollout", "dependency changes"],
              "scope_limits": ["selected windows/batches, not all dataset windows decoded",
                               "GPU probe does not establish ACT training memory or speed",
                               "runtime PASS does not establish independent learner mastery"],
              "field_semantics": {"observation.state": "current pusher [x,y] in 0..512 scene coordinates",
                                  "action": "absolute target pusher [x,y] in the same scene coordinates",
                                  "observation.image": "RGB float32 CHW [0,1] before standardization",
                                  "camera_stack_axis_1": "camera, not observation time"}}
    try:
        spec, policy = load_config(args.config)
        report["effective_policy_config"] = json_value(dataclasses.asdict(policy))
        execute(spec, policy, report, args.device)
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        print(f"{report['status']}: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
