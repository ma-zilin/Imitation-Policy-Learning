"""准备 DP5-B 的 episode 划分、训练集统计量与 DataLoader；只读本地缓存，不训练。"""

import argparse
import hashlib
import inspect
import json
import math
from importlib.metadata import version
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from inspect_pusht_dataset import (
    DATASET_REPO_ID,
    POLICY_REPO_ID,
    POLICY_REVISION,
    build_delta_timestamps,
)
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import (
    LEROBOT_HOME,
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.video_utils import decode_video_frames_torchvision
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: dict, *, exclusive: bool = False) -> None:
    """划分和统计量只新建；检查报告可由下一次检查更新。"""
    with path.open("x" if exclusive else "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write("\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split(split: dict, episode_ids: list[int]) -> None:
    """检查完整 episode 划分，绝不从一个 episode 的窗口构造两组数据。"""
    if split.get("schema_version") != 1 or split.get("dataset") != DATASET_REPO_ID:
        raise ValueError("Unsupported split schema or dataset")
    groups = [split.get("train_episodes"), split.get("val_episodes")]
    for group in groups:
        if not isinstance(group, list) or not group:
            raise ValueError("Both train and validation episode lists must be nonempty")
        if any(type(item) is not int for item in group) or len(set(group)) != len(group):
            raise ValueError("Episode IDs must be unique integers within each split")
    train, val = map(set, groups)
    if train & val or train | val != set(episode_ids):
        raise ValueError("Split must be disjoint and cover exactly the dataset episodes")


def build_episode_split(episode_ids: list[int], seed: int, val_fraction: float, path: Path) -> dict:
    """新划分使用独立随机数生成器；已有划分只读取，不重新覆盖。"""
    if len(episode_ids) < 2 or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("Need at least two distinct episodes")
    if not math.isfinite(val_fraction) or not 0 < val_fraction < 1:
        raise ValueError("Validation fraction must be between 0 and 1")
    if path.exists():
        split = json.loads(path.read_text(encoding="utf-8"))
        validate_split(split, episode_ids)
        if split.get("seed") != seed or split.get("val_fraction") != val_fraction:
            raise ValueError("Existing split uses different settings; choose a new output directory")
        return split

    ids = sorted(episode_ids)
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed)).tolist()
    val_count = min(len(ids) - 1, max(1, math.ceil(len(ids) * val_fraction)))
    split = {
        "schema_version": 1, "dataset": DATASET_REPO_ID,
        "seed": seed, "val_fraction": val_fraction,
        "train_episodes": sorted(ids[i] for i in order[val_count:]),
        "val_episodes": sorted(ids[i] for i in order[:val_count]),
    }
    validate_split(split, ids)
    write_json(path, split, exclusive=True)
    return split


def required_files(metadata, episode_ids: list[int]) -> list[Path]:
    """在创建 Dataset 前检查表格和视频，缺失时不会触发下载。"""
    paths = []
    for episode in episode_ids:
        paths.append(metadata.root / metadata.get_data_file_path(episode))
        paths.extend(
            metadata.root / metadata.get_video_file_path(episode, camera)
            for camera in metadata.video_keys
        )
    return paths


def stats_source(metadata, split: dict) -> dict:
    """记录训练文件的校验值，缓存变化后不能静默复用旧统计量。"""
    lengths = {item["episode_index"]: item["length"] for item in metadata.episodes}
    files = required_files(metadata, split["train_episodes"])
    return {
        "dataset": DATASET_REPO_ID, "dataset_root": str(metadata.root.resolve()),
        "dataset_revision": "local cache; exact upstream commit not established",
        "train_episodes": split["train_episodes"], "fps": metadata.fps,
        "raw_frame_count": sum(lengths[episode] for episode in split["train_episodes"]),
        "metadata_sha256": {
            name: file_sha256(metadata.root / "meta" / name)
            for name in ["info.json", "episodes.jsonl"]
        },
        "training_files_sha256": {
            str(path.relative_to(metadata.root)): file_sha256(path) for path in files
        },
        "implementation_sha256": {
            name: file_sha256(Path(inspect.getfile(component)))
            for name, component in {
                "statistics": compute_stats, "dataset": LeRobotDataset,
                "video_decoder": decode_video_frames_torchvision,
            }.items()
        },
        "packages": {name: version(name) for name in ["torch", "torchvision", "datasets", "av"]},
        "method": "LeRobot compute_stats; all raw training frames; no temporal windows or augmentation",
    }


def verified_download_revision(metadata, path: Path) -> str | None:
    """仅在当前缓存仍匹配下载清单时，才将固定 revision 写入统计量来源。"""
    if not path.exists():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    if (
        saved.get("schema_version") != 1 or saved.get("status") != "PASS"
        or saved.get("dataset") != DATASET_REPO_ID
        or saved.get("dataset_root") != str(metadata.root.resolve())
    ):
        raise ValueError("Invalid download provenance")
    revision = saved.get("revision", "")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("Download provenance must pin a full commit hash")
    files = required_files(metadata, [episode["episode_index"] for episode in metadata.episodes])
    files += [metadata.root / "meta" / name for name in ["info.json", "episodes.jsonl", "tasks.jsonl", "stats.json"]]
    for file in files:
        name = str(file.relative_to(metadata.root))
        if saved.get("files", {}).get(name, {}).get("sha256") != file_sha256(file):
            raise ValueError(f"Cached file changed after download verification: {name}")
    return revision


def validate_stats(stats: dict, config) -> None:
    features = {**config.input_features, **config.output_features}
    for key, feature in features.items():
        shape = (feature.shape[0], 1, 1) if feature.type is FeatureType.VISUAL else tuple(feature.shape)
        if key not in stats:
            raise ValueError(f"Missing statistics: {key}")
        for name in ["mean", "std", "min", "max"]:
            value = stats[key].get(name)
            if value is None or tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid statistic shape/value: {key}.{name}; expected {shape}")
        if (stats[key]["min"] > stats[key]["max"]).any() or (stats[key]["std"] < 0).any():
            raise ValueError(f"Inconsistent statistics: {key}")
        mode = config.normalization_mapping[feature.type]
        if mode is NormalizationMode.MEAN_STD and (stats[key]["std"] <= 0).any():
            raise ValueError(f"Zero standard deviation: {key}")
        if mode is NormalizationMode.MIN_MAX and (stats[key]["max"] <= stats[key]["min"]).any():
            raise ValueError(f"Degenerate min/max range: {key}")


def compute_training_stats(dataset, config, source: dict, path: Path) -> dict:
    """统计原始训练帧，返回可直接传给 DiffusionPolicy 的 Tensor 字典。"""
    if dataset.delta_timestamps is not None or dataset.image_transforms is not None:
        raise ValueError("Statistics require raw single frames without windows or augmentation")
    if dataset.episodes != source["train_episodes"] or len(dataset) != source["raw_frame_count"]:
        raise ValueError("Statistics dataset does not match its training-only provenance")
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("schema_version") != 1 or saved.get("source") != source:
            raise ValueError("Existing statistics have different provenance; choose a new output directory")
        stats = {
            key: {name: torch.tensor(value, dtype=torch.float32) for name, value in values.items()}
            for key, values in saved["dataset_stats"].items()
        }
    else:
        print(f"Computing statistics over {len(dataset)} raw training frames (two passes).", flush=True)
        computed = compute_stats(dataset, batch_size=8, num_workers=0, max_num_samples=None)
        stats = {key: computed[key] for key in {**config.input_features, **config.output_features}}
        validate_stats(stats, config)
        write_json(path, {
            "schema_version": 1, "source": source,
            "dataset_stats": {
                key: {name: value.tolist() for name, value in values.items()}
                for key, values in stats.items()
            },
        }, exclusive=True)
    validate_stats(stats, config)
    return stats


def build_dataloaders(metadata, config, split: dict, batch_size: int, seed: int) -> dict:
    """沿用 LeRobot 时间窗和末帧排除规则，不自行拼接跨 episode 序列。"""
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    loaders = {}
    for name in ["train", "val"]:
        dataset = LeRobotDataset(
            DATASET_REPO_ID, root=metadata.root, episodes=split[f"{name}_episodes"],
            delta_timestamps=build_delta_timestamps(config, metadata.fps),
            local_files_only=True,
        )
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index, drop_n_last_frames=config.drop_n_last_frames,
            shuffle=name == "train",
        )
        if not len(sampler):
            raise ValueError(f"No eligible anchors in {name} split")
        # 视频解码先只用主进程；sampler 控制顺序，不再设置 DataLoader.shuffle。
        loaders[name] = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, num_workers=0,
            drop_last=False, generator=torch.Generator().manual_seed(seed + (name == "val")),
        )
    return loaders


def verify_data_pipeline(loaders: dict, config, split: dict, stats: dict, check_batches: int) -> dict:
    """有界读取 batch；检查来源、形状、时间边界、padding 和归一化往返。"""
    if check_batches <= 0:
        raise ValueError("Check batch count must be positive")
    validate_stats(stats, config)
    features = {**config.input_features, **config.output_features}
    normalize = Normalize(features, config.normalization_mapping, stats)
    unnormalize = Unnormalize(features, config.normalization_mapping, stats)
    results = {}
    # 检查消耗的 PyTorch CPU 随机数不改变调用者后续训练的抽样序列。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(split["seed"])
        for name, loader in loaders.items():
            dataset = loader.dataset
            allowed = set(split[f"{name}_episodes"])
            expected_anchors = [
                index
                for start, end in zip(dataset.episode_data_index["from"], dataset.episode_data_index["to"])
                for index in range(int(start), int(end) - config.drop_n_last_frames)
            ]
            if list(loader.sampler.indices) != expected_anchors:
                raise ValueError(f"Unexpected eligible anchors in {name}")
            checked_samples, padded_actions, outside, errors, seen = 0, 0, {}, {}, set()
            shapes = {}
            iterator = iter(loader)
            checked_batches = min(check_batches, len(loader))
            for _ in range(checked_batches):
                batch = next(iterator)
                batch_size = batch["action"].shape[0]
                ids = batch["episode_index"].reshape(-1).tolist()
                frames = batch["frame_index"].reshape(-1).tolist()
                if len(ids) != batch_size or not set(ids) <= allowed:
                    raise ValueError(f"Batch includes a foreign episode in {name}")
                seen.update(ids)
                for key, feature in features.items():
                    steps = config.n_obs_steps if key in config.input_features else config.horizon
                    value = batch[key]
                    expected = (batch_size, steps, *feature.shape)
                    if tuple(value.shape) != expected or value.dtype != torch.float32:
                        raise ValueError(f"Unexpected tensor: {name}.{key}; expected float32 {expected}")
                    if not torch.isfinite(value).all():
                        raise ValueError(f"Nonfinite input: {name}.{key}")
                    if feature.type is FeatureType.VISUAL and ((value < 0).any() or (value > 1).any()):
                        raise ValueError(f"Image outside [0,1]: {name}.{key}")
                    shapes[key] = list(value.shape)
                for row, (episode, frame) in enumerate(zip(ids, frames)):
                    position = dataset.episodes.index(episode)
                    start = int(dataset.episode_data_index["from"][position])
                    end = int(dataset.episode_data_index["to"][position])
                    if not 0 <= frame < end - start - config.drop_n_last_frames:
                        raise ValueError(f"Ineligible anchor: episode {episode}, frame {frame}")
                    indices, padding = dataset._get_query_indices(start + frame, position)
                    for key in features:
                        if not all(start <= index < end for index in indices[key]):
                            raise ValueError("Window crosses an episode boundary")
                        if key in config.input_features and any(index > start + frame for index in indices[key]):
                            raise ValueError("Observation window includes a future frame")
                        torch.testing.assert_close(batch[f"{key}_is_pad"][row], padding[f"{key}_is_pad"])
                        # 对状态/动作核对真实表格值，不只检查窗口形状和索引范围。
                        if features[key].type is not FeatureType.VISUAL:
                            expected_values = torch.stack(list(dataset.hf_dataset.select(indices[key])[key]))
                            torch.testing.assert_close(batch[key][row], expected_values, rtol=0, atol=0)
                normalized = normalize(batch)
                restored = unnormalize(normalized)
                for key, feature in features.items():
                    if not torch.isfinite(normalized[key]).all() or not torch.isfinite(restored[key]).all():
                        raise ValueError(f"Nonfinite normalization result: {name}.{key}")
                    torch.testing.assert_close(restored[key], batch[key], rtol=1e-5, atol=1e-4)
                    error = (restored[key] - batch[key]).abs().max().item()
                    errors[key] = max(errors.get(key, 0), error)
                    if config.normalization_mapping[feature.type] is NormalizationMode.MIN_MAX:
                        count = ((normalized[key] < -1 - 1e-6) | (normalized[key] > 1 + 1e-6)).sum().item()
                        outside[key] = outside.get(key, 0) + count
                checked_samples += batch_size
                padded_actions += batch["action_is_pad"].sum().item()
            results[name] = {
                "episode_count": len(allowed), "raw_frames": len(dataset),
                "eligible_anchors": len(loader.sampler), "loader_batches": len(loader),
                "checked_batches": checked_batches, "checked_samples": checked_samples,
                "observed_episode_ids": sorted(seen), "last_checked_shapes": shapes,
                "padded_action_positions": padded_actions,
                "min_max_values_outside_unit_range": outside, "round_trip_max_error": errors,
            }
            print(f"{name}: episodes={len(allowed)}, anchors={len(loader.sampler)}, checked={checked_samples} samples; PASS", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "dp5_data")
    parser.add_argument("--dataset-root", type=Path, default=LEROBOT_HOME / DATASET_REPO_ID)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--check-batches", type=int, default=2)
    parser.add_argument("--num-threads", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.check_batches <= 0 or args.num_threads <= 0:
        parser.error("Batch size, check batch count and thread count must be positive")
    if not math.isfinite(args.val_fraction) or not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1")
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    torch.set_num_threads(args.num_threads)
    report = {"status": "ERROR", "scope": "data preparation only; no model, optimizer, training or downloads"}
    try:
        for name in ["info.json", "episodes.jsonl", "tasks.jsonl", "stats.json"]:
            if not (args.dataset_root / "meta" / name).is_file():
                raise FileNotFoundError(f"Missing cached metadata: {args.dataset_root / 'meta' / name}")
        config = PreTrainedConfig.from_pretrained(
            POLICY_REPO_ID, revision=POLICY_REVISION, local_files_only=True
        )
        metadata = LeRobotDatasetMetadata(DATASET_REPO_ID, root=args.dataset_root, local_files_only=True)
        ids = [episode["episode_index"] for episode in metadata.episodes]
        if len(ids) != metadata.total_episodes:
            raise ValueError("Metadata episode count is inconsistent")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        split = build_episode_split(ids, args.seed, args.val_fraction, args.output_dir / "split.json")
        report.update({
            "dataset_root": str(metadata.root), "policy_config": f"{POLICY_REPO_ID}@{POLICY_REVISION}",
            "split_sha256": file_sha256(args.output_dir / "split.json"),
            "train_episode_count": len(split["train_episodes"]), "val_episode_count": len(split["val_episodes"]),
            "batch_size": args.batch_size, "num_workers": 0, "num_threads": args.num_threads,
            "drop_n_last_frames": config.drop_n_last_frames,
            "do_mask_loss_for_padding": config.do_mask_loss_for_padding,
            "verification_scope": "all sampler anchors checked structurally; only bounded batches decoded",
        })
        missing = [str(path) for path in required_files(metadata, sorted(ids)) if not path.is_file()]
        if missing:
            report.update({"status": "NEEDS_DATA", "missing_files": missing})
            write_json(args.output_dir / "verification_report.json", report)
            print(f"Split saved/reused: train={len(split['train_episodes'])}, val={len(split['val_episodes'])}")
            print(f"Missing {len(missing)} cached table/video files; nothing downloaded or trained.")
            for path in missing[:4]:
                print(f"  {path}")
            print(f"Full list: {args.output_dir / 'verification_report.json'}")
            raise SystemExit(2)
        source = stats_source(metadata, split)
        revision = verified_download_revision(metadata, args.output_dir / "download_manifest.json")
        if revision is not None:
            source["dataset_revision"] = revision
        raw_train = LeRobotDataset(
            DATASET_REPO_ID, root=metadata.root, episodes=split["train_episodes"], local_files_only=True
        )
        stats = compute_training_stats(raw_train, config, source, args.output_dir / "train_stats.json")
        loaders = build_dataloaders(metadata, config, split, args.batch_size, args.seed)
        report["checks"] = verify_data_pipeline(loaders, config, split, stats, args.check_batches)
        report.update({
            "status": "PASS", "statistics_source": source,
            "train_stats_sha256": file_sha256(args.output_dir / "train_stats.json"),
            "torch_version": torch.__version__, "script_sha256": file_sha256(Path(__file__)),
        })
        write_json(args.output_dir / "verification_report.json", report)
        print(f"Data preparation: PASS; artifacts: {args.output_dir}", flush=True)
    except (ValueError, FileNotFoundError, RuntimeError, AssertionError, KeyError) as error:
        report["error"] = str(error)
        if args.output_dir.is_dir():
            write_json(args.output_dir / "verification_report.json", report)
        parser.exit(1, f"Data preparation failed: {error}\n")


if __name__ == "__main__":
    main()
