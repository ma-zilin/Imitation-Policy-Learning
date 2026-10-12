"""DP5-B：真实 DataLoader 短程训练、固定验证窗口和完整训练状态恢复（离线）。"""

import argparse
import csv
import json
import math
import random
import shutil
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from check_training_step import check_gradients
from inspect_pusht_dataset import POLICY_REPO_ID, POLICY_REVISION
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.policies import PreTrainedConfig
from overfit_fixed_batch import evaluate, git_revision, git_status
from prepare_training_data import (
    DATASET_REPO_ID, PROJECT_ROOT, build_dataloaders, file_sha256,
    stats_source, validate_split, validate_stats, verified_download_revision, write_json,
)


FIELDS = ["phase", "step", "loss", "zero_noise_mse", "gradient_norm", "learning_rate",
          "batch_size", "seconds", "samples_per_second", "cuda_allocated_mib",
          "cuda_reserved_mib", "cuda_peak_mib", "episode_ids", "frame_indices"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def capture_rng():
    """NumPy 状态转成普通对象，允许 weights_only=True 安全读取本地状态。"""
    state = np.random.get_state()
    return {
        "python": random.getstate(), "numpy": (state[0], state[1].tolist(), *state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    algorithm, values, position, has_gauss, cached = state["numpy"]
    np.random.set_state((algorithm, np.asarray(values, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from saved training state")
        torch.cuda.set_rng_state_all(state["cuda"])


class ResumableBatches:
    """保存 epoch 排列和已消费样本数；不靠重新解码旧 batch 来追赶位置。

    沿用 LeRobot 的合格起点，只把 shuffle 改成独立 RNG。workers=0 无预取；
    每个 batch 内窗口仍按原时间顺序读取，末尾不足 batch_size 也保留。
    """

    def __init__(self, loader, seed, state=None):
        if loader.num_workers != 0:
            raise ValueError("Exact cursor recovery requires num_workers=0")
        self.indices = list(loader.sampler.indices)
        self.seed, self.epoch, self.cursor = seed, 0, 0
        self.batch_size = loader.batch_size
        self.order = self.make_order()
        if state is not None:
            if state["seed"] != seed or state["batch_size"] != self.batch_size:
                raise ValueError("Saved data-stream configuration differs")
            self.epoch, self.cursor, self.order = state["epoch"], state["cursor"], state["order"]
            if (self.epoch < 0 or not 0 <= self.cursor <= len(self.indices)
                    or self.order != self.make_order()
                    or (self.cursor != len(self.indices) and self.cursor % self.batch_size)):
                raise ValueError("Invalid saved sampler order/cursor")
        self.dataset = loader.dataset
        self.iterator = None

    def make_order(self):
        positions = torch.randperm(
            len(self.indices), generator=torch.Generator().manual_seed(self.seed + self.epoch)
        ).tolist()
        return [self.indices[index] for index in positions]

    def state_dict(self):
        return {"seed": self.seed, "epoch": self.epoch, "cursor": self.cursor,
                "order": self.order.copy(), "batch_size": self.batch_size}

    def next(self):
        if self.cursor == len(self.order):
            self.epoch += 1
            self.cursor = 0
            self.order = self.make_order()
            self.iterator = None
        if self.iterator is None:
            # DataLoader 初始化 iterator 会取一个 base_seed；使用独立 generator。
            loader = DataLoader(
                self.dataset, batch_size=self.batch_size, sampler=self.order[self.cursor:],
                num_workers=0, drop_last=False,
                generator=torch.Generator().manual_seed(self.seed + self.epoch),
            )
            self.iterator = iter(loader)
        batch = next(self.iterator)
        self.cursor += batch["action"].shape[0]
        return batch


def policy_batch(batch, config, device):
    """仅传策略所需字段，额外的 episode/frame 信息留在日志中。"""
    keys = [*config.input_features, "action", "action_is_pad"]
    result = {key: batch[key].to(device) for key in keys}
    for key, value in result.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Nonfinite batch: {key}")
    return result


def validation(policy, batches, draws, seed):
    """固定窗口、中心裁剪、固定独立噪声序列；按样本数汇总而非平均 batch。"""
    count, loss, zero = 0, 0.0, 0.0
    for index, batch in enumerate(batches):
        metrics = evaluate(policy, batch, draws, seed + index)
        size = batch["action"].shape[0]
        count += size
        loss += size * metrics["loss"]
        zero += size * metrics["zero_noise_mse"]
    if count == 0:
        raise ValueError("Empty validation subset")
    return {"loss": loss / count, "zero_noise_mse": zero / count, "batch_size": count}


def load_prepared(directory, config, batch_size, seed):
    """检查既有数据证据与当前文件，不重新划分、不重新统计、不联网。"""
    report = json.loads((directory / "verification_report.json").read_text())
    split = json.loads((directory / "split.json").read_text())
    saved_stats = json.loads((directory / "train_stats.json").read_text())
    if report.get("status") != "PASS":
        raise ValueError("Run data preparation successfully before training")
    if report["policy_config"] != f"{POLICY_REPO_ID}@{POLICY_REVISION}":
        raise ValueError("Prepared data uses another policy configuration")
    for name, field in [("split.json", "split_sha256"), ("train_stats.json", "train_stats_sha256")]:
        if file_sha256(directory / name) != report[field]:
            raise ValueError(f"Prepared artifact changed: {name}")
    metadata = LeRobotDatasetMetadata(
        DATASET_REPO_ID, root=Path(report["dataset_root"]), local_files_only=True
    )
    validate_split(split, [item["episode_index"] for item in metadata.episodes])
    source = stats_source(metadata, split)
    revision = verified_download_revision(metadata, directory / "download_manifest.json")
    if revision is None:
        raise ValueError("Verified full dataset revision is required")
    source["dataset_revision"] = revision
    if saved_stats.get("schema_version") != 1 or saved_stats["source"] != source:
        raise ValueError("Statistics provenance differs from current training data/code")
    stats = {key: {name: torch.tensor(value, dtype=torch.float32) for name, value in values.items()}
             for key, values in saved_stats["dataset_stats"].items()}
    validate_stats(stats, config)
    loaders = build_dataloaders(metadata, config, split, batch_size, seed)
    return loaders, stats, {
        "dataset_revision": revision, "split_sha256": report["split_sha256"],
        "train_stats_sha256": report["train_stats_sha256"], "dataset_root": str(metadata.root),
    }


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def save_checkpoint(path, policy, optimizer, stream, step, contract, initial, final):
    """发布前写 COMPLETE；不覆盖旧 checkpoint，异常中断留下的目录不会被恢复。"""
    required_bytes = 4 * sum(value.numel() * value.element_size() for value in policy.state_dict().values())
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path.parent).free < required_bytes:
        raise RuntimeError("Insufficient disk space for model and Adam checkpoint; older outputs preserved")
    path.mkdir(parents=True, exist_ok=False)
    rng = capture_rng()
    policy.save_pretrained(path / "policy")
    config_path = path / "policy/config.json"
    serialized = json.loads(config_path.read_text())
    serialized.setdefault("type", policy.config.type)
    write_json(config_path, serialized)
    torch.save({
        "schema_version": 1, "step": step, "optimizer": cpu_tree(optimizer.state_dict()),
        "rng": rng, "stream": stream.state_dict(), "contract": contract,
        "initial_validation": initial, "last_validation": final,
    }, path / "training_state.pt")
    write_json(path / "COMPLETE.json", {
        "step": step, "sha256": {
            str(file.relative_to(path)): file_sha256(file)
            for file in [path / "training_state.pt", config_path, path / "policy/model.safetensors"]
        },
    }, exclusive=True)
    restore_rng(rng)


def verify_checkpoint(path):
    """校验完整清单；rollout 无需把大体积 Adam 状态读入内存。"""
    marker = json.loads((path / "COMPLETE.json").read_text())
    names = {"training_state.pt", "policy/config.json", "policy/model.safetensors"}
    if set(marker["sha256"]) != names:
        raise ValueError("Incomplete checkpoint checksum inventory")
    for name, digest in marker["sha256"].items():
        if file_sha256(path / name) != digest:
            raise ValueError(f"Checkpoint changed: {name}")
    return marker


def load_training_state(path):
    """只恢复本工具生成且文件完整的 checkpoint；不接受任意 pickle。"""
    marker = verify_checkpoint(path)
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
    if state.get("schema_version") != 1 or state["step"] != marker["step"]:
        raise ValueError("Unsupported or inconsistent training state")
    return state


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "outputs/dp5_data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="完整 checkpoint 目录；输出写入新目录。")
    parser.add_argument("--steps", type=int, default=200, help="包括恢复前步数的总目标步数。")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-batches", type=int, default=2)
    parser.add_argument("--eval-draws", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    for key in ["steps", "batch_size", "val_batches", "eval_draws", "eval_every",
                "checkpoint_every", "log_every", "num_threads"]:
        if getattr(args, key) <= 0:
            parser.error(f"{key} must be positive")
    if args.seed < 0 or args.seed >= 2**32 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Invalid seed or learning rate")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(args.num_threads)
    seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = PreTrainedConfig.from_pretrained(
        POLICY_REPO_ID, revision=POLICY_REVISION, local_files_only=True
    )
    if config.prediction_type != "epsilon" or config.pretrained_backbone_weights is not None:
        raise ValueError("Requires fresh backbone and epsilon prediction")
    config.optimizer_lr = args.learning_rate
    loaders, stats, provenance = load_prepared(args.data_dir.resolve(), config, args.batch_size, args.seed)
    # 固定验证窗口，只在开始时读取，绝不交给训练优化器。
    val_iterator = iter(loaders["val"])
    raw_validation = [next(val_iterator) for _ in range(min(args.val_batches, len(loaders["val"])))]
    batches = [policy_batch(batch, config, args.device) for batch in raw_validation]
    import lerobot
    import inspect
    source_files = [Path(__file__), Path(__file__).with_name("prepare_training_data.py"),
                    Path(__file__).with_name("overfit_fixed_batch.py"),
                    Path(inspect.getfile(DiffusionPolicy)),
                    Path(inspect.getfile(config.__class__))]
    contract = {
        **provenance, "config": json.loads(json.dumps(asdict(config))),
        "seed": args.seed, "eval_seed": args.eval_seed, "batch_size": args.batch_size,
        "val_batches": args.val_batches, "eval_draws": args.eval_draws,
        "device": args.device, "num_threads": args.num_threads,
        "packages": {name: version(name) for name in
                     ["torch", "torchvision", "diffusers", "datasets", "av", "safetensors"]},
        "source_sha256": {str(path): file_sha256(path) for path in source_files},
        "scheduler": "constant learning rate; no scheduler", "ema": False, "amp": False,
        "validation_anchors": [{key: batch[key].tolist() for key in ["episode_index", "frame_index"]}
                               for batch in raw_validation],
    }
    state = load_training_state(args.resume.resolve()) if args.resume else None
    if state and (state["contract"] != contract or args.steps <= state["step"]):
        raise ValueError("Resume settings/source/data differ, or target step is not greater")
    output = (args.output_dir or PROJECT_ROOT / "outputs/dp5_smoke" /
              datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    # 每个完整 checkpoint 约为权重的三倍（模型 + Adam 两份动量）。
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 4 * 1024**3:
        raise RuntimeError("Need at least 4 GiB free for a full checkpoint")
    output.mkdir()
    write_json(output / "manifest.json", {
        "arguments": vars(args) | {"data_dir": str(args.data_dir), "output_dir": str(output),
                                  "resume": str(args.resume) if args.resume else None},
        "contract": contract, "project_commit": git_revision(PROJECT_ROOT),
        "project_dirty": git_status(PROJECT_ROOT),
        "lerobot_commit": git_revision(Path(lerobot.__file__).resolve().parents[1]),
        "lerobot_dirty": git_status(Path(lerobot.__file__).resolve().parents[1]),
        "initialization": "resume" if state else "fresh network; no pretrained weights",
        "scope": "bounded smoke test; validation subset, not full validation or policy reproduction",
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
    }, exclusive=True)
    print(f"output: {output}", flush=True)
    policy = (DiffusionPolicy.from_pretrained(
        args.resume / "policy", map_location=args.device, local_files_only=True, strict=True
    ) if state else DiffusionPolicy(config, dataset_stats=stats).to(args.device))
    parameters = list(policy.get_optim_params())
    preset = config.get_optimizer_preset()
    optimizer = preset.build(parameters)
    stream = ResumableBatches(loaders["train"], args.seed, state["stream"] if state else None)
    start_step = state["step"] if state else 0
    initial = state["initial_validation"] if state else validation(policy, batches, args.eval_draws, args.eval_seed)
    if state:
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])  # 必须在模型/loader/optimizer 初始化之后恢复。
        del state  # 释放 CPU 上大体积 Adam 状态副本。
    rows = []
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with (output / "metrics.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()

        def record(row):
            if any(isinstance(value, float) and not math.isfinite(value) for value in row.values()):
                raise RuntimeError("Nonfinite metric")
            rows.append(row)
            writer.writerow(row)
            file.flush()

        current = validation(policy, batches, args.eval_draws, args.eval_seed) if args.resume else initial
        record({"phase": "validation", "step": start_step, **current})
        print(f"start step={start_step}, held-out subset MSE={current['loss']:.6f}", flush=True)
        policy.train()
        for step in range(start_step + 1, args.steps + 1):
            synchronize(args.device)
            started = time.perf_counter()
            raw = stream.next()
            batch = policy_batch(raw, config, args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = policy(batch)["loss"]
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise RuntimeError(f"Invalid training loss at {step}")
            loss.backward()
            if step in {start_step + 1, args.steps}:
                check_gradients(policy.diffusion.rgb_encoder, "visual encoder")
                check_gradients(policy.diffusion.unet, "denoising UNet")
            norm = torch.nn.utils.clip_grad_norm_(parameters, preset.grad_clip_norm, error_if_nonfinite=True)
            optimizer.step()
            synchronize(args.device)
            seconds = time.perf_counter() - started
            size = batch["action"].shape[0]
            row = {
                "phase": "train", "step": step, "loss": loss.item(), "gradient_norm": norm.item(),
                "learning_rate": optimizer.param_groups[0]["lr"], "batch_size": size,
                "seconds": seconds, "samples_per_second": size / seconds,
                "episode_ids": json.dumps(raw["episode_index"].tolist()),
                "frame_indices": json.dumps(raw["frame_index"].tolist()),
            }
            if args.device == "cuda":
                row.update({"cuda_allocated_mib": torch.cuda.memory_allocated() / 1024**2,
                            "cuda_reserved_mib": torch.cuda.memory_reserved() / 1024**2,
                            "cuda_peak_mib": torch.cuda.max_memory_allocated() / 1024**2})
            record(row)
            if step == start_step + 1 or step % args.log_every == 0 or step == args.steps:
                print(f"step={step:04d} loss={loss.item():.6f} time={seconds:.3f}s", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                current = validation(policy, batches, args.eval_draws, args.eval_seed)
                record({"phase": "validation", "step": step, **current})
                print(f"  held-out subset MSE={current['loss']:.6f}", flush=True)
            if step % args.checkpoint_every == 0 or step == args.steps:
                optimizer.zero_grad(set_to_none=True)
                save_checkpoint(output / "checkpoints" / f"step_{step:06d}", policy, optimizer,
                                stream, step, contract, initial, current)
    training = [row for row in rows if row["phase"] == "train"]
    elapsed = sum(row["seconds"] for row in training)
    write_json(output / "summary.json", {
        "status": "PASS", "start_step": start_step, "final_step": args.steps,
        "optimizer_steps_this_process": args.steps - start_step,
        "initial_validation": initial, "final_validation": current,
        "training_seconds_including_data": elapsed,
        "samples_per_second": sum(row["batch_size"] for row in training) / elapsed,
        "peak_cuda_mib": torch.cuda.max_memory_allocated() / 1024**2 if args.device == "cuda" else None,
        "checkpoint": str(output / "checkpoints" / f"step_{args.steps:06d}"),
        "scope": "finite training/validation and checkpoint write only; resume/rollout require separate checks",
    }, exclusive=True)
    print("Training smoke: PASS; resume and rollout are separate checks.", flush=True)


if __name__ == "__main__":
    main()
