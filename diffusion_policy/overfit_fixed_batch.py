"""用新初始化的 LeRobot 策略拟合固定 Push-T batch；训练噪声、时间步与裁剪正常重采样。"""

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import default_collate

from check_training_step import check_gradients
from inspect_pusht_dataset import (
    DATASET_REPO_ID,
    POLICY_REPO_ID,
    POLICY_REVISION,
    build_delta_timestamps,
)
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.policies import PreTrainedConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_FIELDS = ["phase", "step", "loss", "loss_std", "zero_noise_mse", "gradient_norm", "seconds"]


def git_revision(directory: Path) -> str:
    """记录实际源码版本；非 Git 目录也能保留说明。"""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=directory, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def git_status(directory: Path) -> str:
    """版本号不足以描述有本地兼容补丁的源码，同时记录改动文件列表。"""
    result = subprocess.run(
        ["git", "status", "--short"], cwd=directory, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def write_json(path: Path, value: dict) -> None:
    """产物均写入专属输出目录，不修改源码或已有实验。"""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def load_fixed_batch(config, args):
    """只读取一次原始观测/专家动作窗口，不在训练循环里重新取样。"""
    metadata = LeRobotDatasetMetadata(DATASET_REPO_ID, local_files_only=True)
    if not 0 <= args.episode < metadata.total_episodes:
        raise ValueError("Episode is outside dataset metadata")
    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        episodes=[args.episode],
        delta_timestamps=build_delta_timestamps(config, metadata.fps),
        local_files_only=True,
    )
    start = int(dataset.episode_data_index["from"][0])
    end = int(dataset.episode_data_index["to"][0])
    if any(anchor < 0 or anchor >= end - start for anchor in args.anchors):
        raise ValueError(f"Anchors must be in [0, {end - start - 1}]")
    keys = list(config.input_features) + ["action", "action_is_pad"]
    samples = [dataset[start + anchor] for anchor in args.anchors]
    batch = default_collate([{key: sample[key] for key in keys} for sample in samples])
    batch = {key: value.to(args.device) for key, value in batch.items()}
    for key, value in batch.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Nonfinite batch tensor: {key}")
        print(f"input {key}: {tuple(value.shape)}, {value.dtype}", flush=True)
    return metadata, batch


@torch.no_grad()
def evaluate(policy, batch, draws: int, seed: int) -> dict:
    """同一个专家 batch 上多次采样；固定评估序列，不污染训练 RNG 状态。"""
    was_training = policy.training
    scheduler = policy.diffusion.noise_scheduler
    original_add_noise = scheduler.add_noise
    losses, zero_losses = [], []

    def observe_noise(clean, noise, timesteps):
        # 恒零预测器的 MSE 仅为诊断参照；不替换 LeRobot 的加噪或 loss。
        zero_losses.append(noise.square().mean().item())
        return original_add_noise(clean, noise, timesteps)

    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        policy.eval()  # 固定中心裁剪；训练循环仍使用随机裁剪。
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            with patch.object(scheduler, "add_noise", side_effect=observe_noise):
                for _ in range(draws):
                    loss = policy(batch)["loss"]
                    if loss.ndim != 0 or not torch.isfinite(loss):
                        raise RuntimeError("Evaluation loss must be a finite scalar")
                    losses.append(loss.item())
    finally:
        policy.train(was_training)
    if len(zero_losses) != draws:
        raise RuntimeError("Expected one forward-noising call per evaluation draw")
    return {
        "loss": statistics.mean(losses),
        "loss_std": statistics.pstdev(losses),
        "zero_noise_mse": statistics.mean(zero_losses),
    }


@torch.no_grad()
def reload_probe(policy, batch, seed: int):
    """仅对重载检查固定输入任务，记录 UNet 输出与 loss，不改变训练采样。"""
    captured = {}

    def record_prediction(module, positional, output):
        captured["prediction"] = output.detach().cpu().clone()

    was_training = policy.training
    handle = policy.diffusion.unet.register_forward_hook(record_prediction)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        policy.eval()
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            loss = policy(batch)["loss"].detach().cpu()
        return captured["prediction"], loss
    finally:
        handle.remove()
        policy.train(was_training)


def parameter_changes(policy, initial: dict[str, torch.Tensor]) -> dict[str, float]:
    """用参数差的 L2 范数分别检查视觉编码器和 UNet，不以梯度代替更新证据。"""
    squared = {"rgb_encoder": 0.0, "unet": 0.0}
    for name, parameter in policy.diffusion.named_parameters():
        if not parameter.requires_grad:
            continue
        delta = parameter.detach().cpu() - initial[name]
        if not torch.isfinite(delta).all():
            raise RuntimeError(f"Nonfinite updated parameter: {name}")
        group = name.split(".", 1)[0]
        squared[group] += delta.square().sum(dtype=torch.float64).item()
    changes = {name: math.sqrt(value) for name, value in squared.items()}
    if not all(value > 0 for value in changes.values()):
        raise RuntimeError(f"Expected both modules to update: {changes}")
    return changes


def plot_metrics(rows: list[dict], path: Path) -> None:
    """区分随机训练 loss 与同 batch 的中心裁剪评估，不标成 validation。"""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    train = [row for row in rows if row["phase"] == "train"]
    evaluations = [row for row in rows if row["phase"] == "eval"]
    figure = Figure(figsize=(11, 4), constrained_layout=True)
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2)
    losses = [row["loss"] for row in train]
    left.plot([row["step"] for row in train], losses, alpha=0.4, label="raw training MSE")
    smoothed = [statistics.mean(losses[max(0, i - 19):i + 1]) for i in range(len(losses))]
    left.plot([row["step"] for row in train], smoothed, label="moving mean (up to 20 steps)")
    left.set(title="Fixed expert batch; random noise/timestep/crop", xlabel="Optimizer step", ylabel="MSE")
    left.legend()
    right.errorbar(
        [row["step"] for row in evaluations], [row["loss"] for row in evaluations],
        yerr=[row["loss_std"] for row in evaluations], marker="o", label="mean ± draw std",
    )
    right.set(title="Same-batch evaluation; center crop", xlabel="Optimizer step", ylabel="MSE")
    right.legend()
    figure.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--anchors", type=int, nargs="+", default=[79, 80])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-draws", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-loss-ratio", type=float, default=0.5,
                        help="预先声明的学习检查阈值：最终/初始评估 MSE；不要求 loss 为零。")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    for name in ["steps", "eval_draws", "eval_every", "log_every", "num_threads"]:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.max_loss_ratio) or not 0 < args.max_loss_ratio < 1:
        parser.error("--max-loss-ratio must be between 0 and 1")
    if args.learning_rate is not None and (
        not math.isfinite(args.learning_rate) or args.learning_rate <= 0
    ):
        parser.error("--learning-rate must be finite and positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu if intended")

    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)  # 训练只在初始化前设置一次，不在每轮重置。
    output = args.output_dir or (
        PROJECT_ROOT / "outputs" / "dp5_fixed_batch" /
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    )
    output = output.resolve()
    if output.exists():
        parser.error(f"Output directory already exists; choose a new path: {output}")
    output.mkdir(parents=True, exist_ok=False)
    print(f"output: {output}", flush=True)

    config = PreTrainedConfig.from_pretrained(
        POLICY_REPO_ID, revision=POLICY_REVISION, local_files_only=True
    )
    if config.prediction_type != "epsilon" or config.pretrained_backbone_weights is not None:
        raise ValueError("This diagnostic requires epsilon targets and a freshly initialized backbone")
    if args.learning_rate is not None:
        config.optimizer_lr = args.learning_rate
    metadata, batch = load_fixed_batch(config, args)
    original_batch = {key: value.detach().cpu().clone() for key, value in batch.items()}
    policy = DiffusionPolicy(config, dataset_stats=metadata.stats).to(args.device)
    parameters = [p for p in policy.get_optim_params() if p.requires_grad]
    optimizer_config = config.get_optimizer_preset()
    optimizer = optimizer_config.build(parameters)
    initial_parameters = {
        name: p.detach().cpu().clone()
        for name, p in policy.diffusion.named_parameters() if p.requires_grad
    }

    import lerobot
    lerobot_root = Path(lerobot.__file__).resolve().parents[1]
    packages = {}
    for name in ["torch", "torchvision", "datasets", "diffusers", "safetensors", "av"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unknown"
    write_json(output / "manifest.json", {
        "arguments": vars(args), "output_dir": str(output),
        "initialization": "fresh network; no pretrained policy/backbone weights",
        "dataset": DATASET_REPO_ID, "config_repository": POLICY_REPO_ID,
        "config_revision": POLICY_REVISION, "fps": metadata.fps,
        "statistics": "repository meta.stats; not recomputed or verified as train-only",
        "dataset_revision": "local cache; exact upstream commit not established",
        "training_randomness": "fixed raw expert batch; resampled noise, timestep and training crop",
        "evaluation": "same expert batch; center crop; reproducible independent noise/timestep draws",
        "optimizer": asdict(optimizer_config), "lr_schedule": "constant; no EMA",
        "trainable_parameters": sum(p.numel() for p in parameters),
        "project_commit": git_revision(PROJECT_ROOT), "lerobot_commit": git_revision(lerobot_root),
        "project_git_status": git_status(PROJECT_ROOT), "lerobot_git_status": git_status(lerobot_root),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(), "packages": packages, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        "scope": "fixed-batch learning diagnostic, not validation/generalization or closed-loop evaluation",
    })

    rows = []
    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writeheader()

        def record(row):
            rows.append(row)
            writer.writerow(row)
            file.flush()

        initial_eval = evaluate(policy, batch, args.eval_draws, args.eval_seed)
        record({"phase": "eval", "step": 0, **initial_eval})
        print(f"initial same-batch MSE: {initial_eval['loss']:.6f}", flush=True)
        policy.train()
        started = time.perf_counter()
        for step in range(1, args.steps + 1):
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = policy(batch)["loss"]  # 原实现每轮正常生成新的 epsilon 和 k。
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite/scalar training loss at step {step}")
            loss.backward()
            if step in {1, args.steps}:
                check_gradients(policy.diffusion.rgb_encoder, "visual encoder")
                check_gradients(policy.diffusion.unet, "denoising UNet")
            # build() 不会自动执行配置中的梯度裁剪；这里明确调用。
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, optimizer_config.grad_clip_norm, error_if_nonfinite=True
            )
            optimizer.step()
            record({
                "phase": "train", "step": step, "loss": loss.item(),
                "gradient_norm": norm.item(), "seconds": time.perf_counter() - step_started,
            })
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                print(f"step={step:04d} loss={loss.item():.6f} gradient_norm={norm.item():.4f}", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                final_eval = evaluate(policy, batch, args.eval_draws, args.eval_seed)
                record({"phase": "eval", "step": step, **final_eval})
                print(f"  same-batch MSE={final_eval['loss']:.6f} ± {final_eval['loss_std']:.6f}", flush=True)
        training_seconds = time.perf_counter() - started

    changes = parameter_changes(policy, initial_parameters)
    del initial_parameters
    for key, original in original_batch.items():
        torch.testing.assert_close(batch[key].cpu(), original, rtol=0, atol=0)
    before_prediction, before_loss = reload_probe(policy, batch, args.eval_seed + 1)
    optimizer.zero_grad(set_to_none=True)
    del optimizer, loss
    checkpoint = output / "checkpoint"
    policy.cpu().save_pretrained(checkpoint)
    # 当前 draccus/旧版 LeRobot 的 dump 会遗漏 ChoiceRegistry 的类型字段。
    # 只补齐本次输出的配置，不修改第三方保存/加载实现。
    config_path = checkpoint / "config.json"
    saved_config = json.loads(config_path.read_text())
    saved_config.setdefault("type", policy.config.type)
    write_json(config_path, saved_config)
    reloaded = DiffusionPolicy.from_pretrained(
        checkpoint, map_location=args.device, local_files_only=True, strict=True
    )
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, reloaded.state_dict()[key].cpu(), rtol=0, atol=0)
    after_prediction, after_loss = reload_probe(reloaded, batch, args.eval_seed + 1)
    torch.testing.assert_close(before_prediction, after_prediction, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(before_loss, after_loss, rtol=1e-5, atol=1e-6)
    ratio = final_eval["loss"] / initial_eval["loss"] if initial_eval["loss"] > 0 else math.inf
    summary = {
        "steps": args.steps, "initial_evaluation": initial_eval, "final_evaluation": final_eval,
        "loss_ratio": ratio, "max_loss_ratio": args.max_loss_ratio,
        "learning_check_passed": ratio <= args.max_loss_ratio,
        "parameter_change_l2": changes, "raw_batch_unchanged": True,
        "checkpoint_reload_passed": True,
        "reload_prediction_max_error": (before_prediction - after_prediction).abs().max().item(),
        "training_and_evaluation_seconds": training_seconds,
        "checkpoint": str(checkpoint),
        "scope": "same-batch diagnostic only; no held-out validation or rollout",
    }
    write_json(output / "summary.json", summary)
    plot_metrics(rows, output / "loss_curve.png")
    print(f"parameter change L2: {changes}", flush=True)
    print("checkpoint reload: PASS (weights, statistics, prediction and loss)", flush=True)
    print(
        f"learning criterion: {'PASS' if summary['learning_check_passed'] else 'NOT MET'}; "
        f"final/initial MSE={ratio:.4f}, required<={args.max_loss_ratio:.4f}", flush=True,
    )
    print("Numerical/update/reload checks passed; loss criterion is recorded separately.", flush=True)


if __name__ == "__main__":
    main()
