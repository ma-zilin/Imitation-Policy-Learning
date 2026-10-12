"""独立进程短测完整训练更新的显存和吞吐；不保存权重、不评估模型质量。"""

import argparse
import csv
import inspect
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from inspect_pusht_dataset import POLICY_REPO_ID, POLICY_REVISION
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.policies import PreTrainedConfig
from overfit_fixed_batch import git_revision, git_status
from train_smoke import (
    PROJECT_ROOT, ResumableBatches, file_sha256, load_prepared, policy_batch,
    seed_all, synchronize, write_json,
)


STEP_FIELDS = ["step", "phase", "batch_size", "loss", "gradient_norm", "seconds",
               "allocated_mib", "reserved_mib", "phase_peak_allocated_mib",
               "phase_peak_reserved_mib", "free_mib"]
SUMMARY_FIELDS = ["batch_size", "status", "completed_updates", "measured_updates",
                  "samples_per_second", "mean_step_seconds", "median_step_seconds",
                  "p95_step_seconds", "peak_allocated_mib", "peak_reserved_mib",
                  "steady_peak_allocated_mib", "minimum_free_mib", "device_total_mib",
                  "headroom_ok", "error"]


def validate_args(args):
    if (not args.batch_sizes or any(size <= 0 for size in args.batch_sizes)
            or len(set(args.batch_sizes)) != len(args.batch_sizes)):
        raise ValueError("Batch sizes must be distinct positive integers")
    if args.steps <= args.warmup or args.warmup < 1:
        raise ValueError("Require steps > warmup >= 1 so Adam is initialized before measurement")
    if args.num_threads <= 0 or args.worker_timeout <= 0 or not 0 <= args.seed < 2**32:
        raise ValueError("Invalid thread count, worker timeout or seed")
    if not math.isfinite(args.min_headroom_fraction) or not 0 < args.min_headroom_fraction < 1:
        raise ValueError("Headroom fraction must be in (0,1)")


def gpu_snapshot():
    """记录外部显存占用/温度；不更改功耗、时钟或其他进程。"""
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv",
    ], capture_output=True, text=True, timeout=10)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def memory_snapshot(device):
    if device == "cpu":
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_mib": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mib": torch.cuda.memory_reserved() / 1024**2,
        "phase_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "phase_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "free_mib": free / 1024**2, "device_total_mib": total / 1024**2,
    }


def run_updates(policy, optimizer, parameters, stream, config, steps, warmup, device, clip_norm, record):
    """同步计时包含真实读取、搬运、forward/backward、裁剪及 Adam 更新。

    loss 转标量、显存查询、CSV 写入在计时之后；无验证、保存或额外梯度诊断。
    每步裁剪检查有限梯度，首步另检查所有优化参数获得梯度。
    """
    policy.train()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, steps + 1):
        if step == warmup + 1 and device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        synchronize(device)
        started = time.perf_counter()
        batch = policy_batch(stream.next(), config, device)
        optimizer.zero_grad(set_to_none=True)
        loss = policy(batch)["loss"]
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise RuntimeError(f"Invalid training loss at step {step}")
        loss.backward()
        if step == 1 and any(parameter.grad is None for parameter in parameters):
            raise RuntimeError("Some optimized parameters did not receive gradients")
        norm = torch.nn.utils.clip_grad_norm_(parameters, clip_norm, error_if_nonfinite=True)
        optimizer.step()
        synchronize(device)
        seconds = time.perf_counter() - started
        row = {
            "step": step, "phase": "warmup" if step <= warmup else "measured",
            "batch_size": batch["action"].shape[0], "loss": loss.item(),
            "gradient_norm": norm.item(), "seconds": seconds,
            **{key: value for key, value in memory_snapshot(device).items() if key != "device_total_mib"},
        }
        if seconds <= 0 or any(isinstance(value, float) and not math.isfinite(value) for value in row.values()):
            raise RuntimeError("Nonfinite metric or nonpositive duration")
        record(row)


def summarize_rows(rows, batch_size, steps, warmup):
    if len(rows) != steps or [row["step"] for row in rows] != list(range(1, steps + 1)):
        raise ValueError("Missing/reordered update records")
    if any(row["batch_size"] != batch_size for row in rows):
        raise ValueError("A short tail batch would confound this resource comparison")
    measured = [row for row in rows if row["phase"] == "measured"]
    if len(measured) != steps - warmup:
        raise ValueError("Unexpected measured update count")
    durations = [row["seconds"] for row in measured]
    if any(not math.isfinite(value) or value <= 0 for value in durations):
        raise ValueError("Invalid timing record")
    result = {
        "status": "PASS", "batch_size": batch_size, "completed_updates": len(rows),
        "measured_updates": len(measured),
        "samples_per_second": sum(row["batch_size"] for row in measured) / sum(durations),
        "mean_step_seconds": statistics.mean(durations),
        "median_step_seconds": statistics.median(durations),
        "p95_step_seconds": float(np.percentile(durations, 95)),
    }
    if "allocated_mib" in rows[0]:
        result.update({
            "peak_allocated_mib": max(row["phase_peak_allocated_mib"] for row in rows),
            "peak_reserved_mib": max(row["phase_peak_reserved_mib"] for row in rows),
            "steady_peak_allocated_mib": max(row["phase_peak_allocated_mib"] for row in measured),
            # 查询点在每次更新后，不是对全机空闲显存的连续采样。
            "minimum_free_mib": min(row["free_mib"] for row in rows),
        })
    return result


def resource_candidate(results, fraction):
    """余量是显式启发式，不能保证加入 EMA/验证后仍够用，也不比较 loss。"""
    for result in results:
        total, free = result.get("device_total_mib"), result.get("minimum_free_mib")
        result["headroom_ok"] = bool(result["status"] == "PASS" and total and free is not None
                                     and free >= fraction * total)
    candidates = [result for result in results if result["headroom_ok"]]
    if not candidates:
        return None
    return max(candidates, key=lambda result: result["samples_per_second"])["batch_size"]


def run_worker(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    batch_size = args.worker_batch_size
    rows = []
    report = {"batch_size": batch_size, "status": "ERROR", "completed_updates": 0,
              "scope": "resource probe only; temporary parameter updates, no checkpoint or rollout"}
    try:
        torch.set_num_threads(args.num_threads)
        seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        before = gpu_snapshot() if args.device == "cuda" else "CPU test"
        config = PreTrainedConfig.from_pretrained(
            POLICY_REPO_ID, revision=POLICY_REVISION, local_files_only=True
        )
        if config.prediction_type != "epsilon" or config.pretrained_backbone_weights is not None:
            raise ValueError("Requires fresh network/backbone and epsilon targets")
        loaders, stats, provenance = load_prepared(args.data_dir.resolve(), config, batch_size, args.seed)
        if len(loaders["train"].sampler) < batch_size * args.steps:
            raise ValueError("Probe must fit before the first epoch's partial tail batch")
        import lerobot
        source_files = [Path(__file__), Path(__file__).with_name("train_smoke.py"),
                        Path(__file__).with_name("prepare_training_data.py"),
                        Path(inspect.getfile(DiffusionPolicy)), Path(inspect.getfile(config.__class__))]
        write_json(output / "manifest.json", {
            "batch_size": batch_size, "steps": args.steps, "warmup": args.warmup,
            "seed": args.seed, "device": args.device, "num_threads": args.num_threads,
            "config": json.loads(json.dumps(asdict(config))), "data": provenance,
            "source_sha256": {str(path): file_sha256(path) for path in source_files},
            "packages": {name: version(name) for name in
                         ["torch", "torchvision", "datasets", "diffusers", "av", "safetensors"]},
            "project_commit": git_revision(PROJECT_ROOT), "project_dirty": git_status(PROJECT_ROOT),
            "lerobot_commit": git_revision(Path(lerobot.__file__).resolve().parents[1]),
            "lerobot_dirty": git_status(Path(lerobot.__file__).resolve().parents[1]),
            "gpu_before": before, "scheduler": "constant LR", "ema": False, "amp": False,
            "validation": False, "checkpoint": False, "rollout": False,
        }, exclusive=True)
        policy = DiffusionPolicy(config, dataset_stats=stats).to(args.device)
        parameters = list(policy.get_optim_params())
        preset = config.get_optimizer_preset()
        optimizer = preset.build(parameters)
        stream = ResumableBatches(loaders["train"], args.seed)
        with (output / "steps.csv").open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=STEP_FIELDS)
            writer.writeheader()

            def record(row):
                rows.append(row)
                writer.writerow(row)
                file.flush()

            run_updates(policy, optimizer, parameters, stream, config, args.steps, args.warmup,
                        args.device, preset.grad_clip_norm, record)
        report.update(summarize_rows(rows, batch_size, args.steps, args.warmup))
        if args.device == "cuda":
            report["device_total_mib"] = memory_snapshot(args.device)["device_total_mib"]
        report["gpu_after"] = gpu_snapshot() if args.device == "cuda" else "CPU test"
    except torch.cuda.OutOfMemoryError as error:
        report.update({"status": "OOM", "error": str(error), "completed_updates": len(rows)})
    except Exception as error:
        report.update({"status": "ERROR", "error": f"{type(error).__name__}: {error}",
                       "completed_updates": len(rows)})
    write_json(output / "result.json", report, exclusive=True)
    print(json.dumps(report, indent=2), flush=True)
    return {"PASS": 0, "OOM": 2, "ERROR": 3}[report["status"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "outputs/dp5_data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker-timeout", type=int, default=300)
    parser.add_argument("--min-headroom-fraction", type=float, default=0.15)
    parser.add_argument("--worker-batch-size", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    if args.worker_batch_size is not None:
        if args.worker_batch_size <= 0 or args.output_dir is None:
            parser.error("Worker needs positive batch size and an explicit output directory")
        raise SystemExit(run_worker(args))
    output = (args.output_dir or PROJECT_ROOT / "outputs/dp6_batch_benchmark" /
              datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", {
        "batch_sizes_in_order": args.batch_sizes, "steps": args.steps, "warmup": args.warmup,
        "seed": args.seed, "data_dir": str(args.data_dir.resolve()), "device": args.device,
        "num_threads": args.num_threads, "worker_timeout_seconds": args.worker_timeout,
        "min_headroom_fraction": args.min_headroom_fraction,
        "script_sha256": file_sha256(Path(__file__)),
        "scope": "one sequential resource pass, independent processes; not a loss/quality comparison",
    }, exclusive=True)
    print(f"output: {output}", flush=True)
    results = []
    for batch_size in args.batch_sizes:
        worker_output = output / f"batch_{batch_size:03d}"
        command = [sys.executable, str(Path(__file__).resolve()),
                   "--worker-batch-size", str(batch_size), "--output-dir", str(worker_output),
                   "--data-dir", str(args.data_dir.resolve()), "--steps", str(args.steps),
                   "--warmup", str(args.warmup), "--seed", str(args.seed),
                   "--num-threads", str(args.num_threads), "--device", args.device]
        print(f"batch={batch_size}: launching independent worker", flush=True)
        with (output / f"batch_{batch_size:03d}.log").open("x", encoding="utf-8") as log:
            try:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=args.worker_timeout, check=False)
                path = worker_output / "result.json"
                if not path.is_file():
                    result = {"batch_size": batch_size, "status": "ERROR",
                              "error": f"Worker exited {process.returncode} without result"}
                else:
                    result = json.loads(path.read_text())
                    expected = {"PASS": 0, "OOM": 2, "ERROR": 3}.get(result["status"])
                    if process.returncode != expected or result["batch_size"] != batch_size:
                        result = {"batch_size": batch_size, "status": "ERROR",
                                  "error": "Worker status/exit code/batch size mismatch"}
            except subprocess.TimeoutExpired:
                result = {"batch_size": batch_size, "status": "ERROR", "error": "Worker timed out"}
        results.append(result)
        candidate = resource_candidate(results, args.min_headroom_fraction)
        summary = {
            "status": "ERROR" if result["status"] == "ERROR" else
                      ("RUNNING" if len(results) < len(args.batch_sizes) else
                       ("PASS" if candidate is not None else "NO_SAFE_CANDIDATE")),
            "results": results, "resource_candidate_batch_size": candidate,
            "scope": "resource candidate only; no training configuration changed; no policy-quality claim",
        }
        write_json(output / "summary.json", summary)
        with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"batch={batch_size}: {result['status']}; samples/s={result.get('samples_per_second')}", flush=True)
        if result["status"] == "ERROR":
            parser.exit(1, f"Probe stopped; inspect {output}\n")
    print(f"Resource candidate: {candidate}; all reports: {output}", flush=True)


if __name__ == "__main__":
    main()
