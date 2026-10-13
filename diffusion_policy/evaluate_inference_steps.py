"""DP7：固定预训练权重的 100/50 步配对闭环评测，不重新训练。"""

import csv
import hashlib
import inspect
import json
import os
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import imageio.v2 as imageio
import numpy as np
import torch

from check_inference_steps import MODEL_REPO, MODEL_REVISION, cached_model_directory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(5))
SETTINGS = (100, 50)
MAX_ENVIRONMENT_STEPS = 300
DEVICE = "cuda"


def make_plan() -> list[dict]:
    """预先固定顺序；奇偶 seed 交替先运行哪一组，不根据结果挑选。"""
    return [
        {"environment_seed": seed, "sampling_seed": seed, "inference_steps": steps}
        for seed in SEEDS
        for steps in (SETTINGS if seed % 2 == 0 else SETTINGS[::-1])
    ]


def save_json(path: Path, value: dict) -> None:
    """单目录内替换小型状态文件，保留已完成 episode 的记录。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initial_fingerprint(observation: dict, info: dict) -> str:
    """同时核对图像、agent 状态与 reset 返回的实际物理状态。"""
    digest = hashlib.sha256()
    arrays = {**observation, **{key: info[key] for key in
                              ("pos_agent", "vel_agent", "block_pose", "goal_pose")}}
    for key, value in sorted(arrays.items()):
        array = np.asarray(value)
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def finite_values(values) -> list[float]:
    result = [float(value) for value in values]
    if not all(np.isfinite(result)):
        raise RuntimeError("指标包含 NaN 或 Inf。")
    return result


def latency_summary(latencies_ms: list[float]) -> dict:
    values = finite_values(latencies_ms)
    if not values:
        raise RuntimeError("episode 没有重新规划。")
    steady = values[1:]
    return {
        "replan_count": len(values), "first_replan_latency_ms": values[0],
        "mean_replan_latency_ms": float(np.mean(values)),
        "p95_replan_latency_ms": float(np.percentile(values, 95)),
        "steady_mean_replan_latency_ms": float(np.mean(steady)) if steady else None,
        "steady_p95_replan_latency_ms": float(np.percentile(steady, 95)) if steady else None,
    }


def format_latency(value: float | None) -> str:
    return f"{value:.2f}ms" if value is not None else "n/a"


def evaluate_episode(policy, env, setting: dict, directory: Path,
                     expected_initial_fingerprint: str | None = None) -> dict:
    """执行一次真正的闭环；每步记录并 flush，视频编码在计时区间外。"""
    from evaluate_pretrained import prepare_observation, seed_episode

    directory.mkdir(parents=True, exist_ok=False)
    seed = setting["environment_seed"]
    steps = setting["inference_steps"]
    policy.config.num_inference_steps = steps
    policy.diffusion.num_inference_steps = steps
    policy.diffusion.noise_scheduler.set_timesteps(steps)
    if len(policy.diffusion.noise_scheduler.timesteps) != steps:
        raise RuntimeError("scheduler 没有采用指定步数。")
    seed_episode(setting["sampling_seed"])
    policy.reset()
    observation, info = env.reset(seed=seed)
    fingerprint = initial_fingerprint(observation, info)
    if expected_initial_fingerprint is not None and fingerprint != expected_initial_fingerprint:
        raise RuntimeError("同一 seed 两组的初始状态不一致；停止而不是继续比较。")

    save_json(directory / "status.json", {
        "status": "RUNNING", **setting, "initial_fingerprint": fingerprint,
    })
    rewards, coverages, actions, replan_latencies = [], [], [], []
    trace_path = directory / "trajectory.csv"
    video_path = directory / "rollout.mp4"
    terminated = truncated = False
    fields = ["environment_step", "action_x", "action_y", "action_delta", "reward",
              "coverage", "terminated", "truncated", "replan", "replan_latency_ms"]
    try:
        with trace_path.open("x", newline="", encoding="utf-8") as trace_file, \
                imageio.get_writer(str(video_path), fps=env.metadata["render_fps"],
                                   macro_block_size=2) as video:
            writer = csv.DictWriter(trace_file, fieldnames=fields)
            writer.writeheader()
            video.append_data(env.render())
            for step in range(1, MAX_ENVIRONMENT_STEPS + 1):
                prepared = prepare_observation(observation)
                needs_replan = not policy._queues["action"]
                if needs_replan:
                    torch.cuda.synchronize()
                    start = perf_counter()
                with torch.inference_mode():
                    action = policy.select_action(prepared)
                latency_ms = None
                if needs_replan:
                    torch.cuda.synchronize()
                    latency_ms = (perf_counter() - start) * 1000
                    replan_latencies.append(latency_ms)

                if tuple(action.shape) != (1, 2) or not torch.isfinite(action).all().item():
                    raise RuntimeError("策略动作形状错误或包含非有限数值。")
                numpy_action = action.squeeze(0).detach().cpu().numpy().copy()
                delta = float(np.linalg.norm(numpy_action - actions[-1])) if actions else None
                actions.append(numpy_action)
                observation, reward, terminated, truncated, info = env.step(numpy_action)
                coverage = float(info["coverage"])
                finite_values([reward, coverage])
                rewards.append(float(reward))
                coverages.append(coverage)
                writer.writerow({
                    "environment_step": step, "action_x": float(numpy_action[0]),
                    "action_y": float(numpy_action[1]), "action_delta": delta,
                    "reward": float(reward), "coverage": coverage,
                    "terminated": bool(terminated), "truncated": bool(truncated),
                    "replan": bool(needs_replan), "replan_latency_ms": latency_ms,
                })
                trace_file.flush()
                video.append_data(env.render())
                if terminated or truncated:
                    break
            else:
                raise RuntimeError("环境达到 300 步却未结束；停止并检查 TimeLimit。")

        success = bool(info["is_success"])
        if success != (coverages[-1] > float(env.unwrapped.success_threshold)):
            raise RuntimeError("环境 success 与原始 coverage 判据不一致。")
        timesteps = policy.diffusion.noise_scheduler.timesteps.cpu().tolist()
        if len(timesteps) != steps or policy.diffusion.num_inference_steps != steps:
            raise RuntimeError("采样过程中实际运行步数发生变化。")
        deltas = np.linalg.norm(np.diff(np.stack(actions), axis=0), axis=1)
        result = {
            **setting, "success": success, "terminated": bool(terminated),
            "truncated": bool(truncated), "episode_length": len(actions),
            "max_reward": max(rewards), "max_coverage": max(coverages),
            **latency_summary(replan_latencies),
            "mean_action_delta": float(np.mean(deltas)) if len(deltas) else None,
            "p95_action_delta": float(np.percentile(deltas, 95)) if len(deltas) else None,
            "max_action_delta": float(np.max(deltas)) if len(deltas) else None,
            "initial_fingerprint": fingerprint, "scheduler_timesteps": timesteps,
            "replan_latencies_ms": finite_values(replan_latencies),
            "video_path": str(video_path.relative_to(directory.parent)),
            "trace_path": str(trace_path.relative_to(directory.parent)),
        }
        save_json(directory / "result.json", result)
        save_json(directory / "status.json", {"status": "COMPLETE", **setting})
        return result
    except BaseException as error:
        save_json(directory / "status.json", {
            "status": "ERROR", **setting, "completed_environment_steps": len(rewards),
            "error": f"{type(error).__name__}: {error}",
        })
        raise


def save_metrics(path: Path, results: list[dict]) -> None:
    if not results:
        return
    excluded = {"replan_latencies_ms", "scheduler_timesteps"}
    fields = [key for key in results[0] if key not in excluded]
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in results)
    temporary.replace(path)


def summarize(results: list[dict]) -> dict:
    expected = {(seed, steps) for seed in SEEDS for steps in SETTINGS}
    actual = {(row["environment_seed"], row["inference_steps"]) for row in results}
    if len(results) != len(expected) or actual != expected:
        raise RuntimeError("配对评测尚未完整，不能输出完整实验汇总。")
    for seed in SEEDS:
        pair = [row for row in results if row["environment_seed"] == seed]
        if pair[0]["initial_fingerprint"] != pair[1]["initial_fingerprint"]:
            raise RuntimeError("配对初始状态不一致。")

    groups = {}
    for steps in SETTINGS:
        rows = [row for row in results if row["inference_steps"] == steps]
        # 每个 episode 单独排除首次；此处合并的是原始延迟，不是平均 p95。
        steady = finite_values([value for row in rows for value in row["replan_latencies_ms"][1:]])
        groups[str(steps)] = {
            "episodes": len(rows), "successes": sum(row["success"] for row in rows),
            "success_rate": sum(row["success"] for row in rows) / len(rows),
            "mean_max_coverage": float(np.mean([row["max_coverage"] for row in rows])),
            "max_coverage_by_seed": {str(row["environment_seed"]): row["max_coverage"] for row in rows},
            "mean_episode_length": float(np.mean([row["episode_length"] for row in rows])),
            "total_environment_steps": sum(row["episode_length"] for row in rows),
            "total_replans": sum(row["replan_count"] for row in rows),
            "steady_replan_samples": len(steady),
            "steady_replan_mean_ms": float(np.mean(steady)) if steady else None,
            "steady_replan_p95_ms": float(np.percentile(steady, 95)) if steady else None,
            "mean_episode_action_delta": float(np.mean([
                row["mean_action_delta"] for row in rows if row["mean_action_delta"] is not None
            ])) if any(row["mean_action_delta"] is not None for row in rows) else None,
        }
    paired = []
    for seed in SEEDS:
        pair = {row["inference_steps"]: row for row in results if row["environment_seed"] == seed}
        paired.append({
            "environment_seed": seed, "success_100": pair[100]["success"],
            "success_50": pair[50]["success"],
            "max_coverage_100": pair[100]["max_coverage"],
            "max_coverage_50": pair[50]["max_coverage"],
            "coverage_delta_50_minus_100": pair[50]["max_coverage"] - pair[100]["max_coverage"],
        })
    baseline = groups["100"]["steady_replan_mean_ms"]
    reduced = groups["50"]["steady_replan_mean_ms"]
    total_steps = sum(row["episode_length"] for row in results)
    if total_steps > 3000:
        raise RuntimeError("环境交互超过约定预算。")
    return {
        "status": "COMPLETE", "groups": groups, "paired_results": paired,
        "pooled_steady_latency_ratio_50_to_100": reduced / baseline if baseline and reduced else None,
        "total_environment_steps": total_steps, "optimizer_steps": 0,
        "scope": "Five paired environment seeds; observational, not statistical superiority or paper reproduction.",
        "latency_scope": "Synchronized select_action replans, excluding the first replan of each episode; not real-time control.",
        "randomness_scope": "Matching seeds do not guarantee matching full noise trajectories.",
    }


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    model_directory = cached_model_directory()
    import gymnasium as gym
    from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from diffusers import DDPMScheduler
    from evaluate_pretrained import prepare_observation

    if not torch.cuda.is_available():
        raise RuntimeError("该协议固定使用 CUDA；不可用时停止。")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    policy = DiffusionPolicy.from_pretrained(
        model_directory, map_location=DEVICE, local_files_only=True, strict=True
    )
    policy.eval()
    config = policy.config
    invariant = (config.n_obs_steps, config.horizon, config.n_action_steps,
                 config.num_train_timesteps, config.noise_scheduler_type, config.prediction_type)
    if invariant != (2, 16, 8, 100, "DDPM", "epsilon"):
        raise RuntimeError(f"固定配置发生变化：{invariant}")
    scheduler_config = dict(policy.diffusion.noise_scheduler.config)
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                   max_episode_steps=MAX_ENVIRONMENT_STEPS, render_mode="rgb_array")
    results = []
    output_directory = None
    try:
        if env.unwrapped.success_threshold != 0.95:
            raise RuntimeError("环境 success 阈值发生变化，停止。")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output_directory = PROJECT_ROOT / "outputs/dp7_inference_steps" / timestamp
        output_directory.mkdir(parents=True, exist_ok=False)
        plan = make_plan()
        source_files = [Path(__file__), Path(inspect.getfile(prepare_observation)),
                        Path(inspect.getfile(DiffusionPolicy)),
                        Path(inspect.getfile(DDPMScheduler)),
                        Path(inspect.getfile(type(env.unwrapped)))]
        manifest = {
            "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
            "model_directory": str(model_directory),
            "model_sha256": file_sha256(model_directory / "model.safetensors"),
            "original_model_config": json.loads((model_directory / "config.json").read_text()),
            "scheduler_config": scheduler_config,
            "plan": plan, "max_episode_steps": MAX_ENVIRONMENT_STEPS,
            "max_total_environment_steps": 3000, "max_episodes": 10,
            "success_rule": "info['is_success']; coverage > 0.95",
            "horizons": {"observation": 2, "prediction": 16, "execution": 8},
            "num_threads": torch.get_num_threads(), "device": DEVICE,
            "gpu": torch.cuda.get_device_name(), "video_macro_block_size": 2,
            "fresh_environment_per_episode": True,
            "versions": {name: version(name) for name in (
                "torch", "torchvision", "diffusers", "gym-pusht", "gymnasium",
                "safetensors", "lerobot", "numpy", "imageio")},
            "project_head": subprocess.check_output(
                ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "project_worktree": subprocess.check_output(
                ["git", "-C", str(PROJECT_ROOT), "status", "--short"], text=True),
            "source_sha256": {str(path): file_sha256(path) for path in source_files},
            "command": "python diffusion_policy/evaluate_inference_steps.py",
            "latency_rule": "CUDA synchronized around select_action when action cache is empty; exclude first replan per episode",
            "no_training": True, "no_warmup_rollouts": True, "no_automatic_retries": True,
        }
        save_json(output_directory / "manifest.json", manifest)
        save_json(output_directory / "status.json", {"status": "RUNNING", "completed_episodes": 0})
        print(f"Output: {output_directory}", flush=True)
        for index, setting in enumerate(plan, start=1):
            seed, steps = setting["environment_seed"], setting["inference_steps"]
            previous = next((row for row in results if row["environment_seed"] == seed), None)
            directory = output_directory / f"steps_{steps}_seed_{seed:03d}"
            # 新建同配置环境，避免上一 episode 的可视化动作标记残留。
            episode_env = gym.make(
                "gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                max_episode_steps=MAX_ENVIRONMENT_STEPS, render_mode="rgb_array")
            try:
                result = evaluate_episode(policy, episode_env, setting, directory,
                                          previous["initial_fingerprint"] if previous else None)
            finally:
                episode_env.close()
            results.append(result)
            save_metrics(output_directory / "episodes.csv", results)
            save_json(output_directory / "status.json", {
                "status": "RUNNING", "completed_episodes": len(results),
                "completed_environment_steps": sum(row["episode_length"] for row in results),
            })
            current = (config.n_obs_steps, config.horizon, config.n_action_steps,
                       config.num_train_timesteps, config.noise_scheduler_type, config.prediction_type)
            if current != invariant or dict(policy.diffusion.noise_scheduler.config) != scheduler_config:
                raise RuntimeError("除推理步数以外的模型配置发生变化。")
            print(f"[{index}/10] steps={steps} seed={seed} success={result['success']} "
                  f"length={result['episode_length']} coverage={result['max_coverage']:.4f} "
                  f"steady_mean={format_latency(result['steady_mean_replan_latency_ms'])}", flush=True)
        summary = summarize(results)
        save_json(output_directory / "summary.json", summary)
        save_json(output_directory / "status.json", {
            "status": "COMPLETE", "completed_episodes": 10,
            "completed_environment_steps": summary["total_environment_steps"],
        })
        for steps, group in summary["groups"].items():
            print(f"steps={steps}: success={group['successes']}/{group['episodes']}; "
                  f"mean_max_coverage={group['mean_max_coverage']:.4f}; "
                  f"pooled_steady_mean={format_latency(group['steady_replan_mean_ms'])}; "
                  f"pooled_steady_p95={format_latency(group['steady_replan_p95_ms'])}", flush=True)
        print("DP7 paired evaluation: COMPLETE; no training; small-sample boundaries apply.", flush=True)
    except BaseException as error:
        if output_directory is not None:
            save_json(output_directory / "status.json", {
                "status": "ERROR", "completed_episodes": len(results),
                "error": f"{type(error).__name__}: {error}",
            })
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
