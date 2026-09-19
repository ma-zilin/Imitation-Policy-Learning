"""Evaluate a pretrained Diffusion Policy on fixed Push-T seeds."""

import csv
from pathlib import Path
from time import perf_counter

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio.v2 as imageio
import numpy
import torch

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs/eval/example_pusht_diffusion"
METRICS_PATH = PROJECT_ROOT / "artifacts/dp1_pretrained_evaluation.csv"

DEVICE = "cuda"
PRETRAINED_POLICY_PATH = "lerobot/diffusion_pusht"
PRETRAINED_POLICY_REVISION = "202c869"
EVALUATION_SEEDS = tuple(range(10))


def seed_episode(seed: int) -> None:
    """固定环境之外的 NumPy 与 PyTorch 随机性。"""
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def prepare_observation(numpy_observation: dict) -> dict[str, torch.Tensor]:
    """把环境的 HWC/NumPy 观测转换为策略需要的 BCHW/PyTorch 观测。"""
    state = torch.from_numpy(numpy_observation["agent_pos"]).to(torch.float32)
    image = torch.from_numpy(numpy_observation["pixels"]).to(torch.float32) / 255
    image = image.permute(2, 0, 1)

    return {
        "observation.state": state.to(DEVICE, non_blocking=True).unsqueeze(0),
        "observation.image": image.to(DEVICE, non_blocking=True).unsqueeze(0),
    }


def evaluate_episode(policy: DiffusionPolicy, env: gym.Env, seed: int) -> dict[str, object]:
    """运行一个闭环 episode，并返回该 episode 的诊断指标。"""
    seed_episode(seed)
    policy.reset()
    numpy_observation, info = env.reset(seed=seed)

    rewards: list[float] = []
    coverages: list[float] = []
    actions: list[numpy.ndarray] = []
    replan_latencies: list[float] = []
    frames = [env.render()]

    terminated = False
    truncated = False
    step = 0

    while not (terminated or truncated):
        observation = prepare_observation(numpy_observation)

        # 队列为空时，select_action() 会运行完整扩散采样并生成新的 action chunk。
        needs_replan = len(policy._queues["action"]) == 0
        if needs_replan and DEVICE.startswith("cuda"):
            torch.cuda.synchronize()
        if needs_replan:
            start_time = perf_counter()

        with torch.inference_mode():
            action = policy.select_action(observation)

        if needs_replan:
            if DEVICE.startswith("cuda"):
                torch.cuda.synchronize()
            replan_latencies.append(perf_counter() - start_time)

        numpy_action = action.squeeze(0).detach().cpu().numpy()
        actions.append(numpy_action.copy())

        numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
        rewards.append(float(reward))
        coverages.append(float(info["coverage"]))
        frames.append(env.render())
        step += 1

    success = bool(info["is_success"])
    video_path = OUTPUT_DIRECTORY / f"{'success' if success else 'failure'}_seed_{seed:03d}.mp4"
    imageio.mimsave(str(video_path), numpy.stack(frames), fps=env.metadata["render_fps"])

    replan_latencies_ms = numpy.asarray(replan_latencies) * 1000
    steady_latencies_ms = replan_latencies_ms[1:]

    actions_array = numpy.stack(actions)
    action_deltas = numpy.linalg.norm(numpy.diff(actions_array, axis=0), axis=1)

    return {
        "seed": seed,
        "success": success,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "episode_length": step,
        "max_reward": max(rewards),
        "max_coverage": max(coverages),
        "replan_count": len(replan_latencies_ms),
        "first_replan_latency_ms": replan_latencies_ms[0],
        "mean_replan_latency_ms": replan_latencies_ms.mean(),
        "p95_replan_latency_ms": numpy.percentile(replan_latencies_ms, 95),
        "steady_mean_replan_latency_ms": (
            steady_latencies_ms.mean() if len(steady_latencies_ms) else numpy.nan
        ),
        "steady_p95_replan_latency_ms": (
            numpy.percentile(steady_latencies_ms, 95) if len(steady_latencies_ms) else numpy.nan
        ),
        "mean_action_delta": action_deltas.mean(),
        "p95_action_delta": numpy.percentile(action_deltas, 95),
        "max_action_delta": action_deltas.max(),
        "video_path": str(video_path.relative_to(PROJECT_ROOT)),
    }


def save_metrics(results: list[dict[str, object]]) -> None:
    """将逐 episode 指标保存为便于后续分析的 CSV。"""
    with METRICS_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def print_episode_result(result: dict[str, object]) -> None:
    """打印紧凑的逐 episode 摘要，避免输出数千条 step 日志。"""
    print(
        f"seed={result['seed']:02d} "
        f"success={result['success']} "
        f"length={result['episode_length']} "
        f"max_coverage={result['max_coverage']:.4f} "
        f"replans={result['replan_count']} "
        f"replan_mean={result['mean_replan_latency_ms']:.2f}ms "
        f"action_delta_mean={result['mean_action_delta']:.2f}"
    )


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    policy = DiffusionPolicy.from_pretrained(
        PRETRAINED_POLICY_PATH,
        map_location=DEVICE,
        revision=PRETRAINED_POLICY_REVISION,
        local_files_only=True,
    )
    policy.eval()

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        max_episode_steps=300,
        render_mode="rgb_array",
    )

    print("Policy input features:", policy.config.input_features)
    print("Environment observation space:", env.observation_space)
    print("Policy output features:", policy.config.output_features)
    print("Environment action space:", env.action_space)

    results = []
    try:
        for seed in EVALUATION_SEEDS:
            result = evaluate_episode(policy, env, seed)
            results.append(result)
            # 每个 episode 后立即落盘，避免后续运行中断时丢失已完成结果。
            save_metrics(results)
            print_episode_result(result)
    finally:
        env.close()

    successes = sum(bool(result["success"]) for result in results)
    print(f"\nSuccess rate: {successes}/{len(results)} = {successes / len(results):.1%}")
    print(f"Mean max coverage: {numpy.mean([result['max_coverage'] for result in results]):.4f}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
