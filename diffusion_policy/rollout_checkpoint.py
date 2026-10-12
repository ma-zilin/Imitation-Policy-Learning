"""复用 DP1 闭环评估逻辑，检查 DP5 本地 checkpoint；不下载公开权重。"""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import torch

import evaluate_pretrained as evaluation
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionModel, DiffusionPolicy
from train_smoke import PROJECT_ROOT, file_sha256, seed_all, verify_checkpoint, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="含 COMPLETE.json 的完整 checkpoint 目录。")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--untrained", action="store_true",
                        help="保留同样的归一化统计量，但重新初始化网络，作为未训练闭环对照。")
    parser.add_argument("--initialization-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.max_steps <= 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Positive step budget and unique seeds required")
    if any(seed < 0 or seed >= 2**32 for seed in [*args.seeds, args.initialization_seed]):
        parser.error("Seeds must fit uint32")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(1)
    checkpoint = args.checkpoint.resolve()
    marker = verify_checkpoint(checkpoint)
    output = (args.output_dir or PROJECT_ROOT / "outputs/dp5_rollout" /
              datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")).resolve()
    # DP1 指标将视频路径记为项目相对路径；明确约束，避免跑完才路径报错。
    if not output.is_relative_to(PROJECT_ROOT):
        parser.error("Rollout output must be inside this project")
    output.mkdir(parents=True, exist_ok=False)
    evaluation.DEVICE = args.device
    evaluation.OUTPUT_DIRECTORY = output
    policy = DiffusionPolicy.from_pretrained(
        checkpoint / "policy", map_location=args.device, local_files_only=True, strict=True
    )
    if args.untrained:
        if policy.config.pretrained_backbone_weights is not None:
            raise ValueError("Untrained baseline must not download backbone weights")
        seed_all(args.initialization_seed)
        # 归一化模块在 policy 层，网络在 diffusion 层；只替换网络，不换统计量。
        del policy.diffusion
        policy.diffusion = DiffusionModel(policy.config).to(args.device)
        policy.reset()
    policy.eval()
    write_json(output / "manifest.json", {
        "checkpoint": str(checkpoint), "optimizer_step": marker["step"],
        "checkpoint_sha256": marker["sha256"], "seeds": args.seeds,
        "max_steps": args.max_steps, "device": args.device,
        "policy_kind": "fresh network with checkpoint statistics" if args.untrained else "trained checkpoint",
        "initialization_seed": args.initialization_seed if args.untrained else None,
        "source_sha256": {str(path): file_sha256(path) for path in
                          [Path(__file__), Path(evaluation.__file__)]},
        "torch_version": torch.__version__,
        "scope": "local checkpoint control-chain check; success not required for short-training smoke",
    }, exclusive=True)
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                   max_episode_steps=args.max_steps, render_mode="rgb_array")
    results = []
    try:
        for seed in args.seeds:
            result = evaluation.evaluate_episode(policy, env, seed)
            if not Path(PROJECT_ROOT / result["video_path"]).is_file():
                raise RuntimeError("Missing rollout video")
            results.append(result)
            with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=result.keys())
                writer.writeheader()
                writer.writerows(results)
            evaluation.print_episode_result(result)
    finally:
        env.close()
    write_json(output / "summary.json", {
        "status": "PASS", "episodes": len(results),
        "successes": sum(bool(row["success"]) for row in results),
        "scope": "sampling/environment/video pipeline works; not a policy-quality acceptance",
    }, exclusive=True)
    print(f"Local checkpoint rollout chain: PASS; outputs: {output}")


if __name__ == "__main__":
    main()
