"""DP7 最小采样自检：只做两次规划，不训练、不执行环境动作。"""

import json
import os
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = "lerobot/diffusion_pusht"
MODEL_REVISION = "202c8695947a458d8ce08ce1c0219f2588d60b33"
INFERENCE_STEPS = (100, 50)
ENVIRONMENT_SEED = 0
SAMPLING_SEED = 0


def cached_model_directory() -> Path:
    """仅解析固定版本的缓存；缺文件时停止，不发起下载。"""
    from huggingface_hub import hf_hub_download

    paths = [
        Path(hf_hub_download(MODEL_REPO, name, revision=MODEL_REVISION,
                             local_files_only=True))
        for name in ("config.json", "model.safetensors")
    ]
    if paths[0].parent != paths[1].parent or not all(path.is_file() for path in paths):
        raise RuntimeError("固定版本的配置与权重缓存不完整。")
    config = json.loads(paths[0].read_text(encoding="utf-8"))
    # 该 checkpoint 本身已包含视觉编码器；禁止构造时额外下载骨干权重。
    if config.get("pretrained_backbone_weights") is not None:
        raise RuntimeError("该配置可能额外下载视觉骨干，本自检停止。")
    return paths[0].parent


def build_conditioning_batch(policy, observation: dict) -> dict:
    """复用首次 select_action 的归一化、相机合并和队列填充逻辑。"""
    from lerobot.common.policies.utils import populate_queues

    policy.reset()
    batch = dict(policy.normalize_inputs(observation))
    batch["observation.images"] = torch.stack(
        [batch[key] for key in policy.config.image_features], dim=-4
    )
    # 初始观测重复两次，与 LeRobot 首次规划的行为一致。
    policy._queues = populate_queues(policy._queues, batch)
    return {
        key: torch.stack(list(policy._queues[key]), dim=1)
        for key in batch if key in policy._queues
    }


def check_one_setting(policy, global_cond: torch.Tensor, steps: int, seed: int):
    """调用真实采样函数；hook 记录实际调用与时间步，退出时移除。"""
    config = policy.config
    diffusion = policy.diffusion
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= config.num_train_timesteps:
        raise ValueError("推理步数必须是训练时间步范围内的正整数。")
    if global_cond.shape[0] != 1:
        raise ValueError("本自检限定 batch size 为 1。")

    original_config_steps = config.num_inference_steps
    original_runtime_steps = diffusion.num_inference_steps
    calls = []
    initial_noise = []

    def record_call(module, args):
        sample, timesteps = args[:2]
        if not calls:
            initial_noise.append(sample.detach().cpu().clone())
        calls.append(int(timesteps.item()))

    handle = diffusion.unet.register_forward_pre_hook(record_call)
    try:
        config.num_inference_steps = steps
        diffusion.num_inference_steps = steps
        generator = torch.Generator(device=global_cond.device).manual_seed(seed)
        with torch.inference_mode():
            complete = diffusion.conditional_sample(1, global_cond=global_cond, generator=generator)

        timesteps = diffusion.noise_scheduler.timesteps.detach().cpu().tolist()
        if len(timesteps) != steps or len(calls) != steps or calls != timesteps:
            raise RuntimeError("实际 UNet 调用与 scheduler 时间步不一致。")
        if not all(a > b for a, b in zip(timesteps, timesteps[1:])) or timesteps[-1] != 0:
            raise RuntimeError("本次时间步序列未按预期降序到达 0。")

        action_dim = config.action_feature.shape[0]
        expected_complete = (1, config.horizon, action_dim)
        if tuple(complete.shape) != expected_complete:
            raise RuntimeError(f"完整序列形状错误：{tuple(complete.shape)}")
        # 与 generate_actions() 相同：当前动作从序列位置 n_obs_steps - 1 开始。
        start = config.n_obs_steps - 1
        end = start + config.n_action_steps
        selected = complete[:, start:end]
        expected_selected = (1, config.n_action_steps, action_dim)
        if tuple(selected.shape) != expected_selected:
            raise RuntimeError("执行片段的索引或形状错误。")
        if not torch.isfinite(complete).all().item():
            raise RuntimeError("生成序列包含 NaN 或 Inf。")
        if complete.requires_grad:
            raise RuntimeError("采样意外记录了梯度。")
        with torch.inference_mode():
            physical_actions = policy.unnormalize_outputs({"action": selected})["action"]
        if not torch.isfinite(physical_actions).all().item():
            raise RuntimeError("反归一化后的执行片段包含 NaN 或 Inf。")

        return {
            "status": "PASS", "inference_steps": steps,
            "runtime_steps": diffusion.num_inference_steps,
            "scheduler_timesteps": timesteps, "unet_timesteps": calls,
            "unet_calls": len(calls), "complete_shape": list(complete.shape),
            "selected_shape": list(selected.shape), "selected_indices": [start, end],
            "normalized_finite": True, "unnormalized_finite": True,
            "requires_grad": False,
        }, initial_noise[0]
    finally:
        handle.remove()
        config.num_inference_steps = original_config_steps
        diffusion.num_inference_steps = original_runtime_steps


def main() -> None:
    # 只影响本进程；双重限制 HF 客户端联网，不修改用户环境配置。
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    model_directory = cached_model_directory()

    import gymnasium as gym
    from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from evaluate_pretrained import DEVICE, prepare_observation

    if not torch.cuda.is_available():
        raise RuntimeError("本自检使用现有 cuda 预处理接口，CUDA 不可用时停止。")
    torch.set_num_threads(1)
    torch.manual_seed(SAMPLING_SEED)
    policy = DiffusionPolicy.from_pretrained(
        model_directory, map_location=DEVICE, local_files_only=True, strict=True
    )
    policy.eval()
    config = policy.config
    invariants = (config.n_obs_steps, config.horizon, config.n_action_steps,
                  config.num_train_timesteps, config.noise_scheduler_type, config.prediction_type)
    if invariants != (2, 16, 8, 100, "DDPM", "epsilon"):
        raise RuntimeError(f"固定模型的关键配置发生变化：{invariants}")

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                   max_episode_steps=300, render_mode="rgb_array")
    try:
        observation, _ = env.reset(seed=ENVIRONMENT_SEED)
        with torch.inference_mode():
            batch = build_conditioning_batch(policy, prepare_observation(observation))
            global_cond = policy.diffusion._prepare_global_conditioning(batch)
    finally:
        env.close()
    if not torch.isfinite(global_cond).all().item():
        raise RuntimeError("观测条件包含非有限值。")
    original_condition = global_cond.clone()

    results = []
    initial_noises = []
    for steps in INFERENCE_STEPS:
        result, initial_noise = check_one_setting(policy, global_cond, steps, SAMPLING_SEED)
        results.append(result)
        initial_noises.append(initial_noise)
        if not torch.equal(global_cond, original_condition):
            raise RuntimeError("共享观测条件被采样修改。")
        print(f"steps={steps}: UNet calls={result['unet_calls']}; "
              f"complete={tuple(result['complete_shape'])}; "
              f"selected={tuple(result['selected_shape'])}; finite=PASS")
    if not torch.equal(initial_noises[0], initial_noises[1]):
        raise RuntimeError("两组初始高斯噪声不一致。")

    summary = {
        "status": "PASS", "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
        "model_directory": str(model_directory), "environment_seed": ENVIRONMENT_SEED,
        "sampling_seed": SAMPLING_SEED, "same_initial_noise": True,
        "same_global_condition": True, "global_condition_shape": list(global_cond.shape),
        "conditioning_batch_shapes": {key: list(value.shape) for key, value in batch.items()},
        "horizons": {"observation": 2, "prediction": 16, "execution": 8},
        "scheduler": dict(policy.diffusion.noise_scheduler.config),
        "device": str(next(policy.parameters()).device),
        "gpu": torch.cuda.get_device_name(), "num_threads": torch.get_num_threads(),
        "versions": {name: version(name) for name in
                     ("torch", "diffusers", "safetensors", "gym-pusht", "lerobot")},
        "environment_steps": 0, "optimizer_steps": 0,
        "scope": "Two sampling calls only; not a closed-loop or latency evaluation.",
        "results": results,
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_directory = PROJECT_ROOT / "outputs/dp7_sampling_check" / timestamp
    output_directory.mkdir(parents=True, exist_ok=False)
    output_path = output_directory / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("DP7 sampling self-check: PASS; no training or closed-loop evaluation.")
    print(f"Summary: {output_path}")


if __name__ == "__main__":
    main()
