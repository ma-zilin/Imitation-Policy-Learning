"""用真实 Push-T batch 检查 LeRobot 的一次 forward/backward，不更新参数。"""

import argparse
from unittest.mock import patch

import torch
from torch.utils.data import default_collate

from inspect_pusht_dataset import (
    DATASET_REPO_ID,
    POLICY_REPO_ID,
    POLICY_REVISION,
    build_delta_timestamps,
)
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.policies import PreTrainedConfig


def check_gradients(module: torch.nn.Module, label: str) -> None:
    """检查模块中所有可训练参数有有限梯度，且模块至少有一个非零梯度。"""
    parameters = [(name, p) for name, p in module.named_parameters() if p.requires_grad]
    if not parameters:
        raise RuntimeError(f"{label}: no trainable parameters")
    missing = [name for name, p in parameters if p.grad is None]
    if missing:
        raise RuntimeError(f"{label}: missing gradients: {missing}")
    nonzero = 0
    squared_norm = 0.0
    for name, parameter in parameters:
        gradient = parameter.grad
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f"{label}.{name}: nonfinite gradient")
        nonzero += int(torch.count_nonzero(gradient).item() > 0)
        squared_norm += gradient.double().square().sum().item()
    if nonzero == 0:
        raise RuntimeError(f"{label}: all gradients are zero")
    print(
        f"{label}: PASS; tensors={len(parameters)}, nonzero={nonzero}, "
        f"gradient_norm={squared_norm ** 0.5:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--anchors", type=int, nargs="+", default=[79, 80],
                        help="选中 episode 内的零起始帧索引。")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu if intended")
    torch.manual_seed(args.seed)

    # 沿用 DP1 配置和 DP2 缓存；不下载新数据或重新划分数据集。
    config = PreTrainedConfig.from_pretrained(
        POLICY_REPO_ID, revision=POLICY_REVISION, local_files_only=True
    )
    metadata = LeRobotDatasetMetadata(DATASET_REPO_ID, local_files_only=True)
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
    samples = [dataset[start + anchor] for anchor in args.anchors]
    keys = list(config.input_features) + ["action", "action_is_pad"]
    batch = default_collate([{key: sample[key] for key in keys} for sample in samples])
    batch = {key: value.to(args.device) for key, value in batch.items()}
    print(f"episode={args.episode}, anchors={args.anchors}, seed={args.seed}")
    for key, value in batch.items():
        print(f"input {key}: {tuple(value.shape)}, {value.dtype}")

    # checkpoint 同时提供模型参数和归一化统计量，不采用新划分的统计量。
    policy = DiffusionPolicy.from_pretrained(
        POLICY_REPO_ID, revision=POLICY_REVISION,
        map_location=args.device, local_files_only=True,
    )
    policy.train()
    policy.zero_grad(set_to_none=True)
    captured = {}

    def record_noise(clean, noise, timesteps):
        # 仅记录原函数的输入输出，仍由原 scheduler 执行加噪。
        result = original_add_noise(clean, noise, timesteps)
        captured.update(clean=clean.detach().clone(), noise=noise.detach().clone(),
                        timesteps=timesteps.detach().clone(), noisy=result.detach().clone())
        return result

    def record_unet_inputs(module, positional, keyword):
        captured["condition"] = keyword["global_cond"].detach().clone()
        torch.testing.assert_close(positional[0], captured["noisy"])
        torch.testing.assert_close(positional[1], captured["timesteps"])

    def record_prediction(module, positional, output):
        captured["prediction"] = output.detach().clone()

    unet = policy.diffusion.unet
    handles = [
        unet.register_forward_pre_hook(record_unet_inputs, with_kwargs=True),
        unet.register_forward_hook(record_prediction),
    ]
    scheduler = policy.diffusion.noise_scheduler
    original_add_noise = scheduler.add_noise
    try:
        with patch.object(scheduler, "add_noise", side_effect=record_noise) as observer:
            loss = policy(batch)["loss"]
            if observer.call_count != 1:
                raise RuntimeError("Expected one forward-noising call")
    finally:
        for handle in handles:
            handle.remove()

    expected_shape = (len(args.anchors), config.horizon, config.action_feature.shape[0])
    for key in ["clean", "noise", "noisy", "prediction"]:
        value = captured[key]
        if tuple(value.shape) != expected_shape or not torch.isfinite(value).all():
            raise RuntimeError(f"Invalid {key} tensor")
        print(f"{key}: {tuple(value.shape)}")
    timesteps = captured["timesteps"]
    assert timesteps.shape == (len(args.anchors),)
    assert ((timesteps >= 0) & (timesteps < config.num_train_timesteps)).all()
    print(f"timesteps: {timesteps.tolist()} (one per sample, repeats are allowed)")
    if captured["condition"].shape[0] != len(args.anchors) or not torch.isfinite(captured["condition"]).all():
        raise RuntimeError("Invalid observation condition")
    print(f"condition: {tuple(captured['condition'].shape)}")

    # 独立核对预测目标和最终 MSE；不另写训练逻辑。
    target = captured["noise"] if config.prediction_type == "epsilon" else captured["clean"]
    expected_loss = (captured["prediction"] - target).square()
    if config.do_mask_loss_for_padding:
        expected_loss *= (~batch["action_is_pad"]).unsqueeze(-1)
    torch.testing.assert_close(loss.detach(), expected_loss.mean())
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError("Loss must be a finite scalar")
    print(f"loss={loss.item():.6f}; target={config.prediction_type}; MSE check PASS")

    loss.backward()
    check_gradients(policy.diffusion.rgb_encoder, "visual encoder")
    check_gradients(unet, "denoising UNet")
    print("DP3 forward/backward: PASS; no optimizer step or checkpoint save")


if __name__ == "__main__":
    main()
