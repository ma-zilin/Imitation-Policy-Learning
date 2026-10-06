"""Inspect one Push-T episode and Diffusion Policy temporal windows."""

import argparse
from pathlib import Path
from typing import Any

import numpy
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.configs.policies import PreTrainedConfig


DATASET_REPO_ID = "lerobot/pusht"
POLICY_REPO_ID = "lerobot/diffusion_pusht"
POLICY_REVISION = "202c869"


def as_scalar(value: Any) -> Any:
    """把单元素 Tensor/NumPy 数组转换成便于打印的 Python 标量。"""
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    if isinstance(value, numpy.ndarray) and value.size == 1:
        return value.item()
    return value


def build_delta_timestamps(config: DiffusionConfig, fps: int) -> dict[str, list[float]]:
    """把策略的相对帧索引转换成 LeRobotDataset 使用的秒偏移。"""
    observation_times = [index / fps for index in config.observation_delta_indices]
    action_times = [index / fps for index in config.action_delta_indices]

    delta_timestamps = {
        key: observation_times for key in config.input_features
    }
    delta_timestamps.update({key: action_times for key in config.output_features})
    return delta_timestamps


def print_dataset_summary(
    metadata: LeRobotDatasetMetadata,
    dataset: LeRobotDataset,
    episode_index: int,
    config: DiffusionConfig,
) -> None:
    """只打印理解当前检查所需的数据集和时间窗信息。"""
    print(
        f"Push-T: episode={episode_index}, frames={dataset.num_frames}, "
        f"fps={metadata.fps}; repository episodes={metadata.total_episodes}"
    )
    print(
        f"Window: observations={config.n_obs_steps}, actions={config.horizon}, "
        f"execute={config.n_action_steps}, drop_last={config.drop_n_last_frames}"
    )


def print_window(dataset: LeRobotDataset, anchor: int, label: str) -> dict[str, Any]:
    """合并重复的图像/状态索引，只保留帧索引与填充位置。"""
    indices, padding = dataset._get_query_indices(anchor, 0)
    sample = dataset[anchor]
    if indices["observation.image"] != indices["observation.state"]:
        raise ValueError("Image and state observation windows are not aligned")
    obs_pad = torch.nonzero(padding["observation.image_is_pad"]).flatten().tolist()
    action_pad = torch.nonzero(padding["action_is_pad"]).flatten().tolist()
    print(
        f"{label}: t={int(as_scalar(sample['frame_index']))}, "
        f"time={as_scalar(sample['timestamp']):.2f}s"
    )
    print(f"  obs frames={indices['observation.image']}, pad positions={obs_pad}")
    print(f"  action frames={indices['action']}, pad positions={action_pad}")
    return sample


def check_normalization(
    sample: dict[str, Any], dataset: LeRobotDataset, config: DiffusionConfig
) -> None:
    """调用 LeRobot 原实现，逐通道检查边界和往返还原误差。"""
    features = {**config.input_features, **config.output_features}
    normalize = Normalize(features, config.normalization_mapping, dataset.meta.stats)
    unnormalize = Unnormalize(features, config.normalization_mapping, dataset.meta.stats)
    normalized = normalize(sample)
    restored = unnormalize(normalized)

    print("\nNormalization: raw range -> normalized range; outside=count, error=max_abs")
    print("Stats: repository meta/stats.json; selecting an episode does not recompute stats.")
    print("Training-only provenance is not verified; define the episode split before training.")
    for key, feature in features.items():
        # 图像通道在倒数第三维；状态和动作坐标在最后一维。
        channel_dim = -3 if key == "observation.image" else -1
        raw = sample[key].movedim(channel_dim, 0).flatten(start_dim=1)
        norm = normalized[key].movedim(channel_dim, 0).flatten(start_dim=1)
        back = restored[key].movedim(channel_dim, 0).flatten(start_dim=1)
        stats = dataset.meta.stats[key]
        mode = config.normalization_mapping[feature.type].value
        if mode == "MEAN_STD":
            mean, std = stats["mean"].flatten(), stats["std"].flatten()
            if not torch.isfinite(mean).all() or not (torch.isfinite(std) & (std > 0)).all():
                raise ValueError(f"Invalid mean/std for {key}")
            lower, upper = -mean / (std + 1e-8), (1 - mean) / (std + 1e-8)
            raw_lower, raw_upper = torch.zeros_like(mean), torch.ones_like(mean)
            tolerance = 1e-6
        elif mode == "MIN_MAX":
            raw_lower, raw_upper = stats["min"].flatten(), stats["max"].flatten()
            if not (torch.isfinite(raw_lower) & torch.isfinite(raw_upper) & (raw_upper > raw_lower)).all():
                raise ValueError(f"Invalid min/max for {key}")
            lower, upper = -torch.ones_like(raw_lower), torch.ones_like(raw_upper)
            tolerance = 1e-4
        else:
            raise ValueError(f"Unsupported normalization mode: {mode}")

        print(f"{key}: shape={tuple(sample[key].shape)}, mode={mode}")
        for channel in range(raw.shape[0]):
            finite = torch.isfinite(raw[channel]) & torch.isfinite(norm[channel]) & torch.isfinite(back[channel])
            outside = ((norm[channel] < lower[channel] - 1e-5) |
                       (norm[channel] > upper[channel] + 1e-5)).sum().item()
            raw_outside = ((raw[channel] < raw_lower[channel] - 1e-5) |
                           (raw[channel] > raw_upper[channel] + 1e-5)).sum().item()
            error = (back[channel] - raw[channel]).abs().max().item()
            print(
                f"  c{channel}: [{raw[channel].min():.4f}, {raw[channel].max():.4f}]"
                f" -> [{norm[channel].min():.4f}, {norm[channel].max():.4f}]"
                f"; expected=[{lower[channel]:.4f}, {upper[channel]:.4f}]"
                f"; outside={outside}, raw_outside={raw_outside},"
                f" nonfinite={(~finite).sum().item()}, error={error:.2e}"
            )
            if not finite.all() or error > tolerance:
                raise ValueError(f"Normalization round-trip failed for {key} channel {channel}")
        if mode == "MIN_MAX":
            print(f"  stats min={raw_lower.tolist()}, max={raw_upper.tolist()}")
    print("Normalization round-trip: PASS (range counts are reported separately)")


def save_window_plot(sample: dict[str, Any], config: DiffusionConfig, output_path: Path) -> None:
    """将同一个样本的观测图像和专家动作序列画在一张图中。"""
    images = sample["observation.image"]
    actions = sample["action"].detach().cpu().numpy()
    execute_start = config.n_obs_steps - 1
    execute_end = execute_start + config.n_action_steps

    if images.ndim != 4 or images.shape[:2] != (config.n_obs_steps, 3):
        raise ValueError(f"Unexpected image window shape: {tuple(images.shape)}")
    if actions.shape != (config.horizon, 2) or execute_end > len(actions):
        raise ValueError(f"Unexpected action chunk shape: {actions.shape}")

    figure = Figure(figsize=(14, 5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(
        1,
        config.n_obs_steps + 1,
        gridspec_kw={"width_ratios": [1] * config.n_obs_steps + [1.6]},
    )

    anchor_frame = int(as_scalar(sample["frame_index"]))
    for step, axis in enumerate(axes[:-1]):
        image = images[step].detach().cpu().permute(1, 2, 0).numpy()
        axis.imshow(numpy.clip(image, 0.0, 1.0))
        frame = anchor_frame + config.observation_delta_indices[step]
        axis.set_title(f"Observation frame {frame}")
        axis.axis("off")

    axis = axes[-1]
    # j < execute_start 是历史动作；连接到当前动作以显示序列连续性。
    if execute_start > 0:
        axis.plot(
            actions[: execute_start + 1, 0],
            actions[: execute_start + 1, 1],
            color="gray",
            linestyle=":",
            linewidth=2,
            label=f"Past (j < {execute_start})",
        )
        axis.scatter(
            actions[:execute_start, 0],
            actions[:execute_start, 1],
            color="gray",
            marker="s",
            s=35,
            zorder=3,
        )

    # 从最后一个待执行动作连到第一个丢弃动作；虚线部分本次不会执行。
    if execute_end < len(actions):
        axis.plot(
            actions[execute_end - 1 :, 0],
            actions[execute_end - 1 :, 1],
            color="darkorange",
            linestyle="--",
            linewidth=2,
            label=f"Not kept (j >= {execute_end})",
        )
        axis.scatter(
            actions[execute_end:, 0],
            actions[execute_end:, 1],
            facecolors="white",
            edgecolors="darkorange",
            s=38,
            zorder=3,
        )

    axis.plot(
        actions[execute_start:execute_end, 0],
        actions[execute_start:execute_end, 1],
        color="royalblue",
        linestyle="-",
        marker="o",
        markersize=5,
        linewidth=2.5,
        label=f"Kept for execution (j={execute_start}..{execute_end - 1})",
        zorder=4,
    )

    for index in sorted({0, execute_start, execute_end - 1, len(actions) - 1}):
        axis.annotate(
            f"j={index}",
            actions[index],
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    axis.set(
        title="Expert action targets",
        xlabel="Action x (environment coordinates)",
        ylabel="Action y (environment coordinates)",
        xlim=(0, 512),
        ylim=(512, 0),
    )
    axis.set_aspect("equal")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.suptitle(
        f"Episode {as_scalar(sample['episode_index'])}, anchor frame {anchor_frame}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one Push-T episode and its Diffusion Policy temporal windows."
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to inspect. Only this episode is downloaded.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only the locally cached dataset and video files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 配置已随 DP1 checkpoint 缓存在本机，不需要再次访问 Hub。
    config = PreTrainedConfig.from_pretrained(
        POLICY_REPO_ID,
        revision=POLICY_REVISION,
        local_files_only=True,
    )

    # 元数据很小；首次运行时，随后只下载选中的一个 episode 及其视频。
    metadata = LeRobotDatasetMetadata(
        DATASET_REPO_ID, local_files_only=args.local_files_only
    )
    delta_timestamps = build_delta_timestamps(config, metadata.fps)
    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        episodes=[args.episode],
        delta_timestamps=delta_timestamps,
        download_videos=True,
        local_files_only=args.local_files_only,
    )

    print_dataset_summary(metadata, dataset, args.episode, config)

    episode_start = int(dataset.episode_data_index["from"][0])
    episode_end = int(dataset.episode_data_index["to"][0])
    middle = episode_start + (episode_end - episode_start) // 2
    last_trainable = max(episode_start, episode_end - 1 - config.drop_n_last_frames)
    episode_last = episode_end - 1

    print_window(dataset, episode_start, "episode start")
    middle_sample = print_window(dataset, middle, "episode middle")
    print_window(dataset, last_trainable, "last normally sampled training anchor")
    print_window(dataset, episode_last, "physical episode end")
    check_normalization(middle_sample, dataset, config)

    plot_path = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "dp2"
        / f"episode_{args.episode:06d}_anchor_{middle:06d}.png"
    )
    save_window_plot(middle_sample, config, plot_path)
    print(f"saved plot: {plot_path}")


if __name__ == "__main__":
    main()
