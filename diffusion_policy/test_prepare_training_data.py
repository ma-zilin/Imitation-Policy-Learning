"""数据准备的离线测试；合成 episode 只用于测试，不冒充 Push-T 实测。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

import prepare_training_data as preparation
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.configs.types import FeatureType, PolicyFeature


def make_config():
    return DiffusionConfig(
        input_features={
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    )


class SyntheticEpisodes(Dataset):
    """六个互相独立的合成 episode，支持单帧视图和裁到边界的时间窗视图。"""

    def __init__(self, repo_id=None, root=None, episodes=None, delta_timestamps=None, **kwargs):
        self.episodes = list(episodes)
        self.delta_timestamps = delta_timestamps
        self.image_transforms = None
        self.features = {key: {} for key in [
            "observation.image", "observation.state", "action", "index", "episode_index", "frame_index"
        ]}
        self.meta = SimpleNamespace(camera_keys=["observation.image"])
        self.episode_data_index = {
            "from": torch.arange(len(episodes), dtype=torch.long) * 24,
            "to": (torch.arange(len(episodes), dtype=torch.long) + 1) * 24,
        }
        self.read_frames = []

        def select(indices):
            frames = [self.frame(*divmod(index, 24)) for index in indices]
            return {key: [frame[key] for frame in frames] for key in self.features}

        self.hf_dataset = SimpleNamespace(select=select)

    def __len__(self):
        return len(self.episodes) * 24

    def frame(self, position, frame):
        episode = self.episodes[position]
        grid = torch.linspace(0, 0.2, 96 * 96).reshape(96, 96)
        image = torch.stack([grid + 0.1 * (channel + 1) + episode * 0.02 + frame * 0.001 for channel in range(3)])
        return {
            "observation.image": image,
            "observation.state": torch.tensor([10 + episode * 40 + frame, 30 + episode * 20 + frame * 2]).float(),
            "action": torch.tensor([12 + episode * 50 + frame, 25 + episode * 30 + frame * 2]).float(),
            "index": torch.tensor(position * 24 + frame),
            "episode_index": torch.tensor(episode), "frame_index": torch.tensor(frame),
        }

    def _get_query_indices(self, index, position):
        start, end = position * 24, (position + 1) * 24
        indices, pads = {}, {}
        for key, times in self.delta_timestamps.items():
            raw = [index + round(time * 10) for time in times]
            indices[key] = [max(start, min(end - 1, item)) for item in raw]
            pads[f"{key}_is_pad"] = torch.tensor([item < start or item >= end for item in raw])
        return indices, pads

    def __getitem__(self, index):
        position, frame = divmod(index, 24)
        item = self.frame(position, frame)
        self.read_frames.append((self.episodes[position], frame))
        if self.delta_timestamps is not None:
            indices, pads = self._get_query_indices(index, position)
            for key, queries in indices.items():
                item[key] = torch.stack([self.frame(position, query % 24)[key] for query in queries])
            item.update(pads)
        return item


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="dp5-data-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = make_config()

    def source(self, episodes):
        return {"train_episodes": episodes, "raw_frame_count": len(episodes) * 24}

    def test_split_reproducibility_reuse_and_rng(self):
        before = torch.get_rng_state().clone()
        first = preparation.build_episode_split(list(range(206)), 0, 0.1, self.root / "split.json")
        second = preparation.build_episode_split(list(range(206)), 0, 0.1, self.root / "other.json")
        self.assertEqual(first, second)
        self.assertEqual(len(first["train_episodes"]), 185)
        self.assertEqual(len(first["val_episodes"]), 21)
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(first, preparation.build_episode_split(list(range(206)), 0, 0.1, self.root / "split.json"))
        with self.assertRaisesRegex(ValueError, "different settings"):
            preparation.build_episode_split(list(range(206)), 1, 0.1, self.root / "split.json")

    def test_invalid_and_overlapping_splits(self):
        for fraction in [0, 1, float("nan")]:
            with self.assertRaises(ValueError):
                preparation.build_episode_split([0, 1], 0, fraction, self.root / "split.json")
        with self.assertRaises(ValueError):
            preparation.build_episode_split([0], 0, 0.1, self.root / "split.json")
        split = preparation.build_episode_split([0, 1, 2], 0, 0.1, self.root / "split.json")
        split["val_episodes"] = [split["train_episodes"][0]]
        with self.assertRaises(ValueError):
            preparation.validate_split(split, [0, 1, 2])

    def test_raw_train_stats_exactness_reuse_and_provenance(self):
        raw = SyntheticEpisodes(episodes=[0, 2, 4])
        source = self.source(raw.episodes)
        path = self.root / "stats.json"
        stats = preparation.compute_training_stats(raw, self.config, source, path)
        self.assertEqual(set(raw.read_frames), {(episode, frame) for episode in raw.episodes for frame in range(24)})
        frames = [raw.frame(*divmod(index, 24)) for index in range(len(raw))]
        for key in self.config.input_features | self.config.output_features:
            values = torch.stack([frame[key] for frame in frames])
            dimensions = (0, 2, 3) if key == "observation.image" else (0,)
            shape = (3, 1, 1) if key == "observation.image" else (2,)
            expected = {
                "mean": values.mean(dim=dimensions).reshape(shape),
                "std": values.std(dim=dimensions, correction=0).reshape(shape),
                "min": values.amin(dim=dimensions).reshape(shape),
                "max": values.amax(dim=dimensions).reshape(shape),
            }
            for name in expected:
                torch.testing.assert_close(stats[key][name], expected[name], rtol=1e-5, atol=1e-6)
        with patch.object(preparation, "compute_stats", side_effect=AssertionError("Should reuse stats")):
            reused = preparation.compute_training_stats(raw, self.config, source, path)
        for key in stats:
            for name in stats[key]:
                torch.testing.assert_close(reused[key][name], stats[key][name], rtol=0, atol=0)
        changed = {**source, "new_file_hash": "changed"}
        with self.assertRaisesRegex(ValueError, "different provenance"):
            preparation.compute_training_stats(raw, self.config, changed, path)

    def test_reject_window_stats_and_degenerate_stats(self):
        raw = SyntheticEpisodes(episodes=[0])
        raw.delta_timestamps = {"action": [0]}
        with self.assertRaisesRegex(ValueError, "raw single frames"):
            preparation.compute_training_stats(raw, self.config, self.source([0]), self.root / "stats.json")
        raw.delta_timestamps = None
        stats = preparation.compute_training_stats(raw, self.config, self.source([0]), self.root / "stats.json")
        bad = copy.deepcopy(stats)
        bad["observation.image"]["std"].zero_()
        with self.assertRaisesRegex(ValueError, "Zero standard"):
            preparation.validate_stats(bad, self.config)
        bad = copy.deepcopy(stats)
        bad["action"]["max"] = bad["action"]["min"].clone()
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            preparation.validate_stats(bad, self.config)

    def test_loaders_windows_padding_and_validation_outliers(self):
        split = {"schema_version": 1, "dataset": preparation.DATASET_REPO_ID, "seed": 0,
                 "train_episodes": [0, 2, 4], "val_episodes": [1, 3, 5]}
        preparation.validate_split(split, list(range(6)))
        raw = SyntheticEpisodes(episodes=split["train_episodes"])
        stats = preparation.compute_training_stats(raw, self.config, self.source(raw.episodes), self.root / "stats.json")
        metadata = SimpleNamespace(root=self.root, fps=10)
        with patch.object(preparation, "LeRobotDataset", SyntheticEpisodes):
            loaders = preparation.build_dataloaders(metadata, self.config, split, 7, 0)
        self.assertTrue(loaders["train"].sampler.shuffle)
        self.assertFalse(loaders["val"].sampler.shuffle)
        before = torch.get_rng_state().clone()
        result = preparation.verify_data_pipeline(loaders, self.config, split, stats, 100)
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        for name in ["train", "val"]:
            self.assertEqual(result[name]["eligible_anchors"], 51)
            self.assertEqual(result[name]["checked_samples"], 51)
            self.assertEqual(result[name]["last_checked_shapes"]["action"], [2, 16, 2])
            self.assertGreater(result[name]["padded_action_positions"], 0)
        self.assertGreater(result["val"]["min_max_values_outside_unit_range"]["action"], 0)

    def test_complete_cli_reuse_and_changed_files(self):
        """主入口使用临时合成缓存；不接触实际 Hub 或真实数据集目录。"""
        metadata = SimpleNamespace(
            root=self.root, fps=10, total_episodes=6, video_keys=["observation.image"],
            episodes=[{"episode_index": episode, "length": 24} for episode in range(6)],
            get_data_file_path=lambda episode: Path(f"data/episode_{episode}.parquet"),
            get_video_file_path=lambda episode, camera: Path(f"videos/episode_{episode}.mp4"),
        )
        (self.root / "meta").mkdir()
        for name in ["info.json", "episodes.jsonl", "tasks.jsonl", "stats.json"]:
            preparation.write_json(self.root / "meta" / name, {"synthetic_test": True})
        for path in preparation.required_files(metadata, list(range(6))):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"Synthetic test placeholder, not a real Push-T episode")
        output = self.root / "output"
        argv = ["prepare_training_data.py", "--dataset-root", str(self.root), "--output-dir", str(output)]
        with patch("sys.argv", argv), \
             patch.object(preparation.PreTrainedConfig, "from_pretrained", return_value=self.config), \
             patch.object(preparation, "LeRobotDatasetMetadata", return_value=metadata), \
             patch.object(preparation, "LeRobotDataset", SyntheticEpisodes):
            preparation.main()
            report = json.loads((output / "verification_report.json").read_text())
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["train_episode_count"], 5)
            self.assertEqual(report["val_episode_count"], 1)
            split_bytes = (output / "split.json").read_bytes()
            stats_bytes = (output / "train_stats.json").read_bytes()
            # 保留真实 compute_stats 对象用于源码指纹，只监测是否重新计算。
            with patch.object(preparation, "compute_training_stats", wraps=preparation.compute_training_stats) as observed:
                preparation.main()
                self.assertEqual(observed.call_count, 1)
            self.assertEqual(split_bytes, (output / "split.json").read_bytes())
            self.assertEqual(stats_bytes, (output / "train_stats.json").read_bytes())
            changed = self.root / metadata.get_data_file_path(report["statistics_source"]["train_episodes"][0])
            changed.write_bytes(b"Changed synthetic cache")
            with self.assertRaises(SystemExit) as caught:
                preparation.main()
            self.assertEqual(caught.exception.code, 1)
            self.assertEqual(json.loads((output / "verification_report.json").read_text())["status"], "ERROR")
            self.assertEqual(stats_bytes, (output / "train_stats.json").read_bytes())

    def test_foreign_episode_rejected(self):
        class ForeignEpisode(SyntheticEpisodes):
            def __getitem__(self, index):
                item = super().__getitem__(index)
                item["episode_index"] = torch.tensor(999)
                return item

        split = {"seed": 0, "train_episodes": [0, 2], "val_episodes": [1]}
        raw = SyntheticEpisodes(episodes=[0, 2])
        stats = preparation.compute_training_stats(raw, self.config, self.source([0, 2]), self.root / "stats.json")
        with patch.object(preparation, "LeRobotDataset", ForeignEpisode):
            loaders = preparation.build_dataloaders(SimpleNamespace(root=self.root, fps=10), self.config, split, 2, 0)
        with self.assertRaisesRegex(ValueError, "foreign episode"):
            preparation.verify_data_pipeline(loaders, self.config, split, stats, 1)

    def test_misaligned_action_values_rejected(self):
        class WrongAction(SyntheticEpisodes):
            def __getitem__(self, index):
                item = super().__getitem__(index)
                item["action"] = item["action"] + 1
                return item

        split = {"seed": 0, "train_episodes": [0, 2], "val_episodes": [1]}
        raw = SyntheticEpisodes(episodes=[0, 2])
        stats = preparation.compute_training_stats(raw, self.config, self.source([0, 2]), self.root / "stats.json")
        with patch.object(preparation, "LeRobotDataset", WrongAction):
            loaders = preparation.build_dataloaders(SimpleNamespace(root=self.root, fps=10), self.config, split, 2, 0)
        with self.assertRaises(AssertionError):
            preparation.verify_data_pipeline(loaders, self.config, split, stats, 1)


if __name__ == "__main__":
    unittest.main()
