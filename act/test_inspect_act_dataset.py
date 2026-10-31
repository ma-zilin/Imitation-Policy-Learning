"""A2 的关键反例检查；这些合成数据不作为真实 Push-T 验证证据。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

import inspect_act_dataset as inspection


class ACTDataChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.spec, self.policy = inspection.load_config(
            inspection.PROJECT_ROOT / "configs/act/pusht_a2.json")

    def write_spec(self, directory, spec):
        path = Path(directory) / "config.json"
        path.write_text(json.dumps(spec))
        return path

    def test_act_offsets_do_not_add_observation_time_axis(self):
        offsets = inspection.build_delta_timestamps(self.policy, 10)
        self.assertEqual(set(offsets), {"action"})
        self.assertEqual(offsets["action"][0], 0)
        self.assertEqual(offsets["action"][-1], 1.5)

    def test_missing_frozen_field_and_external_weights_rejected(self):
        for modification, message in (("missing", "完整冻结"), ("weights", "骨干权重")):
            with self.subTest(modification=modification), tempfile.TemporaryDirectory() as directory:
                spec = copy.deepcopy(self.spec)
                if modification == "missing":
                    del spec["policy"]["kl_weight"]
                else:
                    spec["policy"]["pretrained_backbone_weights"] = "ResNet18_Weights.IMAGENET1K_V1"
                with self.assertRaisesRegex(ValueError, message):
                    inspection.load_config(self.write_spec(directory, spec))

    def test_last_complete_and_first_padded_windows(self):
        _, complete, mask = inspection.expected_window(145, 161, 16)
        self.assertEqual(complete[-1], 160)
        self.assertEqual(sum(mask), 0)
        _, partial, mask = inspection.expected_window(146, 161, 16)
        self.assertEqual(partial[-2:], [160, 160])
        self.assertEqual(mask, [False] * 15 + [True])

    def test_short_episode_never_borrows_next_episode(self):
        requested, actual, mask = inspection.expected_window(1, 3, 16)
        self.assertEqual(requested[-1], 16)
        self.assertEqual(actual, [1] + [2] * 15)
        self.assertEqual(mask, [False, False] + [True] * 14)

    def test_split_overlap_duplicates_and_missing_episode_rejected(self):
        valid = {"schema_version": 1, "dataset": "lerobot/pusht",
                 "train_episodes": [0, 1], "val_episodes": [2]}
        inspection.validate_split(valid, [0, 1, 2])
        for train, val in (([0, 1], [1, 2]), ([0, 0, 1], [2]), ([0], [2])):
            with self.subTest(train=train, val=val):
                broken = {**valid, "train_episodes": train, "val_episodes": val}
                with self.assertRaises(ValueError):
                    inspection.validate_split(broken, [0, 1, 2])

    def test_validation_frames_cannot_enter_training_statistics(self):
        split = {"train_episodes": [0], "val_episodes": [1]}
        source = {"train_episodes": [0], "raw_frame_count": 3, "fps": 10,
                  "method": "LeRobot compute_stats; all raw training frames; no temporal windows or augmentation"}
        inspection.validate_stats_source(source, split, {0: 3, 1: 4}, 10)
        source.update(train_episodes=[0, 1], raw_frame_count=7)
        with self.assertRaisesRegex(ValueError, "训练 episode"):
            inspection.validate_stats_source(source, split, {0: 3, 1: 4}, 10)

    def synthetic_stats(self):
        return {
            key: {"mean": torch.full(shape, 2.0), "std": torch.full(shape, 0.5),
                  "min": torch.zeros(shape), "max": torch.full(shape, 10.0)}
            for key, shape in (("observation.image", (3, 1, 1)),
                               ("observation.state", (2,)), ("action", (2,)))
        }

    def test_zero_standard_deviation_rejected(self):
        stats = self.synthetic_stats()
        stats["action"]["std"][0] = 0
        with self.assertRaisesRegex(ValueError, "标准差"):
            inspection.validate_stats(stats, {**self.policy.input_features, **self.policy.output_features})

    def test_normalization_uses_given_statistics_and_preserves_padding(self):
        stats = self.synthetic_stats()
        batch = {"observation.image": torch.full((2, 3, 96, 96), 0.5),
                 "observation.state": torch.full((2, 2), 2.0),
                 "action": torch.full((2, 16, 2), 3.0),
                 "action_is_pad": torch.tensor([[False] * 16, [False] + [True] * 15])}
        before = {key: value.clone() for key, value in batch.items()}
        result = inspection.check_normalization(batch, self.policy, stats,
                                                 self.spec["inspection"]["round_trip_atol"])
        self.assertAlmostEqual(result["observation.image"]["normalized_min"], -3)
        self.assertAlmostEqual(result["observation.state"]["normalized_max"], 0)
        self.assertAlmostEqual(result["action"]["normalized_max"], 2)
        for key in batch:
            self.assertTrue(torch.equal(batch[key], before[key]))

    def test_modified_statistics_file_rejected_before_data_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered_stats.json"
            path.write_text("{}")
            spec = copy.deepcopy(self.spec)
            spec["dataset"]["stats_path"] = str(path)
            with self.assertRaisesRegex(ValueError, "指纹改变：stats"):
                inspection.verify_provenance(spec)


if __name__ == "__main__":
    unittest.main()
