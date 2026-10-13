"""配对评测工具的 CPU 离线测试；合成 episode 不代表策略表现。"""

import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from diffusers import DDPMScheduler

import evaluate_inference_steps as evaluation


class EvaluationTests(unittest.TestCase):
    def test_plan_is_paired_bounded_and_alternates(self):
        plan = evaluation.make_plan()
        self.assertEqual(len(plan), 10)
        for seed in range(5):
            pair = plan[2 * seed:2 * seed + 2]
            self.assertEqual({row["inference_steps"] for row in pair}, {100, 50})
            self.assertTrue(all(row["environment_seed"] == row["sampling_seed"] == seed for row in pair))
        self.assertEqual([row["inference_steps"] for row in plan[:4]], [100, 50, 50, 100])

    def test_latency_excludes_first_and_no_steady_is_null(self):
        result = evaluation.latency_summary([1000, 10, 20])
        self.assertEqual(result["steady_mean_replan_latency_ms"], 15)
        self.assertEqual(result["steady_p95_replan_latency_ms"], 19.5)
        self.assertIsNone(evaluation.latency_summary([100])["steady_mean_replan_latency_ms"])
        self.assertEqual(evaluation.format_latency(None), "n/a")
        with self.assertRaises(RuntimeError):
            evaluation.latency_summary([float("nan")])

    def rows(self):
        return [
            {"environment_seed": row["environment_seed"], "inference_steps": row["inference_steps"],
             "initial_fingerprint": str(row["environment_seed"]), "success": row["environment_seed"] % 2 == 0,
             "max_coverage": 0.96, "episode_length": 10, "replan_count": 3,
             "replan_latencies_ms": [1000, row["inference_steps"], 2 * row["inference_steps"]],
             "mean_action_delta": 1.0}
            for row in evaluation.make_plan()
        ]

    def test_summary_pools_raw_steady_samples(self):
        summary = evaluation.summarize(self.rows())
        self.assertEqual(summary["groups"]["100"]["steady_replan_mean_ms"], 150)
        self.assertEqual(summary["groups"]["50"]["steady_replan_mean_ms"], 75)
        self.assertEqual(summary["pooled_steady_latency_ratio_50_to_100"], 0.5)
        self.assertEqual(summary["groups"]["100"]["steady_replan_samples"], 10)
        self.assertEqual(summary["total_environment_steps"], 100)

    def test_incomplete_duplicate_and_unpaired_rejected(self):
        rows = self.rows()
        for invalid in [rows[:-1], rows[:-1] + [rows[0]]]:
            with self.assertRaises(RuntimeError):
                evaluation.summarize(invalid)
        rows[-1]["initial_fingerprint"] = "different"
        with self.assertRaises(RuntimeError):
            evaluation.summarize(rows)

    def test_state_fingerprint_checks_physical_state(self):
        observation = {"pixels": np.zeros((2, 2, 3), dtype=np.uint8), "agent_pos": np.zeros(2)}
        info = {key: np.zeros(2) for key in ("pos_agent", "vel_agent", "block_pose", "goal_pose")}
        before = evaluation.initial_fingerprint(observation, info)
        info["block_pose"][0] = 1
        self.assertNotEqual(before, evaluation.initial_fingerprint(observation, info))

    def test_episode_reset_trace_and_success_rule(self):
        class Policy:
            def __init__(self):
                self.config = SimpleNamespace(num_inference_steps=100)
                self.diffusion = SimpleNamespace(
                    num_inference_steps=100, noise_scheduler=DDPMScheduler(num_train_timesteps=100))
                self.resets = 0

            def reset(self):
                self.resets += 1
                self._queues = {"action": deque()}

            def select_action(self, observation):
                if not self._queues["action"]:
                    self._queues["action"].extend([torch.ones(1, 2)] * 8)
                return self._queues["action"].popleft()

        class Env:
            metadata = {"render_fps": 10}
            success_threshold = 0.95

            def __init__(self):
                self.unwrapped = self
                self.steps = 0

            def reset(self, seed):
                self.seed = seed
                observation = {"pixels": np.zeros((2, 2, 3), dtype=np.uint8), "agent_pos": np.zeros(2)}
                info = {key: np.zeros(2) for key in ("pos_agent", "vel_agent", "block_pose", "goal_pose")}
                return observation, info

            def render(self):
                return np.zeros((4, 4, 3), dtype=np.uint8)

            def step(self, action):
                self.steps += 1
                return {}, 1.0, True, False, {"coverage": 0.96, "is_success": True}

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "episode"
            policy, env = Policy(), Env()
            with patch("evaluate_pretrained.prepare_observation", return_value={}), \
                    patch("evaluate_pretrained.seed_episode") as seed_all, \
                    patch("torch.cuda.synchronize"), \
                    patch.object(evaluation.imageio, "get_writer"):
                result = evaluation.evaluate_episode(
                    policy, env, {"environment_seed": 2, "sampling_seed": 2, "inference_steps": 50},
                    directory)
            seed_all.assert_called_once_with(2)
            self.assertEqual(policy.resets, 1)
            self.assertEqual(env.steps, 1)
            self.assertTrue(result["success"])
            self.assertEqual(len(result["scheduler_timesteps"]), 50)
            self.assertIsNone(result["steady_mean_replan_latency_ms"])
            self.assertTrue((directory / "trajectory.csv").is_file())
            self.assertEqual(json.loads((directory / "status.json").read_text())["status"], "COMPLETE")

    def test_metrics_and_json_are_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            evaluation.save_json(path / "status.json", {"status": "RUNNING"})
            evaluation.save_metrics(path / "metrics.csv", self.rows())
            self.assertTrue((path / "metrics.csv").is_file())
            self.assertFalse(list(path.glob("*.tmp")))
            with self.assertRaises(ValueError):
                evaluation.save_json(path / "bad.json", {"value": float("nan")})


if __name__ == "__main__":
    unittest.main()
