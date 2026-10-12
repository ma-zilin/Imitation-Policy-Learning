"""资源短测的离线测试；小型合成网络不构成真实 GPU 性能证据。"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import benchmark_batch_size as benchmark


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def args(self, **changes):
        values = {"batch_sizes": [2, 4, 8], "steps": 50, "warmup": 10,
                  "num_threads": 1, "worker_timeout": 300, "seed": 0,
                  "min_headroom_fraction": 0.15}
        return SimpleNamespace(**(values | changes))

    def test_argument_validation(self):
        benchmark.validate_args(self.args())
        for change in [{"steps": 10}, {"warmup": 0}, {"batch_sizes": [2, 2]},
                       {"batch_sizes": [0]}, {"min_headroom_fraction": float("nan")}]:
            with self.assertRaises(ValueError):
                benchmark.validate_args(self.args(**change))

    def test_summary_excludes_warmup(self):
        rows = [{"step": i, "phase": "warmup" if i <= 2 else "measured",
                 "seconds": 100.0 if i <= 2 else 0.5, "batch_size": 4} for i in range(1, 6)]
        result = benchmark.summarize_rows(rows, 4, 5, 2)
        self.assertEqual(result["samples_per_second"], 8.0)
        self.assertEqual(result["mean_step_seconds"], 0.5)
        self.assertEqual(result["measured_updates"], 3)
        rows[-1]["batch_size"] = 1
        with self.assertRaises(ValueError):
            benchmark.summarize_rows(rows, 4, 5, 2)

    def test_resource_candidate_ignores_loss_and_rejects_low_headroom(self):
        results = [
            {"batch_size": 2, "status": "PASS", "device_total_mib": 100,
             "minimum_free_mib": 40, "samples_per_second": 10, "loss": 0.01},
            {"batch_size": 4, "status": "PASS", "device_total_mib": 100,
             "minimum_free_mib": 20, "samples_per_second": 15, "loss": 10},
            {"batch_size": 8, "status": "PASS", "device_total_mib": 100,
             "minimum_free_mib": 5, "samples_per_second": 20},
            {"batch_size": 16, "status": "OOM"},
        ]
        self.assertEqual(benchmark.resource_candidate(results, 0.15), 4)
        self.assertIsNone(benchmark.resource_candidate(results, 0.9))

    def test_complete_cpu_updates_change_weights(self):
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)

            def forward(self, batch):
                return {"loss": self.linear(batch["action"]).square().mean()}

        class Stream:
            def next(self):
                return {"action": torch.ones(2, 1), "action_is_pad": torch.zeros(2, dtype=torch.bool)}

        policy = Policy()
        initial = policy.linear.weight.detach().clone()
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
        rows = []
        benchmark.run_updates(policy, optimizer, list(policy.parameters()), Stream(),
                              SimpleNamespace(input_features={}), 3, 1, "cpu", 10, rows.append)
        self.assertEqual([row["phase"] for row in rows], ["warmup", "measured", "measured"])
        self.assertFalse(torch.equal(initial, policy.linear.weight))
        self.assertEqual(optimizer.state[policy.linear.weight]["step"].item(), 3)

    def test_worker_records_oom_and_errors_without_checkpoint(self):
        for error, status, code in [(torch.cuda.OutOfMemoryError("test OOM"), "OOM", 2),
                                    (RuntimeError("not OOM"), "ERROR", 3)]:
            with tempfile.TemporaryDirectory() as directory:
                args = self.args(device="cpu", output_dir=Path(directory) / "worker",
                                 worker_batch_size=2, data_dir=Path(directory))
                with patch.object(benchmark.PreTrainedConfig, "from_pretrained", side_effect=error):
                    result = benchmark.run_worker(args)
                self.assertEqual(result, code)
                saved = benchmark.json.loads((args.output_dir / "result.json").read_text())
                self.assertEqual(saved["status"], status)
                self.assertEqual(saved["completed_updates"], 0)
                self.assertFalse(list(args.output_dir.rglob("*.safetensors")))
                self.assertFalse(list(args.output_dir.rglob("*.pt")))


if __name__ == "__main__":
    unittest.main()
