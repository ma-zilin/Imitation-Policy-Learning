"""短程训练辅助设施离线测试；合成数据不是 Push-T 实验成果。"""

import copy
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import train_smoke as training


class TinyDataset(Dataset):
    def __len__(self):
        return 7

    def __getitem__(self, index):
        return {"action": torch.tensor([float(index)]), "index": torch.tensor(index)}


class TinySampler:
    indices = list(range(7))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class SmokeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def loader(self):
        return DataLoader(TinyDataset(), batch_size=2, sampler=TinySampler(), num_workers=0)

    def test_stream_resume_across_epoch_and_short_batch(self):
        stream = training.ResumableBatches(self.loader(), 3)
        stream.next()
        stream.next()
        saved = copy.deepcopy(stream.state_dict())
        expected = [stream.next()["index"] for _ in range(5)]
        resumed = training.ResumableBatches(self.loader(), 3, saved)
        actual = [resumed.next()["index"] for _ in range(5)]
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(expected[1].numel(), 1)
        with self.assertRaises(ValueError):
            training.ResumableBatches(self.loader(), 4, saved)
        saved["order"][0] = 999
        with self.assertRaises(ValueError):
            training.ResumableBatches(self.loader(), 3, saved)

    def test_stream_does_not_consume_global_torch_rng(self):
        torch.manual_seed(12)
        saved = torch.get_rng_state().clone()
        stream = training.ResumableBatches(self.loader(), 3)
        for _ in range(6):
            stream.next()
        torch.testing.assert_close(saved, torch.get_rng_state(), rtol=0, atol=0)

    def test_rng_round_trip_and_safe_torch_load(self):
        training.seed_all(123)
        saved = training.capture_rng()
        expected = (random.random(), np.random.rand(), torch.rand(3))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rng.pt"
            torch.save(saved, path)
            training.restore_rng(torch.load(path, weights_only=True))
        self.assertEqual(expected[0], random.random())
        self.assertEqual(expected[1], np.random.rand())
        torch.testing.assert_close(expected[2], torch.rand(3), rtol=0, atol=0)

    def test_validation_sample_weighting(self):
        batches = [{"action": torch.zeros(2, 16, 2)}, {"action": torch.zeros(1, 16, 2)}]
        metrics = [{"loss": 1., "zero_noise_mse": 2.}, {"loss": 4., "zero_noise_mse": 5.}]
        with patch.object(training, "evaluate", side_effect=metrics):
            result = training.validation(None, batches, 2, 100)
        self.assertEqual(result, {"loss": 2., "zero_noise_mse": 3., "batch_size": 3})
        with self.assertRaises(ValueError):
            training.validation(None, [], 2, 100)

    def test_validation_preserves_mode_and_rng(self):
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.diffusion = SimpleNamespace(noise_scheduler=SimpleNamespace(
                    add_noise=lambda clean, noise, times: clean + noise))

            def forward(self, batch):
                noise = torch.randn_like(batch["action"])
                times = torch.randint(0, 100, (noise.shape[0],))
                value = self.diffusion.noise_scheduler.add_noise(batch["action"], noise, times)
                return {"loss": value.square().mean()}

        policy = Policy().train()
        batch = {"action": torch.ones(2, 16, 2)}
        rng = training.capture_rng()
        first = training.validation(policy, [batch], 3, 123)
        second = training.validation(policy, [batch], 3, 123)
        self.assertTrue(policy.training)
        self.assertEqual(first, second)
        torch.testing.assert_close(rng["torch"], torch.get_rng_state(), rtol=0, atol=0)

    def test_adam_resume_with_stream_and_rng(self):
        training.seed_all(0)
        net = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        stream = training.ResumableBatches(self.loader(), 0)

        def step():
            batch = stream.next()
            optimizer.zero_grad(set_to_none=True)
            loss = (net(batch["action"]) - torch.randn_like(batch["action"])).square().mean()
            loss.backward()
            optimizer.step()
            return loss.detach().clone(), batch["index"].clone()

        step()
        state = copy.deepcopy({"weights": net.state_dict(), "optimizer": optimizer.state_dict(),
                               "stream": stream.state_dict(), "rng": training.capture_rng()})
        expected = step()
        expected_weights = copy.deepcopy(net.state_dict())
        net = torch.nn.Linear(1, 1)
        net.load_state_dict(state["weights"])
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        optimizer.load_state_dict(state["optimizer"])
        stream = training.ResumableBatches(self.loader(), 0, state["stream"])
        training.restore_rng(state["rng"])
        actual = step()
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        for name, value in net.state_dict().items():
            torch.testing.assert_close(value, expected_weights[name], rtol=0, atol=0)

    def test_checkpoint_completion_checksum_and_no_overwrite(self):
        from safetensors.torch import save_file

        class Policy(torch.nn.Linear):
            config = SimpleNamespace(type="diffusion")

            def save_pretrained(self, path):
                path.mkdir()
                training.write_json(path / "config.json", {"dummy": True})
                save_file(self.state_dict(), path / "model.safetensors")

        policy = Policy(1, 1)
        optimizer = torch.optim.Adam(policy.parameters())
        policy(torch.ones(1, 1)).sum().backward()
        optimizer.step()
        stream = training.ResumableBatches(self.loader(), 0)
        stream.next()
        rng = training.capture_rng()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            training.save_checkpoint(path, policy, optimizer, stream, 1, {"test": True}, {}, {})
            state = training.load_training_state(path)
            self.assertEqual(state["step"], 1)
            self.assertEqual(state["stream"], stream.state_dict())
            self.assertEqual(state["contract"], {"test": True})
            with self.assertRaises(FileExistsError):
                training.save_checkpoint(path, policy, optimizer, stream, 1, {}, {}, {})
            with (path / "policy/config.json").open("a") as file:
                file.write(" ")
            with self.assertRaises(ValueError):
                training.load_training_state(path)
        torch.testing.assert_close(rng["torch"], torch.get_rng_state(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
