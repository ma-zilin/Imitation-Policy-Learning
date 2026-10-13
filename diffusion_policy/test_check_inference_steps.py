"""CPU 离线测试：验证自检能发现失效，不构成真实策略性能证据。"""

import unittest
from types import SimpleNamespace

import torch
from diffusers import DDPMScheduler
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionModel

from check_inference_steps import check_one_setting


class TinyUnet(torch.nn.Module):
    def __init__(self, nonfinite=False):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.1))
        self.nonfinite = nonfinite

    def forward(self, sample, timesteps, global_cond=None):
        return sample * (float("nan") if self.nonfinite else self.scale)


class TinyDiffusion(torch.nn.Module):
    conditional_sample = DiffusionModel.conditional_sample

    def __init__(self, config, nonfinite=False):
        super().__init__()
        self.config = config
        self.unet = TinyUnet(nonfinite)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")
        self.num_inference_steps = 100


def make_policy(nonfinite=False):
    config = SimpleNamespace(num_inference_steps=None, num_train_timesteps=100,
                             n_obs_steps=2, horizon=16, n_action_steps=8,
                             action_feature=SimpleNamespace(shape=(2,)))
    return SimpleNamespace(config=config, diffusion=TinyDiffusion(config, nonfinite),
                           unnormalize_outputs=lambda batch: batch)


class SamplingCheckTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.condition = torch.zeros(1, 132)

    def test_actual_loop_and_equal_initial_noise(self):
        policy = make_policy()
        before = policy.diffusion.unet.scale.detach().clone()
        full, noise100 = check_one_setting(policy, self.condition, 100, 0)
        reduced, noise50 = check_one_setting(policy, self.condition, 50, 0)
        self.assertEqual(full["unet_timesteps"], list(range(99, -1, -1)))
        self.assertEqual(reduced["unet_timesteps"], list(range(98, -1, -2)))
        self.assertEqual(reduced["complete_shape"], [1, 16, 2])
        self.assertEqual(reduced["selected_shape"], [1, 8, 2])
        self.assertEqual(reduced["selected_indices"], [1, 9])
        self.assertTrue(torch.equal(noise100, noise50))
        self.assertTrue(torch.equal(before, policy.diffusion.unet.scale))
        self.assertIsNone(policy.config.num_inference_steps)
        self.assertEqual(policy.diffusion.num_inference_steps, 100)
        self.assertFalse(policy.diffusion.unet._forward_pre_hooks)

    def test_nonfinite_fails_and_cleans_up(self):
        policy = make_policy(nonfinite=True)
        with self.assertRaisesRegex(RuntimeError, "NaN"):
            check_one_setting(policy, self.condition, 50, 0)
        self.assertIsNone(policy.config.num_inference_steps)
        self.assertEqual(policy.diffusion.num_inference_steps, 100)
        self.assertFalse(policy.diffusion.unet._forward_pre_hooks)

    def test_invalid_step_count_rejected(self):
        for steps in (0, -1, 101, 50.0, True):
            with self.assertRaises(ValueError):
                check_one_setting(make_policy(), self.condition, steps, 0)

    def test_runtime_ignoring_step_setting_fails(self):
        policy = make_policy()
        actual_sample = policy.diffusion.conditional_sample

        def ignore_setting(*args, **kwargs):
            policy.diffusion.num_inference_steps = 100
            return actual_sample(*args, **kwargs)

        policy.diffusion.conditional_sample = ignore_setting
        with self.assertRaisesRegex(RuntimeError, "调用"):
            check_one_setting(policy, self.condition, 50, 0)
        self.assertFalse(policy.diffusion.unet._forward_pre_hooks)

    def test_invalid_execution_slice_fails(self):
        policy = make_policy()
        policy.config.n_action_steps = 16
        with self.assertRaisesRegex(RuntimeError, "片段"):
            check_one_setting(policy, self.condition, 50, 0)


if __name__ == "__main__":
    unittest.main()
