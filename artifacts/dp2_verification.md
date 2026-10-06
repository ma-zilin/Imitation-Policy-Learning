# DP2 verification and roadmap audit

Audit date: 2026-10-06. This record distinguishes historical evaluation results from checks rerun during this audit.

## Learning route and boundaries

DP0 establishes the paper/system map. DP1 validates a pretrained LeRobot policy in a closed-loop environment. DP2 inspects how LeRobot constructs training inputs. DP3 traces conditional diffusion loss and verifies one backward pass; DP4 explains architecture and conditioning. DP5 prepares episode-level splits and training-only normalization, then checks learning on a fixed batch and short training run. DP6–DP8 cover formal training, controlled evaluation, and interpretation.

DP0 completion reflects the existing learning record, not a new unaided comprehension exam. No local Diffusion Policy training, full original-paper reproduction, or ACT comparison is claimed.

## DP1: historical results and current interface check

- Existing [CSV](dp1_pretrained_evaluation.csv): 10 seeds, 7 successes, mean maximum coverage 0.9440938523. Failed seeds: 0, 4, 5.
- The ten-episode evaluation was not rerun. Its video files are absent from current local outputs, so this audit does not independently revalidate video-based failure attribution.
- Current offline smoke check passed: cached checkpoint load on CUDA, finite action of shape `(1, 2)`, and one environment step. This checks interface health, not success rate.

## DP2: real data checks

Reproduction command, from the project root in the verified environment:

```bash
HF_HUB_OFFLINE=1 /home/wordwoods/miniconda3/envs/imitation-policy/bin/python diffusion_policy/inspect_pusht_dataset.py --episode 0 --local-files-only
```

The command passed. It requires the locally cached policy config, dataset metadata, episode table and video. Dataset: `lerobot/pusht`; repository metadata reports 206 episodes and 25,650 frames. Only episode 0 was decoded: 161 frames at 10 FPS. The cached dataset's exact upstream commit is not established by this audit; these results are local-cache checks, not proof that current Hub main behaves identically.

Configuration: observation horizon 2, prediction horizon 16, execution horizon 8; observation offsets `[-1, 0]`, action offsets `[-1, ..., 14]`, and `drop_n_last_frames=7`.

| Anchor frame | Observation frames | Action frames | Padded action positions (zero-based) |
| --- | --- | --- | --- |
| 0 | 0, 0 | 0, 0, 1 through 14 | 0 |
| 80 | 79, 80 | 79 through 94 | None |
| 153 | 152, 153 | 152 through 160, then seven repeats of 160 | 9 through 15 |
| 160 | 159, 160 | 159, 160, then fourteen repeats of 160 | 2 through 15 |

Additional assertions passed for all four anchors: query indices stay inside episode 0, image/state indices agree, observations are not from the future, and repeated boundary indices produce equal action values. Two complete windows stacked into a batch produce image `(2, 2, 3, 96, 96)`, state `(2, 2, 2)`, and action `(2, 16, 2)` tensors. This is a batch-construction check, not a training step.

At anchor 80, LeRobot normalization and unnormalization passed: maximum image reconstruction error `8.94e-08`, state/action error `0`; no nonfinite values or counted range violations in that sample. Image channels use mean/std normalization; state/action use min/max. Observed sample ranges need not equal the theoretical transformed bounds.

Statistics are loaded from repository `meta/stats.json`, not learned by the model and not recomputed when selecting an episode. Their provenance relative to a future training/validation split is not verified. Passing this check does not establish full-dataset validity or train-only statistics.

The [window figure](dp2_episode0_window.png) shows two observations and the expert action chunk, distinguishing past, intended execution, and unused future positions. This is an expert-data visualization, not a generated policy rollout. Excluding seven final anchors reduces padding but does not eliminate it; current training code does not mask padded action positions out of loss.

## Verified environment and compatibility

- Python environment: `/home/wordwoods/miniconda3/envs/imitation-policy`.
- GPU available: NVIDIA GeForce RTX 5070 Ti Laptop GPU.
- Policy: `lerobot/diffusion_pusht`, revision `202c869`.
- LeRobot checkout: `f495db0e1199325df5c62c1a7667378c90e4dc7e`, branch `dp1-compat`, plus the [DP2 compatibility patch](lerobot_dp2_compat.patch).

| Package | Installed version |
| --- | --- |
| torch | 2.13.0 |
| torchvision | 0.28.0 |
| datasets | 5.0.1 |
| av | 17.1.0 |
| matplotlib | 3.10.9 |
| gym-pusht | 0.1.5 |
| safetensors | 0.8.0 |
| diffusers | 0.40.0 |
| draccus | 0.11.6 |

The patch preserves dataset tensor stacking with current Hugging Face `Column` objects and provides the narrow video-reading interface required when torchvision lacks `VideoReader`. It records existing dependency changes; it does not alter policy mathematics. A clean-environment rebuild was not tested in this audit.

## Next: DP3

Read `DiffusionPolicy.forward()` and `DiffusionModel.compute_loss()` first. Map clean action, observation condition, diffusion timestep, sampled noise, noisy action, prediction target and loss to actual tensors. Then use LeRobot for one real-batch forward/backward check, verifying shapes, finite loss and finite gradients in the intended trainable modules. A full training run and new train/validation statistics remain out of scope until DP5.
