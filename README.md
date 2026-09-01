# Imitation Policy Learning

An evidence-driven learning portfolio for visuomotor imitation policies in robot manipulation.

The repository starts with Diffusion Policy on Push-T and will later add an ACT baseline under the same data and evaluation protocol. Its purpose is to connect policy-learning concepts to inspectable tensors, reproducible training runs, closed-loop rollouts, and controlled comparisons—not merely to reproduce a successful demo.

## Current Status

- **Completed:** DP0 — first-pass paper reading and the global Diffusion Policy map.
- **Next:** DP1 — evaluate a pretrained Push-T policy and verify the closed-loop interface.
- **Not yet completed:** local environment pinning, dataset inspection, training, multi-seed evaluation, and ACT comparison.

The active gate definitions and acceptance criteria are recorded in the [Diffusion Policy learning plan](plans/DIFFUSION_POLICY_LEARNING_PLAN.md).

## Project Questions

This project is organized around a small set of testable questions:

- How are observation windows and expert action chunks aligned at episode boundaries?
- How does a conditional denoising process generate temporally coherent actions?
- What changes between training-time noise prediction and deployment-time action sampling?
- How do observation, prediction, and execution horizons affect responsiveness and latency?
- When do offline losses disagree with closed-loop task performance?
- Under a shared Push-T protocol, how do Diffusion Policy and ACT differ in stability, latency, action smoothness, and failure modes?

## Learning and Experiment Path

```text
paper and system map
→ pretrained closed-loop evaluation
→ Push-T temporal alignment
→ formula-to-code tracing
→ fixed-batch overfit
→ short training smoke test
→ three-seed formal training
→ controlled closed-loop evaluation
→ ACT comparison
```

Formal Diffusion Policy evaluation will use at least 50 episodes per training seed. Reports will include success counts and rates, target-coverage scores, replanning latency, action smoothness, representative failures, and the exact configuration associated with each checkpoint. Training or validation loss alone is not treated as evidence of closed-loop control capability.

## Relationship to DeepLearningFoundations

The sibling `DeepLearningFoundations` project contains the prerequisite mechanism studies: CNNs, attention, point-mass behavior cloning, two-dimensional DDPM, and two-dimensional Flow Matching. This repository begins where those foundations become robot-policy experiments: image-conditioned action chunks, simulation rollouts, policy evaluation, and controlled algorithm comparisons. Completed foundation artifacts remain in their original repository.

## Repository Structure

```text
ImitationPolicyLearning/
├── act/                  # Future ACT baseline under the shared Push-T protocol
├── artifacts/            # Small plots, metrics, manifests, and failure analyses
├── common/
│   ├── data/             # Shared dataset inspection and temporal-alignment utilities
│   └── evaluation/       # Shared rollout metrics and evaluation protocol
├── configs/              # Versioned experiment configurations
├── diffusion_policy/     # Diffusion Policy inspection, training, and evaluation code
└── plans/                # Gate-based learning and experiment plans
```

Large datasets, checkpoints, raw rollout videos, environment caches, and external experiment logs are intentionally excluded from Git. Their source, revision, configuration, and reproduction instructions should be recorded instead.

## Reproducibility Policy

Before DP1 begins, the project will pin and record the LeRobot revision, Python version, PyTorch/CUDA stack, GPU, Push-T dataset revision when available, policy configuration, and complete commands. Setup commands are deliberately omitted until that environment is verified; an untested installation recipe would not be a reproducible interface.

Every formal experiment must preserve:

- code and dependency revisions;
- dataset identity and episode-level split;
- observation/action keys and all horizon values;
- normalization, scheduler, backbone, optimizer, and training budget;
- training and evaluation seeds;
- checkpoint-selection rule and success criterion;
- failed rollouts as well as successful ones.

## Scope Boundary

This is a learning and controlled-experiment repository, not a production robotics framework. The current scope excludes real-robot deployment, SO-101 integration, large VLA training, reinforcement-learning post-training, and claims of sim-to-real capability. Two-dimensional generative-model studies remain in `DeepLearningFoundations`; action-policy versions belong here.

## References

- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
- [Official Diffusion Policy repository](https://github.com/real-stanford/diffusion_policy)
- [LeRobot](https://github.com/huggingface/lerobot)
