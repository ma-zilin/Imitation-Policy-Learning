# Imitation Policy Learning

An evidence-driven learning portfolio for visuomotor imitation policies in robot manipulation.

The repository studies Diffusion Policy and Action Chunking with Transformers (ACT) on Push-T through mechanism learning and bounded experiments. It connects policy-learning concepts to inspectable tensors, training updates, closed-loop rollouts, and controlled comparisons. Shared data and evaluation conventions support interpretation; differences in checkpoint provenance and training budget are recorded explicitly.

## Current Status

- **Completed:** DP0 — first-pass paper reading and the global Diffusion Policy map; DP1 — pinned pretrained Push-T evaluation; DP2 — real-episode inspection of temporal windows, padding, batching, and normalization; DP3 — conditional diffusion training-path tracing and a real-batch forward/backward check; DP4 — architecture/data-flow study, inference-path tracing, conditioning and CNN/Transformer comparisons, and guided understanding checks.
- **DP5 closed:** runtime checks passed for fixed-batch learning, episode-level data preparation, a 200-step training smoke test, full-state resume, and local-checkpoint rollouts. A guided conceptual check distinguished expert-window noise MSE from observations produced by the policy's own executed actions; this is not a claim of reliable task performance or independent mastery of every topic.
- **DP6–DP8 closed:** the [training protocol](artifacts/dp6_training_protocol.md), one bounded pretrained-policy inference comparison, and [system/evidence synthesis](artifacts/dp8_closeout.md) are complete under the mechanism-study scope. Understanding was checked through guided questions, not independent written mastery of every topic or a full performance reproduction.
- **DP7 comparison:** the same pinned pretrained policy was evaluated with 100 versus 50 inference diffusion steps on five paired environment/sampling seeds. Success counts were 3/5 and 5/5; pooled mean steady replanning latency was approximately 1318 ms and 656 ms. These are small-sample observations, not proof that 50 steps are universally better.
- **Foundation milestone completed:** two-dimensional Conditional Flow Matching F0–F6 closed on 2026-10-26 in `DeepLearningFoundations`; its [closeout index](../DeepLearningFoundations/FlowMatching/artifacts/f6_closeout.md) records conditional generation, bounded evaluation, and comparison limits. This is a sibling-repository reference, not evidence of a robot Flow Matching policy.
- **Current learning target:** ACT A0 — problem framing and the system map. The [ACT learning plan](plans/ACT_LEARNING_PLAN.md) and [working-directory entry](act/README.md) are initialized; no ACT gate or experiment has been completed. The path includes a separate minimal VAE/CVAE unit before implementation tracing.
- **Agreed ACT budget:** one training seed (0), at most 500 fixed-batch updates, 200 real-DataLoader updates, and one local-checkpoint rollout with environment seed 0 capped at 300 steps. These are future limits, not executed results. Reliable task performance is not the milestone exit criterion.
- **Deferred optional extension:** long-run training from scratch and formal three-training-seed performance evaluation. These are not required for the current learning milestone and have not been completed. Full-validation generalization and reliable task performance of the locally trained policy remain unverified; ACT comparison is also unfinished.

| Evidence | Result | Boundary |
| --- | --- | --- |
| [DP1 evaluation](artifacts/dp1_pretrained_evaluation.csv) | 7/10 successful episodes; mean maximum coverage 0.9441 | Historical pretrained-policy evaluation, not local training |
| [DP2 verification](artifacts/dp2_verification.md) | Four boundary/interior windows checked; normalization round-trip passed | One decoded episode, not a full-dataset audit |
| [DP3 training-chain check](diffusion_policy/check_training_step.py) | MSE agreement and finite, nonzero gradients in the visual encoder and denoising U-Net | Pretrained weights; one backward pass, no parameter update |
| [DP4 study record](plans/DIFFUSION_POLICY_LEARNING_PLAN.md#当前进度) | Architecture, inference flow, conditioning comparisons, and guided checks recorded | Reading/understanding completion; no new runtime experiment |
| [DP5 verification](artifacts/dp5_verification.md) | Fixed-batch MSE 1.304865 → 0.078235; held-out subset MSE 1.328379 → 0.154085; new-process resume passed | Validation uses only 4 fixed windows; the 200-step policy failed its single 300-step rollout (maximum coverage 0.184486) |
| [DP6 protocol](artifacts/dp6_training_protocol.md) | Training control, full-state recovery, validation comparability, and configuration-use checks recorded | Candidate warmup/cosine and best-selection rules were not executed |
| [DP7 inference comparison](artifacts/dp7_inference_steps.md) | 100/50 steps: 3/5 vs 5/5 successes; pooled mean steady latency 1317.90 vs 655.69 ms | Five paired seeds, unchanged pretrained weights; not local training or real-time control validation |
| [DP8 closeout](artifacts/dp8_closeout.md) | End-to-end data/control flow and guided conceptual checks summarized | FM was only an introductory bridge at DP closeout; its later 2D completion is recorded in the sibling project |
| [ACT plan](plans/ACT_LEARNING_PLAN.md) | A0–A7 scope and first-run limits documented | Planning only; no ACT implementation or runtime evidence yet |

DP7 reran both configurations under its own fixed protocol rather than reusing the historical DP1 results as a paired baseline. All ten episodes have local metrics, action/coverage traces, and videos; raw outputs remain excluded from Git.

The completed DP milestone, historical gate definitions, and acceptance boundaries are recorded in the [Diffusion Policy learning plan](plans/DIFFUSION_POLICY_LEARNING_PLAN.md). The active learning authority is now the [ACT plan](plans/ACT_LEARNING_PLAN.md).

## Project Questions

This project is organized around a small set of testable questions:

- How are observation windows and expert action chunks aligned at episode boundaries?
- How does a conditional denoising process generate temporally coherent actions?
- What changes between training-time noise prediction and deployment-time action sampling?
- How do observation, prediction, and execution horizons affect responsiveness and latency?
- When do offline losses disagree with closed-loop task performance?
- As a future comparison, under a shared Push-T protocol, how do Diffusion Policy and ACT differ in stability, latency, action smoothness, and failure modes?

## Learning and Experiment Path

```text
paper and system map
→ pretrained closed-loop evaluation
→ Push-T temporal alignment
→ conditional training-path tracing and one backward pass
→ policy architecture and condition injection
→ fixed-batch overfit
→ short training smoke test
→ training-flow and experiment-design study
→ one bounded pretrained-policy inference comparison
→ synthesis and evidence boundaries
→ 2D Conditional Flow Matching (in DeepLearningFoundations)
→ ACT system map and minimal VAE/CVAE foundations (current entry)
→ ACT bounded training, execution checks, and synthesis (planned)
```

The completed controlled comparison varied only inference diffusion steps (100/50), keeping the pretrained checkpoint, preprocessing, horizons (2/16/8), environment, seeds, and success criterion fixed. Each of the ten episodes was capped at 300 environment steps; the actual total was 2218. Reports preserve every episode, first/steady replanning latency, coverage, termination/truncation, action behavior, and video. Small-sample observations do not establish general superiority, and failure causes remain unassigned without further evidence.

Long-run training and performance reproduction are deferred optional work, not a hardware-impossibility claim: local parameter updates and resource checks have already run. If explicitly resumed, the formal extension will freeze the training budget and checkpoint-selection protocol, train three seeds, and evaluate at least 50 episodes per training seed. It is not part of the current completion requirement. Training or validation loss alone is not evidence of closed-loop control capability, and pretrained results are not a same-budget local-training baseline. ACT is the current learning stage; its bounded checks will support mechanism and interface comparisons, not a claim of algorithmic superiority.

## Relationship to DeepLearningFoundations

The sibling `DeepLearningFoundations` project contains the prerequisite mechanism studies: CNNs, attention, point-mass behavior cloning, two-dimensional DDPM, and the completed two-dimensional Conditional Flow Matching milestone. This repository connects those foundations to image-conditioned action chunks, simulation rollouts, policy evaluation, and controlled comparisons. Learning now returns here for ACT; completed foundation artifacts remain in their original repository.

## Repository Structure

```text
ImitationPolicyLearning/
├── act/                  # Active ACT learning entry; implementation follows its gates
├── artifacts/            # Small plots, metrics, manifests, and failure analyses
│   └── act/              # Future bounded ACT verification records
├── common/
│   ├── data/             # Shared dataset inspection and temporal-alignment utilities
│   └── evaluation/       # Shared rollout metrics and evaluation protocol
├── configs/              # Versioned experiment configurations
│   └── act/              # ACT configuration snapshots, frozen before execution
├── diffusion_policy/     # Diffusion Policy inspection, training, and evaluation code
└── plans/                # Gate-based learning and experiment plans
```

Large datasets, checkpoints, raw rollout videos, environment caches, and external experiment logs are intentionally excluded from Git. Their source, revision, configuration, and reproduction instructions should be recorded instead.

## Reproducibility Policy

DP1 established a pinned compatibility path. Before rerunning it or entering later gates, recheck and record the LeRobot revision, Python version, PyTorch/CUDA stack, GPU, Push-T dataset revision when available, policy configuration, and complete commands. A previously working environment is evidence, not a guarantee that a changed checkout or dependency set remains compatible.

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
- [ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
- [Official ACT repository](https://github.com/tonyzhaozh/act)

## License

Original code and documentation in this repository are released under the [MIT License](LICENSE). Third-party code, datasets, and model weights remain subject to their respective licenses.
