# DP5 Runtime Verification

Verified on 2026-10-12 using the local `imitation-policy` environment. This record summarizes
fixed-batch learning and bounded training/resume/rollout checks, not a full paper reproduction.
Runtime checks passed; the final learner explanation of the loss/control boundary remains pending.

## Data and Configuration

- Dataset: `lerobot/pusht`, v2.0 commit `a16737ffb6c8dbe935291b9fd2f36d14c9bf0820`.
- All 416 required metadata/table/video files verified against the pinned remote revision.
- Episode split: 185 training / 21 validation; seed 0, validation fraction 0.1.
- Statistics: LeRobot `compute_stats`, all 23037 raw training frames, no windows or augmentation.
  Validation uses these same training-only statistics; repository-wide `meta.stats` is not substituted.
- Split SHA-256: `a8d7085836c143aec2ab7e93ab0d474cb1b78b3693a6840355902cf0b3a2b277`.
- Statistics artifact SHA-256: `9e0035fca0f8cb98a8a168b067434f994e91a61a0c8a7d9225759c1244a4fb94`.
- Config: `lerobot/diffusion_pusht@202c869`; observation/prediction/execution horizons 2/16/8.
- Batch shapes: image `(2,2,3,96,96)`, state `(2,2,2)`, expert actions `(2,16,2)`.
- Fresh network and backbone, epsilon targets, Adam LR 1e-4, gradient clip 10, batch size 2,
  training seed 0; no EMA, AMP or LR scheduler. Padding actions are not masked, matching the config.
- GPU: RTX 5070 Ti Laptop GPU; PyTorch 2.13.0+cu130. Detailed package and implementation
  fingerprints are recorded in local manifests, including the dirty LeRobot compatibility checkout.

Data preparation checked all eligible sampler anchors structurally, but decoded only two batches
per split. Training statistics cover all raw training frames; this does not imply every temporal
window was individually audited. Downloading assumes the DP2 v2.0 metadata cache already exists.

## Measured Results

| Check | Result | Interpretation |
| --- | --- | --- |
| DP5-A fixed expert batch, 200 updates | Same-batch MSE 1.304865 → 0.078235; ratio 0.059956 | Learning diagnostic passed; not held-out validation |
| Visual/UNet parameter changes | L2 changes 5.682455 / 21.235773; reload passed | Both modules updated, weights/statistics reloaded |
| DP5-B validation subset | MSE 1.328379 → 0.154085 | Four fixed windows, four noise/timestep draws per batch; not full validation |
| Zero-noise predictor | Validation-subset MSE 1.131266 | Diagnostic reference with the same evaluation draws |
| Numerical/resource checks | 200 finite training losses; first/last module gradient checks passed; peak allocated 5099.015 MiB | No observed allocated-memory growth trend; not a formal long-run stress test |
| Update throughput | 11.743 samples/s; 34.063 s for 200 updates | Includes data and computation; excludes validation/checkpoint writing |
| New-process resume equivalence | 1→2 resume equals uninterrupted 0→2: 221 weight tensors and 638 training-state tensors exactly equal | Checks Adam, RNG, sampler position, next batch and loss in the tested environment |
| New-process continuation | 200→201 updates passed; final subset MSE 0.154898 | Actual continuation; not a second bitwise-equivalence experiment |
| 200-step checkpoint rollout, seed 0 | 300 actions, 38 replans, coverage 0.184486, success False | Control chain works, task not solved; terminated False, truncated True |

The 300-step rollout also recorded mean/p95 steady replan latency of 1244.280/1251.915 ms
and mean/p95/max action delta of 40.700/129.077/273.688. These are single-run diagnostics,
not a failure-cause attribution. A same-architecture untrained baseline and a two-step trained
checkpoint also ran 16-step interface probes; neither was a full-episode quality evaluation.

## Commands and Local Evidence

Run from the project root in the original `imitation-policy` environment:

```bash
python diffusion_policy/download_pusht_data.py --download
HF_HUB_OFFLINE=1 python diffusion_policy/prepare_training_data.py
HF_HUB_OFFLINE=1 python diffusion_policy/overfit_fixed_batch.py
HF_HUB_OFFLINE=1 python diffusion_policy/train_smoke.py
HF_HUB_OFFLINE=1 python diffusion_policy/train_smoke.py \
  --resume outputs/dp5_smoke/20261012T032433_748841Z/checkpoints/step_000200 --steps 201
HF_HUB_OFFLINE=1 python diffusion_policy/rollout_checkpoint.py \
  outputs/dp5_smoke/20261012T032433_748841Z/checkpoints/step_000200 --seeds 0 --max-steps 300
HF_HUB_OFFLINE=1 python -m unittest discover -s diffusion_policy -p 'test_*.py' -q
```

The last command covers 17 offline tests; synthetic tests are not real-data training evidence.
The earlier two-step resume probe used `--steps 2 --eval-every 1 --checkpoint-every 1
--val-batches 1 --eval-draws 2 --log-every 1`, then resumed its step-1 checkpoint to step 2.
Full tensor/state comparison is recorded in its `resume_verification.json`.

Local evidence directories (ignored by Git; not downloadable repository links):

- `outputs/dp5_data/`: split, train-only statistics, provenance and bounded verification report.
- `outputs/dp5_fixed_batch/20261011T101422_976716Z/`: fixed-batch metrics, curve and checkpoint.
- `outputs/dp5_smoke/20261012T031624_285769Z/`: uninterrupted two-step reference.
- `outputs/dp5_smoke/20261012T031722_257234Z/`: resumed two-step run and equivalence report.
- `outputs/dp5_smoke/20261012T032433_748841Z/`: 200-step manifest, CSV and checkpoints.
- `outputs/dp5_smoke/20261012T032730_375676Z/`: 200→201 continuation.
- `outputs/dp5_rollout/20261012T033212_973362Z/`: 300-step metrics and failure video.

Small result summaries are versioned here; raw data, weights, optimizer state and videos remain
local. See [training instructions](../diffusion_policy/TRAINING_SMOKE.md) for checkpoint layout
and strict resume settings. Source fingerprints describe the tested code even when the run's
Git HEAD predates its uncommitted implementation.

## Boundary and Next Gate

Low epsilon MSE on four teacher-data windows is not proof of accurate generated trajectories
or closed-loop success under the policy's own visited states. The smoke-test PASS means finite
training/validation and functional save/resume/control interfaces, not a solved Push-T task.
DP6 budget, full-validation protocol, checkpoint-selection rule and formal multi-seed training
remain to be designed and run; no formal success-rate claim is made here.
