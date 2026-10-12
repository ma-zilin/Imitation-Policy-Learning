# DP5-B 短程训练

`train_smoke.py` 是训练入口，不重写 LeRobot 的视觉编码器、UNet 或扩散过程。
数据来自 `prepare_training_data.py` 已固定的 episode 划分；训练与验证使用同一份**仅训练集**统计量。
入口会重新检查文件校验值及统计量来源，不下载数据、不重新划分、不重新计算统计量。

## 数据与更新

每步从训练 DataLoader 读取一个新 batch，调用 `policy(batch)`，随后执行
`zero_grad → backward → 梯度裁剪 → optimizer.step`。
每个 epoch 对 LeRobot 提供的合格起点重新排列，不改变窗口内时间顺序。
噪声、扩散时间步和训练裁剪正常重采样；未屏蔽 padding 动作，与当前配置一致。

默认配置：随机初始化、batch size 2、200 次更新、Adam 学习率 1e-4、裁剪阈值 10。
**不使用 EMA、AMP 或学习率调度器**；这是小预算链路检查，不是完整论文训练配方。

验证固定使用验证 DataLoader 开头两个 batch（默认共 4 个窗口），中心裁剪，
每个 batch 评估 4 组固定且独立的噪声/时间步。按样本数加权汇总 MSE；
验证不更新参数、不改变训练模式或训练随机状态。
这不是整个验证集的泛化评估。初始网络 loss 和恒零噪声预测器 MSE 同时记录。

## 运行

在项目根目录、`imitation-policy` 环境运行：

```bash
HF_HUB_OFFLINE=1 python diffusion_policy/train_smoke.py
```

每 50 步记录验证并保存 checkpoint；终点也保存。输出使用独立时间戳目录，
不会覆盖原有实验。`metrics.csv` 包含训练/验证 loss、学习率、梯度范数、
样本来源、GPU 同步计时、吞吐及 allocated/reserved/peak 显存。
训练计时包含读取与计算，不包含验证和保存；首尾梯度诊断会额外增加耗时。
`manifest.json` 保存数据 revision、划分/统计量校验值、配置、依赖与源码指纹。
`summary.json` 中 PASS 仅表示数值与写入检查通过，不代表策略质量或 loss 必然下降。

## 保存与恢复

完整 checkpoint 的组成：

```text
step_000050/
├── policy/                # 配置、网络权重、归一化缓冲区；供推理使用
├── training_state.pt      # Adam、训练步数、RNG、epoch 排列和已读取位置
└── COMPLETE.json          # 完整性标记与文件校验值
```

仅保存 `policy/` 不足以恢复训练。当前 num_workers=0，无预取；
恢复直接从保存的排列/游标继续，无需重新解码已训练过的 batch。
恢复会核对数据、源码、依赖及关键训练配置，并在初始化完成后恢复随机状态。
不接受不完整或被改动的 checkpoint，也不覆盖旧输出。

```bash
HF_HUB_OFFLINE=1 python diffusion_policy/train_smoke.py \
  --resume outputs/dp5_smoke/<run>/checkpoints/step_000050 --steps 200
```

`--steps` 是包含已完成步数的**总目标步数**，不是额外更新数。
如果原实验改了 batch size、学习率、验证窗口数或评估采样数，恢复时须提供相同参数。
同环境下检查恢复一致性；不承诺跨硬件或不同软件版本逐位一致。
完整 checkpoint 包含大型 Adam 状态，约为单份模型权重的三倍；保存前检查磁盘空间。

## 闭环检查

```bash
HF_HUB_OFFLINE=1 python diffusion_policy/rollout_checkpoint.py \
  outputs/dp5_smoke/<run>/checkpoints/step_000200 --seeds 0
```

复用 DP1 的环境观测转换、动作缓存、coverage/success 判据、延迟、动作差及视频逻辑。
只加载本地策略，不使用公开预训练权重。结果写入独立 `outputs/dp5_rollout/` 目录，
不覆盖 DP1 指标或视频。需要快速检查接口时可加 `--max-steps 16`，
但人为缩短的回合不能用于报告正式成功率。运行完毕的 PASS 表示采样和控制链可运行，
即使 success=False 也可能通过该接口检查。

需要未训练同结构闭环对照时，在同一命令中添加 `--untrained --initialization-seed 0`。
这会保留 checkpoint 的归一化统计量，但重新初始化网络；不把训练后的参数作为对照。
比较时使用相同环境 seed 和回合步数，且不拿公开预训练策略当同预算训练对照。

## 离线测试

```bash
HF_HUB_OFFLINE=1 python -m unittest discover -s diffusion_policy -p test_train_smoke.py -v
```

覆盖跨 epoch 的数据恢复、尾部小 batch、随机状态往返、验证加权与模式恢复、
Adam 恢复更新一致性、checkpoint 完整性与禁止覆盖。合成测试不是真实数据训练证据。
