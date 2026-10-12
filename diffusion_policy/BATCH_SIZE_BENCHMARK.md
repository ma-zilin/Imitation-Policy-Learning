# Batch size 资源短测

`benchmark_batch_size.py` 用于 DP6 的预算准备，不是正式训练或模型质量对照。
复用已固定的 episode 划分、训练集统计量、LeRobot 网络、Adam 和梯度裁剪；
不修改 `train_smoke.py`，不会使已有 checkpoint 的源码恢复契约失效。
本次短测在路线调整前已经完成，保留为本机资源证据；新主线暂缓长程性能训练，
不要求重跑短测，也不将资源候选自动变成正式训练配置。

## 运行

在项目根目录、原 `imitation-policy` 环境中运行：

```bash
HF_HUB_OFFLINE=1 python diffusion_policy/benchmark_batch_size.py
```

默认顺序测试 batch size 2、4、8，每种配置在独立进程中从相同 seed 随机初始化。
每种执行 50 次完整更新，前 10 步预热，后 40 步计入稳定速度统计。
这里的预热用于排除启动阶段的计时影响，不是学习率 warmup；全程使用恒定学习率。
各进程正常结束或显存不足退出后，才启动下一配置，不并行争抢 GPU。
数据读取仍保持专家窗口内时间顺序；训练噪声、扩散时间步和裁剪正常重采样。
每种配置读取的窗口数不同，因此这不是控制训练量相同的学习效果实验。

计时顺序：

```text
CUDA 同步 → 开始计时 → 读取/搬运 batch
→ zero_grad → forward → backward → 有限梯度检查与裁剪 → Adam 更新
→ CUDA 同步 → 结束计时 → 指标查询/CSV 写入
```

稳态吞吐 = 预热后读取的窗口数 / 预热后更新耗时之和。
不包含进程启动、数据来源核验、模型初始化、显存查询或日志写入时间。
无验证、EMA、AMP、学习率调度器、checkpoint 保存或 rollout。
参数只在临时进程中更新，进程结束后不保留训练权重。

## 产物和判断

所有结果写入新的 `outputs/dp6_batch_benchmark/<run>/`，不会覆盖既有实验。

- `manifest.json`：批量大小、运行顺序、预热/总步数、余量阈值和脚本指纹。
- `batch_002/` 等：完整配置与数据/代码来源、逐步 CSV、单配置结果。
- `batch_002.log` 等：独立进程输出，保留 OOM 和普通错误。
- `comparison.csv`：均值/中位数/p95 更新耗时、吞吐及显存汇总。
- `summary.json`：各配置结果及满足余量条件的最快资源候选。

`peak_allocated_mib` 覆盖预热和正式计时；`steady_peak_allocated_mib` 只覆盖后者。
`peak_reserved_mib` 是 PyTorch allocator 保留显存，不能忽略这部分占用。
`minimum_free_mib` 来自每次更新后的 `mem_get_info`，包含其他进程/驱动影响，
但不是全机显存的连续监测。GPU 前后快照记录背景占用和温度。

每种配置只有完成全部更新且数值有限才记为 PASS。OOM 单独记录并允许测试其他配置；
数据、数值或其他运行错误会停止整轮测试，不能归类成“batch 太大”。
未找到有足够余量的配置时，整轮结果为 `NO_SAFE_CANDIDATE`，不报告资源选择通过。

候选筛选默认要求更新后最小空闲显存至少为设备总显存的 **15%**，
再在合格配置中选择吞吐最高者。这个阈值是明确记录的工程启发式，
不是加入 EMA、验证或其他模块后的显存保证；CPU 测试不产生 GPU 资源候选。
不使用 loss 排名，不修改正式训练配置，也不声称最大 batch 或最快配置泛化最佳。
单次短测会受温度、频率、系统缓存与后台任务影响；结果接近时应复测。

## 本机单次实测（2026-10-12）

运行目录：`outputs/dp6_batch_benchmark/20261012T082832_447252Z/`。
RTX 5070 Ti Laptop GPU、PyTorch 2.13.0+cu130；按默认命令顺序测试。
全部配置完成 50 次更新，后 40 次用于计时，无 OOM。

| batch size | 窗口/秒 | 平均每步耗时 | 全程峰值 allocated | 全程峰值 reserved | 更新后最小空闲显存 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 12.876 | 155.328 ms | 5100.163 MiB | 5654 MiB | 4702.375 MiB |
| 4 | 22.674 | 176.416 ms | 5099.772 MiB | 5672 MiB | 4696.438 MiB |
| 8 | 33.888 | 236.069 ms | 5103.741 MiB | 5684 MiB | 4683.875 MiB |

三种配置都满足本次 15% 余量条件，资源候选为 **batch size 8**，
其吞吐约为 batch size 2 的 2.63 倍。没有修改正式训练配置，也没有保存权重。
这些值不能直接代表加入 EMA/完整验证/保存后的吞吐或正式训练总时长；
峰值显存相近不意味着可以任意扩大 batch，尚未测量其他 batch 或分解峰值来源。

## 测试

```bash
HF_HUB_OFFLINE=1 python -m unittest discover -s diffusion_policy -p test_benchmark_batch_size.py -v
```

合成测试覆盖参数检查、排除预热、完整更新、尾部小 batch 拒绝、余量筛选、
OOM 与普通错误区分。它们不是实际 GPU 性能证据。
