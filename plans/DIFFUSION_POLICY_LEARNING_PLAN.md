---
type: learning-plan
topic: diffusion-policy
status: active
current_gate: DP1
started: 2026-09-01
progress_mode: gate-based
---

# Diffusion Policy 学习计划

## 目标与当前位置

本阶段的目标不是“调用一个现成策略得到视频”，也不是从零复刻整个官方仓库，而是建立一条能够被代码、实验和闭环评测共同验证的主线：

> 专家 episode → observation/action 时间窗 → 条件动作扩散训练 → action chunk 反向采样 → receding-horizon 执行 → Push-T 闭环评测

开始本计划时，以下前置内容已经完成：

- 二维 DDPM：前向加噪、时间条件噪声预测、反向采样和分布评测。
- Behavior Cloning B0–B6：episode 数据、监督训练、开环/闭环区别、covariate shift 和 action chunk 概念。
- D2L CNN 第 6–7 章。

D2L 第 10 章作为本计划的理论支线同步学习，但 **Diffusion Policy 是唯一项目级编码主线**。

## 学习原则

- 学习者亲手运行实验、追踪张量并完成必要的最小实现。
- Codex 负责解释论文、物理直觉、公式—代码映射、实验设计和验收。
- 除非学习者明确要求，否则 Codex 不直接代写核心学习实现。
- 先验证数据和闭环接口，再开始昂贵训练。
- 先做单 seed smoke test，再做正式多 seed 实验。
- 不用 training loss 代替闭环表现，不只保留最好看的 rollout。
- 每一关只解决一个主要问题；前一关没有形成证据，不进入下一关。

## 学习与实现边界

### 本阶段必须完成

- 解释 Diffusion Policy 为什么仍然属于 Behavior Cloning。
- 构造对齐的 observation window 与 action chunk。
- 将二维 DDPM 的变量映射到条件动作序列扩散。
- 读懂视觉编码器、条件注入、时间编码和 1D 去噪网络的数据流。
- 跑通预训练 Push-T 策略的闭环评测。
- 完成 fixed-batch overfit 和短程训练 smoke test。
- 正式训练并按统一协议评测多个随机种子。
- 记录成功率、最大目标覆盖度、推理延迟、动作平滑性和失败轨迹。

### 本阶段暂不展开

- Flow Matching、ACT、SmolVLA 或其他策略的正式训练。
- 真机、SO-101、ROS2、MoveIt2 或 Sim-to-Real。
- 从零手写 ResNet、1D U-Net、Transformer block 或 diffusion scheduler。
- 修改 Push-T 任务、采集新机器人数据或追求完整论文表格。
- DAgger、在线纠错、强化学习后训练。
- 在 CNN 与 Transformer 两种 Diffusion Policy backbone 上同时开展完整调参。

## 权威来源与工具边界

概念与方法以以下来源为准：

1. [Diffusion Policy 论文](https://arxiv.org/abs/2303.04137)：问题、方法、horizon、架构与实验结论。
2. [论文项目页](https://diffusion-policy.cs.columbia.edu/)：可视化、补充材料和公开结果。
3. [原作者官方仓库](https://github.com/real-stanford/diffusion_policy)：原始配置、workspace、policy、runner 和评测结构。

最小 Push-T 复现优先使用 [LeRobot](https://github.com/huggingface/lerobot) 的数据、Diffusion Policy 和评测接口。原因是它能把 dataset、policy 和 simulation evaluation 串成较小的工程闭环。原作者仓库用于理解原始设计和核对差异，不要求一开始完整搭建。

LeRobot API、命令和默认配置可能变化。进入 DP1 前必须记录：

- LeRobot commit 或发布版本。
- Python、PyTorch、CUDA 和 GPU 信息。
- `lerobot/pusht` 数据集 revision（若接口支持固定）。
- 实际 policy config 与完整启动命令。

不得把 LeRobot 的复现结果写成原作者官方实现结果。若二者存在 EMA、normalization、scheduler、horizon、视觉 backbone 或评测实现差异，必须在报告中单独列出。

## 核心数据流

### 训练

```text
专家 episode
→ 选取当前决策时刻 t
→ 读取 observation window O_t
→ 读取对齐的干净 action chunk A_t^0
→ 对 action chunk 采样 diffusion timestep k 和噪声 ε
→ 得到 noisy action chunk A_t^k
→ 结合 O_t 与 k 预测 ε
→ epsilon MSE
```

### 推理与控制

```text
最近的 observation window O_t
→ 初始化高斯噪声 action chunk A_t^K
→ 条件反向去噪 K → 0
→ 得到预测 action chunk
→ 只执行其中一个前缀
→ 获取新的 observation
→ 重新规划
```

核心边界：**被扩散的是动作序列；图像和机器人状态作为条件，通常不随动作一起加噪。**

## 核心公式与张量

记：

- $B$：batch size。
- $T_o$：observation horizon。
- $T_p$：prediction horizon。
- $T_a$：一次规划后实际执行的 action horizon。
- $D_a$：action dimension。
- $K$：diffusion timesteps 数量。

干净动作块：

$$
A_t^0\in\mathbb{R}^{B\times T_p\times D_a}
$$

对每个 batch 样本采样扩散时间步 $k$ 和噪声：

$$
\epsilon\sim\mathcal{N}(0,I),
\qquad
A_t^k
=
\sqrt{\bar\alpha_k}A_t^0
+
\sqrt{1-\bar\alpha_k}\epsilon
$$

条件噪声预测：

$$
\hat\epsilon
=
\epsilon_\theta(A_t^k,k,O_t)
$$

训练目标：

$$
\mathcal{L}_{DP}
=
\mathbb{E}
\left[
\left\|\epsilon-\epsilon_\theta(A_t^k,k,O_t)\right\|_2^2
\right]
$$

不能只记公式。进入训练前必须打印并解释真实 batch 中：

- 图像 observation 的形状。
- 状态 observation 的形状。
- action chunk 的形状。
- diffusion timestep 的形状与取值范围。
- noise、noisy action 和 predicted noise 的形状。
- padding mask（若存在）的形状与语义。

## 三种 horizon

- **Observation horizon $T_o$**：策略一次决策能看到多少个最近 observation。
- **Prediction horizon $T_p$**：策略一次生成多长的动作序列。
- **Action/execution horizon $T_a$**：生成后实际执行多少步才重新观察和规划。

通常有：

$$
T_a\leq T_p
$$

$T_a$ 较短，策略能更快响应新 observation，但需要更频繁地执行扩散采样；$T_a$ 较长，推理调用更少，但开环执行更久、对扰动响应更慢。

本计划不预先硬编码三个 horizon 的数值。DP2 必须从固定版本的实际配置与 dataset sampler 中读取，并验证索引，而不是只相信变量名。

---

## DP0：论文第一遍与全局映射

### 本轮问题

1. 普通 MSE BC 在多峰动作分布上可能出现什么问题？
2. Diffusion Policy 学习的是 reward、action，还是 action distribution？
3. observation 如何影响动作去噪？
4. 为什么生成 action chunk，而不是只生成下一步动作？
5. 为什么只执行 chunk 的一部分后要重新规划？

### 阅读

第一遍只读：

- Abstract。
- Figure 1、Figure 2、Figure 3。
- Introduction。
- Method 中与 action diffusion、visual conditioning、receding horizon 有关的部分。
- 论文中关于多模态动作和 action sequence prediction 的实验分析。

第一遍暂不逐项研究全部 benchmark、超参数表和真实机器人细节。

### 必须产物

- 一张训练数据流图。
- 一张推理/闭环数据流图。
- 一张“二维 DDPM → Diffusion Policy”映射表。
- 对上述五个问题的书面回答。

### 通过标准

- 能明确区分 diffusion timestep、环境时间步和 action chunk 内部时间索引。
- 能说明训练时有专家 action，部署时没有专家 action。
- 能解释 Diffusion Policy 改变了 BC 的策略分布表达方式，但没有自动消除 covariate shift。
- 能不看资料画出本计划的两条核心数据流。

## DP1：先评测预训练策略，打通闭环接口

### 为什么先评测

在训练前先运行预训练策略，可以独立验证环境、observation key、normalization、action queue、渲染、视频和评测指标。否则训练结束后失败，无法判断是 checkpoint、数据还是闭环接口的问题。

### 实验

1. 固定 LeRobot 版本并记录环境信息。
2. 加载公开的 Push-T Diffusion Policy checkpoint。
3. 先运行 1 个 episode，确认：
   - observation key 和形状正确；
   - action 维度与环境一致；
   - normalization/denormalization 正常；
   - episode 能终止；
   - 视频和指标能够保存。
4. 再用固定 seed 集合运行至少 10 个诊断 episode。
5. 记录每次 policy replan 的耗时，而不只记录整个 episode 用时。

### 指标

- 最大目标覆盖度或环境提供的连续得分。
- 成功次数/总次数；成功阈值在评测前固定。
- episode 长度。
- 单次 replanning 推理延迟的均值与高分位数。
- action 相邻步变化的简单平滑性指标。
- 成功与失败视频。

### 通过标准

- checkpoint 可以在新进程中加载并完成闭环 rollout。
- 能从代码指出 action chunk 在何处生成、缓存、取出和清空。
- 能说明环境 reward/score 与二值 success 的区别。
- 失败 episode 被保留，且能区分模型失败和评测接口失败。

## DP2：理解 Push-T 数据与时间窗

### 学习内容

- episode、frame、timestamp 和控制频率。
- observation window 与 action chunk 的索引关系。
- episode 边界处的 padding 策略。
- 图像、状态和动作的 normalization。
- 为什么 split 应按 episode，而不是随机拆散 frame。

### 实验

1. 只取一个 episode，画出 observation/action 时间轴。
2. 分别检查 episode 开头、中间和结尾的一个训练样本。
3. 打印该样本所有 observation/action 索引和时间戳。
4. 可视化 observation 图像，并把 action chunk 叠加或单独画成二维轨迹。
5. 检查 padding mask 或边界重复策略。
6. 确认 normalization statistics 只来自训练数据定义的统计范围。

### 必须回答

1. 为什么 action chunk 可能包含当前时刻之前或之后的索引？
2. 哪几步动作会参与 loss，哪几步会在环境中实际执行？
3. episode 末尾不足 $T_p$ 时如何处理？
4. 图像和低维状态如何在时间维上对齐？
5. 为什么 horizon 错一位可能仍能训练，却导致闭环性能异常？

### 通过标准

- 能任选一个真实 sample，逐项说出其来源 episode、时间索引和物理含义。
- 能画出实际配置中的 $T_o$、$T_p$、$T_a$，而不是只背定义。
- 没有跨 episode 拼接或未来 observation 泄漏。
- 能说明 normalize → policy → unnormalize → environment 的边界。

## DP3：把二维 DDPM 迁移到条件动作序列

### 最小机制实验

在正式视觉训练前，从 Push-T dataset 取一个真实 batch：

1. 得到干净动作块 $A^0$。
2. 为每个样本独立采样 diffusion timestep $k$。
3. 生成同形状噪声 $\epsilon$。
4. 使用 scheduler 得到 $A^k$。
5. 将 observation condition、$A^k$ 和 $k$ 输入 policy。
6. 检查 predicted noise 与 target noise 的形状。
7. 计算单次 loss 并反向传播。

### 公式—代码映射表

必须在实际代码中定位：

| 数学对象 | 代码中的真实对象 |
| --- | --- |
| $A^0$ | 干净 action chunk |
| $k$ | batch diffusion timestep |
| $\epsilon$ | sampled noise |
| $A^k$ | noisy action chunk |
| $O_t$ | image/state observation condition |
| $\epsilon_\theta$ | conditional denoiser |
| $\mathcal{L}_{DP}$ | noise prediction loss |

### 通过标准

- 所有关键张量形状被打印并解释。
- 每个 batch 样本可以具有不同的 $k$。
- 只有 action chunk 被 scheduler 加噪。
- 一次 backward 后，视觉编码器和去噪网络中预期训练的参数具有有限梯度。
- 能解释训练为何只采一个随机 $k$，推理为何要迭代多个 $k$。

## DP4：读懂策略结构与条件注入

### 阅读顺序

不要从训练入口逐行漫游整个仓库。沿一个 batch 的真实调用链阅读：

```text
config
→ dataset sample
→ preprocessing/normalization
→ policy.forward
→ visual/state encoder
→ noisy action + timestep + condition
→ denoiser
→ loss
```

然后沿一次推理调用链阅读：

```text
select_action
→ observation queue
→ initialize noisy action chunk
→ scheduler reverse steps
→ denormalize
→ action queue
→ return one action
```

### 必须理解

- CNN 如何把图像变成 condition feature。
- 低维 state 在何处与视觉特征融合。
- timestep embedding 表示的是 diffusion noise level，而不是环境时间。
- 1D temporal network 如何沿 action horizon 处理序列。
- global conditioning 与序列 token conditioning 的差别。
- train-time `forward` 与 deployment-time `select_action` 为什么不是同一条执行路径。

### 与 Ch10 的并行边界

Ch10 同期只完成：

- 用张量写一次 scaled dot-product attention。
- 解释 Q/K/V、$\sqrt{d_k}$、mask 和 attention weight 形状。
- 调用一次 `nn.MultiheadAttention`。
- 区分 self-attention、causal attention 和 cross-attention。

如果当前正式 baseline 使用 CNN/1D U-Net，不为了练 Ch10 强行改成 Transformer。Transformer 只做结构对照和未来 VLA 前置知识。

### 通过标准

- 能画出实际 baseline 的模块图和张量形状。
- 能指出 observation condition 注入去噪网络的具体位置。
- 能说明 1D U-Net 中的“1D”沿哪个维度卷积。
- 能区分 action sequence 的物理时间维与 feature/channel 维。
- 能解释 CNN baseline 与 Transformer variant 的主要归纳偏置差异，但不要求同时训练二者。

## DP5：fixed-batch overfit 与训练 smoke test

### DP5-A：fixed-batch overfit

1. 固定一个很小的真实 batch。
2. 固定随机种子，并明确噪声与 timestep 是每步重采样还是固定；两种选择的目的不同，必须记录。
3. 反复优化同一 batch。
4. 记录 loss、梯度范数和预测噪声误差。
5. 保存并重新加载 checkpoint。

fixed-batch overfit 只证明训练管线具有学习能力，不证明数据泛化或闭环控制成立。

### DP5-B：短程 smoke test

1. 使用正式 dataloader 进行短程训练。
2. 检查吞吐、显存、checkpoint、恢复训练和日志。
3. 用固定小型验证 batch 记录 validation loss。
4. 对中间 checkpoint 运行少量 rollout，确认采样和控制链未损坏。

### 诊断基线

- 未训练同结构 policy 的 loss/rollout。
- 恒零 noise predictor 的 noise MSE。
- 公开预训练 checkpoint 的闭环结果，仅用于验证评测上限与接口，不作为同预算训练对照。

### 通过标准

- fixed batch loss 可以显著下降。
- 正式数据训练无 `NaN/Inf`，显存稳定。
- checkpoint 在新进程中可以恢复训练和 rollout。
- 能解释低 noise MSE 为什么仍不保证 action chunk 或闭环行为正确。
- smoke test 通过后才决定正式训练预算。

## DP6：正式 Push-T 训练

### 实验冻结

训练前保存一份不可悄悄修改的 experiment manifest：

- 代码版本与依赖版本。
- dataset revision 与 split。
- observation/action keys。
- $T_o$、$T_p$、$T_a$。
- 图像尺寸、normalization 和 augmentation。
- diffusion scheduler、训练/推理 diffusion steps。
- backbone、参数量、optimizer、learning-rate schedule。
- batch size、梯度累积、训练步数和 checkpoint 频率。
- 训练 seed 与评测 seed。
- success 判据与正式评测 episode 数。

### 训练顺序

1. seed 0 完成端到端训练并评测，排除系统性错误。
2. 配置冻结后再运行 seed 1、2。
3. 三个 seed 使用相同数据、预算、评测初始状态分布和成功判据。
4. checkpoint 选择规则预先确定，不能逐 seed 手工挑最好视频。

### 训练记录

- training loss 与 validation loss（若配置了无泄漏 validation split）。
- 学习率与梯度异常。
- 每秒训练 step、显存峰值和总训练成本。
- checkpoint 对应的闭环指标。

### 通过标准

- 三个 seed 均完成训练、checkpoint 加载和闭环评测。
- 训练曲线、配置和命令能够对应到具体 checkpoint。
- 不根据单次 rollout 判断模型好坏。
- 能解释 seed 间差异，并区分训练随机性与评测随机性。

## DP7：统一闭环评测与失败分析

### 正式协议

按照 VLA 路线图，每个训练 seed 至少评测 50 个 episode。评测开始前固定 episode seeds 或初始状态生成规则。

每个 seed 报告：

- 成功次数、总次数和成功率。
- Wilson 95% 置信区间。
- 平均最大目标覆盖度及分布。
- 每次 replanning 推理延迟的均值、中位数和高分位数。
- action 平滑性指标。
- episode 长度或完成时间。
- 失败类型与代表视频。

同时汇总三个训练 seed 的均值和离散程度，但不能只给合并后的单一数字。

### 失败归因

失败类型限定为：

- 视觉表征或遮挡。
- observation/action 时间对齐。
- action chunk 生成质量。
- horizon 与 replanning。
- normalization/denormalization。
- diffusion sampling 或推理延迟。
- 数据覆盖/covariate shift。
- 环境或评测接口。

每个失败结论必须附带轨迹、视频、日志或受控实验证据。

### 一个受控实验

只从以下方向选择一个，不同时展开：

- 改变 $T_a$，观察响应性、延迟和成功率。
- 减少推理 diffusion steps，观察速度—质量权衡。
- 移除视觉输入或只用 state，检查 condition 的作用。

除被研究变量外，数据、checkpoint 或训练配置、评测 seeds 和成功判据保持不变。若修改的是推理参数，优先在同一 checkpoint 上比较。

### 通过标准

- 量化结果、失败视频和配置齐全。
- 能区分 open-loop action/noise loss 与 closed-loop task success。
- 能指出至少一个由证据支持的主要瓶颈。
- 不把 Push-T 仿真成功写成真机能力。

## DP8：形成解释闭环并返回 VLA 路线

### 必须回答

1. Diffusion Policy 与普通 BC 的数据来源是否不同？目标函数哪里不同？
2. 为什么扩散能表达多模态 action distribution，而单步 MSE 容易平均？
3. 为什么预测 action chunk 能建模时间一致性？
4. $T_o$、$T_p$、$T_a$ 分别影响什么？
5. observation condition 通过什么模块影响每一步去噪？
6. 为什么训练可以并行采样随机 diffusion timestep，推理却需要迭代？
7. receding horizon 解决什么，又没有解决什么？
8. Diffusion Policy 为什么仍受 covariate shift 影响？
9. 推理 diffusion steps 减少为什么能降低延迟，又可能损害动作质量？
10. 从 DDPM 切换到 Flow Matching 时，训练目标与推理动力学将发生什么变化？

### 最终产物

- 论文阅读与公式—代码映射笔记。
- 数据时间窗与关键张量形状记录。
- fixed-batch overfit 和 smoke-test 证据。
- 三个 seed 的训练曲线、配置和 checkpoint 索引。
- 每个 seed 至少 50 个 Push-T episode 的评测结果。
- 成功/失败视频与失败归因表。
- 一个 horizon 或采样步数受控实验。
- 一份结论与边界说明。

### 最终通过标准

- 能从公式、代码、数据和闭环控制四个角度解释 Diffusion Policy。
- 能在新进程中复现至少一个训练 checkpoint 的评测。
- 能用量化证据说明策略表现和主要失败模式。
- 能明确区分论文方法、LeRobot 实现和自己的实验结论。
- 不继续扩展 Diffusion Policy；返回路线图，进入二维 Conditional Flow Matching。

## 建议产物结构

不要在开始前一次性搭建全部文件。每通过一个 gate，只增加下一关需要的最小产物。

```text
ImitationPolicyLearning/
├── README.md                           # 仓库定位、当前状态与复现边界
├── plans/
│   └── DIFFUSION_POLICY_LEARNING_PLAN.md
├── diffusion_policy/                   # DP 检查、训练与评测代码
├── act/                                # 后续共享协议下的 ACT 对照
├── common/
│   ├── data/                           # 数据检查与时间对齐工具
│   └── evaluation/                     # 统一闭环指标与评测协议
├── configs/                            # 冻结后的实验配置
└── artifacts/                          # 小型曲线、表格与失败分析
```

大型 dataset、checkpoint、环境缓存和原始视频不直接提交 Git。仓库中保存获取方式、revision、校验信息、配置、指标和必要的小型图表。

## 当前进度

- [x] DP0：论文第一遍与全局映射
- [ ] DP1：预训练策略闭环评测
- [ ] DP2：Push-T 数据与时间窗
- [ ] DP3：条件动作扩散的公式—代码映射
- [ ] DP4：策略结构与条件注入
- [ ] DP5：fixed-batch overfit 与 smoke test
- [ ] DP6：三个 seed 正式训练
- [ ] DP7：统一闭环评测与受控实验
- [ ] DP8：解释闭环与阶段总结

## 暂停条件

遇到以下情况时暂停训练，先定位问题：

- observation/action 时间索引无法逐项解释。
- episode 边界存在跨轨迹采样或不明 padding。
- normalization 的来源或执行位置不清楚。
- 预训练 checkpoint 无法正常闭环运行。
- fixed-batch 无法过拟合。
- loss、梯度或 action 出现 `NaN/Inf`。
- checkpoint 无法在新进程中复现采样。
- 评测成功判据或 seeds 在实验后才决定。

这些情况是数据或实验闭环尚未验证，不应通过增加训练步数掩盖。
