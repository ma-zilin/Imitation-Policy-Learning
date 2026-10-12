---
type: learning-plan
topic: diffusion-policy
status: active
current_gate: DP6
started: 2026-09-01
progress_mode: gate-based
---

# Diffusion Policy 学习计划

## 目标与当前位置

本阶段的目标不是“调用一个现成策略得到视频”，也不是从零复刻整个官方仓库，而是建立一条能够被代码、实验和闭环评测共同验证的主线：

> 专家 episode → observation/action 时间窗 → 条件动作扩散训练 → action chunk 反向采样 → receding-horizon 执行 → Push-T 闭环评测

当前主线定位为机制学习与有界验证：保留已完成的 fixed-batch、短程训练和恢复检查；通过训练流程阅读与预训练策略受控评测完成后续学习。长程从头训练、三个训练 seed 的性能复现及大规模统计评测暂缓，作为可选扩展，不是本阶段结业前提。

消费级显卡并非不能训练当前模型，已有本机参数更新和资源短测证据。调整路线是为了使计算成本与学习问题匹配：有限预算下的任务失败不足以区分训练量不足与其他原因，不能据此断言硬件不适用、模型无效或已定位失败根因。

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
- 本阶段以有界检查为主，不自动推进长程训练；未来重启性能复现时，先做单 seed 检查，再冻结配置开展多 seed 实验。
- 不用 training loss 代替闭环表现，不只保留最好看的 rollout。
- 每一关只解决一个主要问题；前一关没有形成证据，不进入下一关。
- 区分已理解、已设计、已运行和性能未验证；预算示例、候选配置及预训练结果不能写成本地正式训练成绩。

## 学习与实现边界

### 本阶段必须完成

- 解释 Diffusion Policy 为什么仍然属于 Behavior Cloning。
- 构造对齐的 observation window 与 action chunk。
- 将二维 DDPM 的变量映射到条件动作序列扩散。
- 读懂视觉编码器、条件注入、时间编码和 1D 去噪网络的数据流。
- 跑通预训练 Push-T 策略的闭环评测。
- 完成 fixed-batch overfit 和短程训练 smoke test。
- 理解训练预算、学习率调度、验证、checkpoint 选择和完整恢复，形成一份标明未执行部分的实验协议。
- 在固定预训练 checkpoint 上完成一个有界的推理受控实验，不要求重新训练模型。
- 记录成功率、最大目标覆盖度、推理延迟、动作平滑性和失败轨迹。

### 本阶段暂不展开

- Flow Matching、ACT、SmolVLA 或其他策略的正式训练。
- 真机、SO-101、ROS2、MoveIt2 或 Sim-to-Real。
- 从零手写 ResNet、1D U-Net、Transformer block 或 diffusion scheduler。
- 修改 Push-T 任务、采集新机器人数据或追求完整论文表格。
- DAgger、在线纠错、强化学习后训练。
- 在 CNN 与 Transformer 两种 Diffusion Policy backbone 上同时开展完整调参。
- 长程从头训练、三个训练 seed 的性能复现及每个训练 seed 至少 50 个 episode 的正式评测；这些保留为可选扩展。

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
6. 检查 normalization statistics 的来源和应用范围，验证 normalize/unnormalize 往返；本阶段不重建训练集统计量。实际按 episode 划分、仅从训练集计算统计量留到 DP5。

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

### 先阅读真实训练调用链

沿 LeRobot 的 `DiffusionPolicy.forward()` → `DiffusionModel.compute_loss()` 阅读，先回答：观测如何成为 condition、动作何时归一化和加噪、每个样本的 diffusion timestep 如何生成、预测目标与 loss 如何对应。

随后做下面的最小机制实验。沿用 LeRobot 的策略、scheduler 与 loss；本仓库只添加检查入口，不从零重写网络，不启动完整训练，也不在本阶段另建训练集划分。

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

训练准备：固定并记录按 episode 划分的训练/验证集合；只用训练 episode 计算归一化统计量，并将同一组统计量用于验证。不得把 DP2 读取的仓库级 `meta.stats` 自动视为新划分的训练集统计量。

1. 使用正式 dataloader 进行短程训练。
2. 检查吞吐、显存、checkpoint、恢复训练和日志。
3. 用固定小型验证 batch 记录 validation loss。
4. 对中间 checkpoint 运行少量 rollout，确认采样和控制链未损坏。

### 诊断基线

- 未训练同结构 policy 的 loss/rollout。
- 恒零 noise predictor 的 noise MSE。
- 公开预训练 checkpoint 的闭环结果，用于接口正向检查与已有策略表现参考，不作为同预算训练对照，也不视为本地训练的性能上限。

### 通过标准

- fixed batch loss 可以显著下降。
- 正式数据训练无 `NaN/Inf`，显存稳定。
- checkpoint 在新进程中可以恢复训练和 rollout。
- 能解释低 noise MSE 为什么仍不保证 action chunk 或闭环行为正确。
- smoke test 通过后可以学习训练预算与验证设计，但不以长程训练效果作为本阶段通过条件；未来重启正式训练时仍须先冻结预算。

## DP6：训练流程与实验设计

### 目标与边界

沿当前 LeRobot 训练调用链理解优化与验证，回答“如果开展正式训练，怎样让过程可解释、可恢复、可比较”。本关以源码阅读、已有 DP5 证据和实验协议为产物，不强制新增长程训练或完整验证运行。

### 学习顺序

1. 从 episode、时间窗、batch 到 step/epoch，说明共享参数、样本打乱与训练噪声重采样的作用；打乱不改变时间窗内部顺序。
2. 区分 optimizer 与扩散 scheduler、学习率 scheduler；理解 warmup/cosine、总预算对曲线的影响及 optimizer/scheduler 更新顺序。
3. 追踪验证调用链：训练集统计量、独立验证 episode、确定性图像处理、固定多组加噪条件、`eval()` 与 `no_grad()`，以及按样本数加权汇总 MSE。
4. 区分小型验证子集与完整验证集；完整验证按 batch 流式处理，不把全部图像预载 GPU；未实测的成本只记录为估算。
5. 提前约定 checkpoint 选择规则；区分用于推理的权重与包含 optimizer、scheduler、RNG、数据游标等的完整恢复状态。
6. 区分离线 noise MSE、闭环任务表现和失败归因，不因短程训练失败而猜测根因。

### 实验协议设计

形成一份可追溯的 experiment manifest 设计；实际执行前另行冻结，未决定的字段标注待定：

- 代码版本与依赖版本。
- dataset revision 与 split。
- observation/action keys。
- $T_o$、$T_p$、$T_a$。
- 图像尺寸、normalization 和 augmentation。
- diffusion scheduler、训练/推理 diffusion steps。
- backbone、参数量、optimizer、learning-rate schedule。
- batch size、梯度累积、训练步数和 checkpoint 频率。
- 训练 seed 与评测 seed，区分训练随机性、环境初始状态与策略采样随机性。
- 验证窗口范围、加噪条件、频率和 checkpoint 选择规则。
- success 判据、评测 episode 数与资源预算。

### 已讨论但未执行的候选规则

- batch size 8 是资源短测中的吞吐候选，不代表泛化最优。
- LeRobot 配置的 Adam、目标学习率 `1e-4`、500 步 warmup 与 cosine 已讨论；DP5 实际使用恒定学习率，没有运行该调度方案，也没有启用 EMA/AMP。
- 完整验证候选为 2466 个窗口、每窗 4 组固定加噪条件，在训练前、每个 epoch 后及训练结束时检查；尚未运行完整验证或测量其耗时。
- 主 checkpoint 候选按完整验证 MSE 严格改善更新 `best`，相同则保留较早版本；另保留 `last`。比较未四舍五入数值，不能依据测试 rollout 事后挑权重。
- 总训练步数尚未冻结；10,000 步等只用于预算换算，不是已获授权的训练任务或足够学习的保证。

### 通过标准

- 能解释 batch/step/epoch、顺序依赖和随机加噪的不同作用。
- 能追踪学习率调度、验证、模型选择和完整恢复的职责与数据流。
- 能说明固定验证条件的收益与局限，理解验证 MSE 不保证任务成功。
- 形成标明待定参数、成本估算和未运行部分的实验协议；不要求训练到良好任务表现。
- 能准确区分官方配置、当前 LeRobot 实现、候选设计和本地已运行证据。

## DP7：预训练策略受控评测

### 目标与协议

固定已能运行的公开预训练 checkpoint，只研究一个推理变量，学习如何用控制变量连接机制与闭环表现，不重新训练策略。

实验前固定模型 revision、环境与依赖、预处理、success 判据、评测 seeds、episode 数和停止条件。先检查改变参数的接口与时间对齐，再在计算预算内选定一组小规模配对评测 seeds；基线与变体使用同一组 seeds，并记录策略采样随机性。

不再强制每个训练 seed 评测 50 个 episode。小样本只支持有限范围的观察，不能宣称统计显著或普遍优越；相同环境 seed 不保证不同推理配置具有相同的采样噪声轨迹。

历史 DP1 的 10-seed 结果可作背景；只有 checkpoint、环境、配置和 seed 协议一致且证据可核对时，才能作为新实验的配对基线，否则重新运行对应基线。

### 一个受控实验

只从以下方向选择一个，不同时展开：

- 改变 $T_a$，观察反馈响应、重规划频率、推理成本与任务表现。
- 减少推理 diffusion steps，观察采样速度与任务表现；先确认当前 scheduler 支持该设置。

除被研究变量外，checkpoint、数据预处理、环境、评测 seeds、最大步数和成功判据保持不变。移除视觉输入或改用 state-only 涉及条件分布/模型适配，不作为本关无需重训的默认对照。

### 记录与解释

- 每个配置的成功次数、episode 总数、成功率与最大 coverage 分布。
- `terminated/truncated`、episode 长度、重规划次数与延迟、action 平滑性。
- 所有 episode 的指标，以及有代表性的成功/失败视频；某配置没有成功时如实记录，不要求制造成功示例。
- 可计算 Wilson 95% 置信区间以表达小样本的不确定性；区分环境 episode 差异与训练 seed 差异，本关没有重新训练多个模型。
- 先描述可观察行为，再给证据支持的解释。失败原因可保持未知，不要求强行找到主要瓶颈；方向错误、对齐失败等观察不能自动归因于视觉、数据覆盖或网络能力。

### 通过标准

- 在预先固定的协议下完成一个推理变量的配对对照；量化结果、配置和实际 checkpoint 能对应。
- 能解释变量的作用、观察到的变化及计算成本；结果相近或不支持预期也属于有效记录。
- 能区分离线噪声指标、闭环表现、行为观察与未经证实的根因。
- 明确小样本、预训练来源、采样随机性和仿真环境的结论边界，不把结果写成本地训练复现或真机能力。

## 可选扩展：正式训练与性能复现（暂缓）

保留原多 seed 实验目标，暂不执行，也不计入当前阶段必做项或已完成项。重启须由用户明确授权，并先确定参考配置、足够的计算/磁盘预算和需要回答的研究问题；不得自动启动 10,000 步预算探索或长程训练。

1. 冻结代码、数据划分、normalization、网络、optimizer/scheduler、训练预算、验证范围与 checkpoint 选择规则；若采用 EMA/AMP，单独核验实现与资源。
2. seed 0 完成端到端训练和评测，检查系统链路；单次任务失败不等于系统错误已定位，须另有证据。
3. 配置冻结后再训练 seed 1、2，保持相同数据、预算、模型选择与评测协议。
4. 每个训练 seed 至少评测 50 个预先规定的环境 episode；报告成功率、Wilson 95% 区间、coverage、延迟、动作行为及视频，并分别汇总训练 seed 的差异。
5. 记录 training/validation loss、学习率、梯度异常、吞吐、显存、实际成本和 checkpoint 索引；任何失败归因均需受控证据。

只有实际完成对应训练与评测后，才能声明性能复现；不以硬件型号或候选预算推断已达到/无法达到论文表现。

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
- 训练流程与实验协议说明，区分候选设置、待定预算和已执行部分。
- 既有 DP5 短程训练的曲线、配置和 checkpoint 索引；不要求新增三个 seed 的正式训练产物。
- 固定预训练 checkpoint 的一个 horizon 或采样步数受控实验，包含评测协议、逐 episode 指标和代表视频。
- 已观察行为与可能解释的记录；证据不足的失败原因明确标注未知。
- 一份结论与边界说明。

### 最终通过标准

- 能从公式、代码、数据和闭环控制四个角度解释 Diffusion Policy。
- 能在新进程中复现至少一个训练 checkpoint 的评测。
- 能用量化证据说明预训练策略在有界对照中的表现与可观察失败行为，不要求定位唯一或主要根因。
- 能明确区分论文方法、LeRobot 实现和自己的实验结论。
- 明确本地训练管线已运行验证，但完整验证集泛化、从头训练到可靠任务表现与多 seed 性能复现尚未验证；暂缓不是已完成。
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
- [x] DP1：预训练策略闭环评测
- [x] DP2：Push-T 数据与时间窗（真实 episode 的边界、中间样本、batch 与归一化往返已验证；见 [验证记录](../artifacts/dp2_verification.md)）
- [x] DP3：条件动作扩散的公式—代码映射（已阅读 forward/compute_loss，并运行 [单步检查脚本](../diffusion_policy/check_training_step.py)：MSE 对照通过，视觉编码器与去噪网络获得有限、非零梯度；未更新参数）
- [x] DP4：策略结构与条件注入（已完成结构与推理链讲解、条件形式与架构对照、关键理解验收；阅读与理解阶段完成，不代表新增运行实验）
- [x] DP5：fixed-batch overfit 与 smoke test（运行检查与关键边界的引导式理解验收通过；不表示模型已具备可靠任务能力；见 [DP5 证据](../artifacts/dp5_verification.md)）
- [ ] DP6：训练流程与实验设计（已讨论预算、学习率调度、验证与 checkpoint 规则；尚未完成协议整理和独立理解验收，不强制长程训练）
- [ ] DP7：预训练策略受控评测（一个推理变量、小规模固定 seeds；尚未选择变量或运行新对照）
- [ ] DP8：解释闭环与阶段总结
- 可选扩展：正式多 seed 训练与性能复现暂缓；未运行，不标记完成。

### DP4 学习记录与收尾验收

本阶段完成的是源码讲解、知识整理与关键理解验收，不是新的运行实验。此前的结构知识记录在外部 Obsidian 笔记 `Diffusion Policy.md`，不作为本仓库已提交的实验产物；本轮新增的推理链与条件/架构对照在此记录，不声称已同步写入外部笔记。

- 已讨论观测整理与视觉编码：图像裁剪、ResNet、SpatialSoftmax 位置摘要，以及视觉特征和机器人状态构成 (B,132) 条件的过程。
- 已讨论扩散时间编码与联合条件：时间编码 (B,128) 与观测条件拼成 (B,260)，由各条件残差块独立映射为 FiLM 的 scale/bias；时间步不是环境帧号。
- 已逐层讨论时序 U-Net：两个条件残差块的关系、保存张量引用、下采样、粗尺度中间块、跳跃拼接、转置卷积上采样、512→2 通道输出，以及 (B,16,2) 噪声预测的完整数据流。已区分通道拼接与残差相加、恢复长度与恢复信息；当前实现保存三份跳跃特征，只取用后两份。
- 已讨论预测机制与边界：带噪动作、观测及时间步提供预测依据，多尺度结构提供建模能力，真实噪声 MSE 决定输出语义；计算过程明确，但学到的内部策略难以完整解释，不保证逐样本精确还原噪声。
- 已讲解完整推理链：select_action 更新最近两帧观测，动作队列空时调用 generate_actions；后者编码条件、调用 conditional_sample 迭代采样，再取序列位置 1～8；上层反归一化、缓存并逐次返回动作。观测每步更新不等于每步重新规划；UNet 输出噪声，scheduler 更新动作块。
- 已讨论预测 16 步、执行 8 步的时间对齐与权衡：较短执行 horizon 提高反馈响应，较长片段可维持动作方案的一致性；8 来自当前 checkpoint 配置，论文有 action horizon 对照，但未在本地证明其最优。每步重规划即使不计计算成本，也不保证必然更可靠。
- 已完成条件形式对照：当前 FiLM 用观测特征与时间编码生成 scale/bias，调制动作特征；Transformer cross-attention 以动作特征构造 Q、观测特征投影为 K/V，按位置读取条件。global 向量的展平不等于平均或丢失帧顺序，token 条件也不必是局部条件。
- 已完成 CNN/Transformer 结构对照：卷积从局部共享规则出发，通过堆叠和多尺度整合扩大覆盖；注意力按输入内容建立允许位置间的联系。论文 Transformer 的动作 causal mask 不意味着逐动作自回归生成，整个带噪块仍可并行预测；论文整体架构差异不能直接归因于 FiLM/cross-attention 单个机制。
- 关键理解验收：先纠正了 UNet.forward 与 DiffusionPolicy.forward 的同名混淆，明确前者返回噪声预测、后者返回 loss，select_action 返回一个动作；随后用户准确复述“专家动作是初始数据，实际加入的随机噪声是标准答案，MSE 比较预测噪声与真实噪声”，并确认缓存非空时不运行 UNet。其他结构对照已讲解，未声称用户完成了全部内容的独立书面复述。

2026-10-10 收尾后将 `current_gate` 移到 DP5。下一步准备 DP5-A：固定真实小 batch、明确初始化及噪声/时间步是否固定、建立参数更新和检查协议，再验证学习能力。本轮未启动优化训练、未保存新 checkpoint、未重跑闭环评测，也未训练 Transformer 或执行 horizon 对照；DP5-B 的 episode 划分与训练集统计量仍待准备。

### DP5 运行验收记录（2026-10-12）

- DP5-A：随机初始化模型，在固定专家 batch 上更新 200 次；训练每步重采样噪声、时间步与裁剪。同 batch 的固定评估 MSE 从 1.304865 降到 0.078235，视觉编码器和 UNet 都实际更新，checkpoint 重载检查通过。这不是泛化验证。
- 数据准备：固定 `lerobot/pusht` v2.0 提交 `a16737ffb6c8dbe935291b9fd2f36d14c9bf0820`，核对 416 个所需文件。按 episode 划为 185 个训练、21 个验证；统计量只来自全部 23037 个原始训练帧，两组共用。检查全部合格起点的结构，并分别解码 2 个 batch，不声称逐项验证所有时间窗。
- DP5-B：随机初始化，batch size 2、seed 0、Adam 恒定学习率 1e-4，200 步；不启用 EMA、AMP 或学习率调度器。固定 4 个验证窗口、每 batch 4 组评估随机样本，MSE 从 1.328379 降到 0.154085；全部训练 loss 有限，末步两个主要模块梯度检查通过，峰值 allocated 显存约 5 GiB。
- 保存与恢复：保存权重、归一化缓冲区、Adam、RNG、epoch 排列与数据游标。新进程 1→2 步恢复与连续 0→2 步的全部权重及训练状态完全一致；另一次新进程从 200→201 步成功继续更新。后者不是重复训练两条分支的逐位对照。
- 闭环检查：200 步 checkpoint 在 seed 0 运行 300 步，重规划 38 次，`terminated=False`、`truncated=True`、`success=False`，最大 coverage 0.184486。采样/控制/视频链路通过，但任务失败；不作失败根因推断或正式成功率结论。
- 证据与命令见 [DP5 验收摘要](../artifacts/dp5_verification.md)，使用说明见 [短程训练说明](../diffusion_policy/TRAINING_SMOKE.md)。大型产物仍留在本地 `outputs/`，不进入 Git。
- 本次运行收尾时 `current_gate` 为 DP5：关键边界已解释，用户尚未独立复述；没有启动正式多 seed 训练。后续路线按下述调整执行，不再默认推进 seed 0→1→2 的长程训练。

### 后续路线调整（2026-10-12）

- 用户确认将后续目标改为“理解并验证机制 → 预训练策略的一个受控实验 → 阶段总结”；路线调整本身只修改计划和状态说明，没有训练或评测。之后用户另行授权本次提交与 push。
- 本机可以执行当前模型的训练更新；暂缓性能复现的理由是成本与学习收益、有限预算结果的归因边界，不是断言消费级显卡不能训练。
- 已讨论训练数据重采样、batch/episode/epoch、warmup/cosine、验证可比性及 checkpoint 选择；没有据此声明全部独立理解验收完成。
- 路线调整时保留 DP0–DP5 的历史证据与 DP5 待确认项，没有自动改变 `current_gate`；随后完成下述 DP5 关键边界验收，当前阶段移至 DP6。完整协议产物和 DP6 独立理解验收仍未完成。
- batch 资源短测代码及说明已有本地产物，见 [资源短测说明](../diffusion_policy/BATCH_SIZE_BENCHMARK.md)；这是吞吐检查，不是正式训练、完整验证或算法性能对照。此次提交纳入代码、离线测试与说明，不提交原始 outputs 或权重，不重跑 GPU 短测。
- 完整验证、其成本短测、10,000 步预算探索和多 seed 正式训练均未启动；未来额外运行须另行说明目的并取得授权。

### DP5 理解收尾与 DP6 学习记录（2026-10-12）

- 用户先提出未见状态与泛化问题；经区分“同一 Push-T 任务”和“未见状态”后，明确回答闭环中下一次观测由模型刚执行的动作产生。关键边界的引导式验收通过：专家窗口上的 noise MSE 不覆盖模型自身动作造成的后续状态与完整闭环任务表现。不将泛化不足写成此次失败的已证实根因，也不声称完成所有内容的独立书面复述。
- DP5 按既有运行证据与上述理解验收收尾，`current_gate` 移至 DP6；未新增训练、完整验证或 rollout。
- DP6 已讲解训练控制层的整体执行顺序、forward/backward/optimizer 分工，以及候选学习率调度和 best 选择尚未在 smoke 脚本实现的边界。
- 已讲解 checkpoint 的推理/续训职责、Adam 状态、随机数状态与 seed 的区别、数据排列和游标，以及初始化之后恢复 RNG 的原因。
- 已讲解实验协议与实际记录的区别、`manifest` 与 `contract` 的自定义职责、指标与 checkpoint 的追溯关系；部分理解问题未获独立作答，DP6 不标记完成。
- 推理扩散步数是 DP7 的建议研究方向，不是已冻结的变量选择或已授权执行的实验。

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
