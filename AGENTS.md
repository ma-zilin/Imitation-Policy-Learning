# 项目协作规则

## 项目边界

- 本仓库用于 Diffusion Policy 与后续 ACT 的策略学习、实验和统一闭环评测。
- 基础 DDPM、二维生成实验等留在 `DeepLearningFoundations`；真机、SO-101、ROS2、Sim-to-Real、大型 VLA 训练暂不属于本仓库。
- 第三方代码、数据集、模型权重沿用各自许可证；MIT 仅覆盖本仓库原创代码与文档。

## 当前状态与事实来源

- DP0–DP3 已完成；当前门槛是 DP4：策略结构与条件注入。
- DP2 已验证真实 episode 的解码、首中末时间窗、padding、batch 形状与归一化往返；证据见 `artifacts/dp2_verification.md`。尚未完成本地训练或正式训练集划分。
- DP3 已用 `diffusion_policy/check_training_step.py` 验证真实 batch 的单次 forward/backward、MSE 对照与视觉编码器/去噪网络梯度；使用预训练权重，未执行参数更新。DP4 已讨论观测整理、视觉编码器、SpatialSoftmax、时间编码与联合条件、FiLM，以及时序 U-Net 的完整下行/中间/上行/输出数据流；知识说明已整理到外部 `Diffusion Policy.md`。尚需推理调用链收尾、global/token conditioning 与 CNN/Transformer 结构对照及整体复述验收，不能写成 DP4 已完成。本轮仅更新学习记录，没有新增运行验证或参数更新。
- 开始工作前依次检查学习计划、Git 状态、最新提交和已有 artifacts。文档与证据冲突时先指出并核实，不把未来计划当成已完成状态。
- LeRobot API、依赖、GPU、checkpoint revision 和私有 action queue 都可能漂移；运行实验前重新核验，不能只沿用旧记录。

## 学习与实现方式

- 在推进下一步学习、实验、代码修改或验证前，先向用户说明接下来要做什么、为什么要做，以及该步骤要回答的问题或预期产出；说明后再执行，已获授权的工作无需重复请求确认。
- 默认“先教后做”：先解释模块职责、输入输出、数据流、张量形状或物理单位、通道顺序、时间索引和闭环因果关系。
- 优先讲直接影响实现的最小理论；区分论文明确内容、后续资料和直觉补充。
- DP2–DP4 以 LeRobot 原实现为学习对象；本仓库脚本用于检查、可视化和最小验证，不自行重写策略网络。实际 episode 划分与训练集统计量计算留到 DP5 训练准备阶段。
- 未经用户明确要求，不代写核心学习实现。用户明确要求实现，或调用 `/build` 后，直接完成约定范围并验证，不继续把实现工作退回给用户。
- 默认使用简体中文和适合终端阅读的普通文本公式。

## 工程与证据要求

- 修改前检查并保留已有修改和未跟踪文件；不得覆盖 `diffusion_policy/inspect_pusht_dataset.py` 等用户工作。
- 先做最小 smoke test，再做正式多 seed 实验；昂贵训练必须建立在数据、接口和固定 batch 验证之后。
- 不以 training loss 或单次 reward 代替闭环证据。评测至少报告 success 边界、coverage、terminated/truncated、episode 长度、重规划延迟、动作行为以及成功和失败视频。
- 近阈值结果和动作平滑性必须结合轨迹或视频解释；平滑不等于方向正确，reward 接近 1 也不自动等于成功。
- 所有完成声明都要对应可复现命令、版本/配置和实际产物；没有运行的检查必须明确标为未验证。
