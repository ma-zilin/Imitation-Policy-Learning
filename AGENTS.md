# 项目协作规则

## 项目边界

- 本仓库用于 Diffusion Policy 与 ACT 的策略学习、实验和统一闭环评测；当前主线为 ACT，讨论已推进到 A3 进入前定位，最早未完成独立验收的 gate 为 A0。
- 基础 DDPM、二维生成实验等留在 `DeepLearningFoundations`；真机、SO-101、ROS2、Sim-to-Real、大型 VLA 训练暂不属于本仓库。
- 第三方代码、数据集、模型权重沿用各自许可证；MIT 仅覆盖本仓库原创代码与文档。

## 当前状态与事实来源

- DP0–DP8 已按“机制学习与有界验证”主线收尾。DP5 保留 fixed-batch、200 步真实 DataLoader 训练、完整 checkpoint、新进程恢复和闭环接口证据；DP6 已形成协议并完成关键引导式理解检查，DP8 已完成系统回顾、边界区分及知识笔记整理。收尾记录见 `artifacts/dp8_closeout.md`；不声称所有知识点已独立书面复述或模型已具备可靠任务能力。
- DP7 已完成固定预训练模型的 100/50 推理步数对照：环境与采样 seeds 0～4，共 10 个 episode、2218 个环境步；成功次数 3/5 与 5/5，稳态平均规划耗时约 1318/656 ms。证据见 `artifacts/dp7_inference_steps.md`；只支持这组小样本观察，不证明 50 步普遍最优，也不代表网络权重或噪声预测能力提升。
- 二维 Conditional Flow Matching F0–F6 已于 2026-10-26 在 `DeepLearningFoundations` 按机制学习、关键对话检查与二维有界验证收尾；事实见 `../DeepLearningFoundations/FlowMatching/LEARNING_PLAN.md` 与 `../DeepLearningFoundations/FlowMatching/artifacts/f6_closeout.md`。二维完成不代表机器人策略或真机验证。
- 当前权威计划为 `plans/ACT_LEARNING_PLAN.md`，工作入口为 `act/README.md`，current_gate 仍为 A0（最早未完成独立验收的 gate）。截至 2026-10-31，A0 整体地图与 A1 VAE/CVAE 最小基础已讲解并有部分引导式检查；A2 已按直接实现授权完成代码、冻结配置与有界运行验证：9 项测试、416 个数据文件校验、20 个窗口、4 个真实 batch、归一化往返及最小 CUDA 检查通过。A0–A2 的完整独立解释与适用新例子验收仍待补，不自动勾选通过。A3 仅完成进入前定位，正式展开待许可；尚未创建策略网络、运行策略 forward/backward、更新参数或执行 rollout。学习进度见 `artifacts/act/a1_a2_learning_progress.md`，运行证据见 `artifacts/act/a2_verification.md`；A0 历史记录保留于 `artifacts/act/a0_learning_progress.md`。
- ACT 首轮范围为单训练 seed 0、固定 batch 最多 500 次更新、真实 DataLoader 200 次更新、本地 checkpoint 的 1 个最多 300 步闭环诊断（环境 seed 0）；配置在运行前冻结。未通过诊断时保留证据并查因，不自动延长预算。文档初始化不等于实验已获执行授权。
- Diffusion Policy 长程从头训练、三个训练 seed 及每 seed 至少 50 个 episode 的性能复现继续暂缓，未完成；ACT 学习不自动重启这些扩展。
- 本机已有训练更新与 batch 资源短测证据，不能写成“消费级显卡不适用于训练”。调整路线基于成本、学习问题和有限预算的归因边界；10,000 步等预算示例、完整验证与其成本短测均未获执行授权，不能自动启动。
- DP2 的真实 episode 解码、首中末时间窗、padding、batch 与归一化往返证据见 `artifacts/dp2_verification.md`。DP5 已完成 185/21 个 episode 的训练/验证划分，并从全部 23037 个原始训练帧计算统计量。最新实验摘要见 `artifacts/dp5_verification.md`；数据准备只做了有界窗口解码检查，不是所有时间窗的完整审计。
- DP5-B 的固定 4 窗口验证 MSE 从 1.328379 降到 0.154085，但 seed 0 的 300 步闭环未成功，最大 coverage 0.184486。脚本 PASS 表示链路可运行，不表示任务成功；不把该结果写成完整论文复现或正式成功率。
- DP3 已用 `diffusion_policy/check_training_step.py` 验证真实 batch 的单次 forward/backward、MSE 对照与视觉编码器/去噪网络梯度；使用预训练权重，未执行参数更新。DP4 已完成结构与数据流讲解、推理调用链、global/token conditioning 和 CNN/Transformer 对照；用户经纠正后准确复述了专家动作/真实噪声的训练角色，并确认动作缓存非空时不调用 UNet。DP4 是阅读与理解阶段完成，不是新的运行验证；本轮未更新参数、未训练 Transformer 或执行 horizon 对照。结构知识见外部 `Diffusion Policy.md`，本轮收尾与验收记录见学习计划。
- 开始工作前依次检查学习计划、Git 状态、最新提交和已有 artifacts。文档与证据冲突时先指出并核实，不把未来计划当成已完成状态。
- LeRobot API、依赖、GPU、checkpoint revision 和私有 action queue 都可能漂移；运行实验前重新核验，不能只沿用旧记录。

## 学习与实现方式

- 在推进下一步学习、实验、代码修改或验证前，先向用户说明接下来要做什么、为什么要做，以及该步骤要回答的问题或预期产出；说明后再执行，已获授权的工作无需重复请求确认。
- 默认“先教后做”：先解释模块职责、输入输出、数据流、张量形状或物理单位、通道顺序、时间索引和闭环因果关系。
- 优先讲直接影响实现的最小理论；区分论文明确内容、后续资料和直觉补充。
- ACT 以本地 LeRobot 实现为运行对象、原论文与作者实现为方法核对来源。已有环境与数据优先复用；默认配置面向 ALOHA，Push-T 的窗口、输入输出、归一化和执行长度须独立检查。公开 ACT 权重不作必需前置，不默认升级或修改上游 LeRobot。
- 知识正确性、逻辑与可核验证据优先于用户说辞、助手先前回答和已有笔记。进入新模块先定位价值、结构、关键联系与边界，获许可后先地图后细节，结束时验证理解并向上归位。
- ACT 核心小节、误解纠正、实验设计和 gate 验收按已授权的 grill-me 教学适配进行；使用前读取 `/home/wordwoods/.codex/skills/grill-me/SKILL.md`。概念检查每轮一个核心开放问题，最多两个独立问题，先等回答；连续两轮未解决转为具体例子，达到独立解释及适用新例子的标准后结束。用户要求暂停追问或直接解释时遵从；不因追问阻碍明确授权的实现。
- ACT 知识笔记约定为 AcademicCodex 的 `20 Notes/02 TechStack/02 AI & DeepLearning/ACT.md`，仅在用户明确要求整理时创建或更新；初始化不创建知识笔记，不把运行通过等同于机制掌握。
- DP2–DP4 以 LeRobot 原实现为学习对象；本仓库脚本用于检查、可视化和最小验证，不自行重写策略网络。实际 episode 划分与训练集统计量计算留到 DP5 训练准备阶段。
- 未经用户明确要求，不代写核心学习实现。用户明确要求实现，或调用 `/build` 后，直接完成约定范围并验证，不继续把实现工作退回给用户。
- 默认使用简体中文和适合终端阅读的普通文本公式。

## 工程与证据要求

- 修改前检查并保留已有修改和未跟踪文件；不得覆盖 `diffusion_policy/inspect_pusht_dataset.py` 等用户工作。
- 当前 ACT 以机制理解、真实数据检查、有界训练/恢复和有限闭环诊断为主；既有 DP 预训练推理对照保留为历史证据。未经用户明确重启可选性能复现，不开展长程训练或正式多 seed 实验。重启时仍须先做最小 smoke test，并核验数据、接口、固定 batch、资源预算和实验冻结规则。
- ACT/DP 对照必须注明 checkpoint 来源、条件窗口、配置与训练预算差异；单个 ACT 诊断 episode 和历史 DP 结果不组成正式成功率或同预算算法排名。ACT 的潜变量编码路径 loss、部署输出与闭环结果分别评估。
- DP7 只改变执行 horizon 或推理 diffusion steps 中的一个，不同时扩大变量和评测规模；先固定 checkpoint revision、配置、环境/采样 seeds 与成功判据。小样本结果不等于普遍优势，历史 DP1 结果不可未经协议核对直接作为配对基线。
- 保留未知根因，不要求强行从失败视频或低 noise MSE 中定位唯一瓶颈；候选方案、实际运行证据与性能未验证的边界分别记录。
- 不以 training loss 或单次 reward 代替闭环证据。评测至少报告 success 边界、coverage、terminated/truncated、episode 长度、重规划延迟、动作行为以及成功和失败视频。
- 近阈值结果和动作平滑性必须结合轨迹或视频解释；平滑不等于方向正确，reward 接近 1 也不自动等于成功。
- 所有完成声明都要对应可复现命令、版本/配置和实际产物；没有运行的检查必须明确标为未验证。
