# ACT 学习工作入口

本目录用于 Action Chunking with Transformers 的机制学习、真实数据检查与有界训练/闭环验证。当前 gate 仍为 **A0：问题与整体地图**：整体地图已讲解并进行引导式检查，完整独立复述待验证，暂未勾选通过。A1 已做进入前定位，正式展开待许可；尚无 ACT 实验结果。详见 [A0 学习进度](../artifacts/act/a0_learning_progress.md)（2026-10-29）。

唯一权威计划：[ACT_LEARNING_PLAN.md](../plans/ACT_LEARNING_PLAN.md)。从仓库根目录 `/home/wordwoods/Projects/ImitationPolicyLearning` 工作，遵循根目录 [AGENTS.md](../AGENTS.md)。

## 学习路径

```text
A0 整体地图
→ A1 VAE/CVAE 最小基础
→ A2 数据与配置
→ A3 网络与训练目标
→ A4 动作生成与执行
→ A5 有界训练与恢复
→ A6 闭环诊断与方法对照
→ A7 收尾
```

先从 ACT 自身的问题和整体结构开始；模块学习先定位再展开，使用理解检查和运行证据分别验收。实现以已有 LeRobot ACT 为对象，本目录的检查与实验脚本到对应 gate 再创建。

## 首轮运行边界

- 单训练 seed 0。
- 固定 batch 诊断最多 500 次参数更新。
- 真实 DataLoader 训练 200 次参数更新。
- 本地 checkpoint、环境 seed 0，先做 1 个最多 300 步的闭环诊断。
- 公开 ACT 权重不作为必需前置；配置与兼容性在运行前核验，不默认升级现有环境。

高成功率不作为本阶段结业条件。预算内的诊断失败要保留并定位，不能用目录或脚本的建立代替通过证据。

## 文件去向

| 内容 | 位置 |
| --- | --- |
| 学习检查与实验代码 | `act/`，按 gate 创建 |
| 冻结的实验配置 | `configs/act/` |
| 小型指标、图表、验证与收尾记录 | `artifacts/act/` |
| 大型权重、原始视频和本地运行输出 | 被 Git 忽略的 `outputs/`、`checkpoints/` 等路径 |
| 阶段知识笔记 | AcademicCodex 的 `20 Notes/02 TechStack/02 AI & DeepLearning/ACT.md`，仅明确要求整理时写入 |

环境优先使用已有 `imitation-policy`；具体依赖与可复现命令在实际检查通过后补充。当前已记录 A0 学习进度，尚未新增 ACT 学习实现或执行实验，未提供未经运行验证的安装或训练命令。

## 参考与前序证据

- [ACT 原论文](https://arxiv.org/abs/2304.13705)
- [原作者实现](https://github.com/tonyzhaozh/act)
- [DP 收尾记录](../artifacts/dp8_closeout.md)
- [二维 FM 收尾记录](../../DeepLearningFoundations/FlowMatching/artifacts/f6_closeout.md)（本机兄弟仓库链接）
