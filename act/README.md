# ACT 学习工作入口

本目录用于 Action Chunking with Transformers 的机制学习、真实数据检查与有界训练/闭环验证。截至 2026-10-31，A1 VAE/CVAE 最小基础已讲解，A2 数据与接口检查已按直接实现授权完成，见 [学习进度](../artifacts/act/a1_a2_learning_progress.md)与 [A2 验证记录](../artifacts/act/a2_verification.md)。A0–A2 的完整独立理解验收待补，权威计划的 current_gate 仍为 **A0**；运行通过不自动勾选学习 gate。下一学习模块 A3 已完成进入前定位，正式展开待许可；尚无策略 forward/backward、参数更新或 ACT 闭环结果。前序记录见 [A0 学习进度](../artifacts/act/a0_learning_progress.md)。

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

环境使用已有 `imitation-policy`。A2 已有真实运行记录；后续网络梯度、参数更新及闭环检查到相应 gate 再执行。

## A2 数据与接口检查

- 实现：[inspect_act_dataset.py](inspect_act_dataset.py)。
- 冻结配置：[pusht_a2.json](../configs/act/pusht_a2.json)。当前为单观测、16 步动作块、8 步执行长度、batch size 2、mean/std 归一化、随机初始化 ResNet18 的配置；本检查不创建网络。
- 证据：[说明](../artifacts/act/a2_verification.md)、[完整 JSON](../artifacts/act/a2_verification.json)。

从仓库根目录运行：

```bash
/home/wordwoods/miniconda3/envs/imitation-policy/bin/python -m unittest discover -s act -p 'test_*.py' -v
/home/wordwoods/miniconda3/envs/imitation-policy/bin/python act/inspect_act_dataset.py --device cuda
```

每次检查在 `outputs/act/a2_<UTC时间>/report.json` 新建报告；失败也保留异常信息，不覆盖旧运行。需要数据已存在于本地缓存，并保留配置引用的 DP5 下载清单、划分和训练统计文件。脚本强制离线、逐文件验证固定 revision 的来源，不下载数据或权重；仅克隆仓库不具备这些本地产物。

不具备 GPU 访问时，可显式使用 `--device cpu` 检查数据部分，结果为 `DATA_PASS_GPU_NOT_RUN`，不能当作 GPU 检查通过。CUDA 仅检查 4×4 矩阵与真实 batch 的归一化，不证明 ACT 网络的训练显存、速度或任务能力。

本轮覆盖 train 的 episode 0/1、val 的 episode 5/22：每个 episode 检查首、中、最后完整窗、首个补位窗和末帧；每组两个真实 DataLoader batch。没有逐窗解码全部数据集。未运行策略 forward/backward、参数更新、checkpoint 或 rollout。

## 参考与前序证据

- [ACT 原论文](https://arxiv.org/abs/2304.13705)
- [原作者实现](https://github.com/tonyzhaozh/act)
- [DP 收尾记录](../artifacts/dp8_closeout.md)
- [二维 FM 收尾记录](../../DeepLearningFoundations/FlowMatching/artifacts/f6_closeout.md)（本机兄弟仓库链接）
