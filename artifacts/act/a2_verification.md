# ACT A2 数据与接口验证

2026-10-31，在现有 `imitation-policy` 环境中按用户“直接实现完整版本”的授权完成。这里记录运行证据，不声明学习者已独立实现或独立掌握，不自动通过 A0/A1/A2 的理解验收。

## 实现与冻结配置

- [实现](../../act/inspect_act_dataset.py)、[9 项离线测试](../../act/test_inspect_act_dataset.py)。
- [运行前冻结配置](../../configs/act/pusht_a2.json)，SHA-256：`1cfda46c118d53b18b47c448cfb262662cea54b73e432b50a4616d4e491a0daa`。
- 本次脚本 SHA-256：`db68814e63c1007edc7bcee52d97ef42a4ecefbaa8baf046d5c4c5221302a940`。
- [完整运行报告](a2_verification.json)。原始报告位于 `outputs/act/a2_20261031T095624_152302Z/report.json`，两个报告内容相同。输出目录被 Git 忽略，版本化 JSON 保留检查细节。

实际数据配置为 `n_obs_steps=1`、`chunk_size=16`、`n_action_steps=8`、batch size 2、seed 0、`num_workers=0`、图像/状态/动作均为 mean/std 归一化。ResNet18 预训练权重字段显式为 `None`，未来构造时随机初始化；A2 没有创建网络或加载权重。8 步执行长度与 optimizer 字段已冻结供后续核对，本轮没有实际执行动作或 optimizer。

## 数据来源与版本

- `lerobot/pusht` 固定 revision：`a16737ffb6c8dbe935291b9fd2f36d14c9bf0820`。
- 清单所需 416 个元数据、表格和视频文件逐一检查大小与 SHA-256，全部匹配；共 9,158,289 字节。没有以文件存在代替指纹验证。
- 185/21 episode 的训练/验证划分互斥且完整；复用全部 23037 个原始训练帧的统计量，训练文件集合 370 个文件与固定清单一致。本轮未重新计算统计量。
- 两个 split 使用同一份训练统计量，未把仓库 `meta/stats.json` 作为归一化输入；统计量源 episode、fps、帧数、元数据与训练文件指纹均复核。
- LeRobot：`f495db0e1199325df5c62c1a7667378c90e4dc7e`，保留 dataset/utils/video 三处兼容修改；统计、dataset、视频解码代码与原统计来源的指纹一致。报告另存当前配置、归一化及修改文件指纹。
- Python 3.10.21；PyTorch 2.13.0、torchvision 0.28.0、LeRobot 0.1.0、datasets 5.0.1、av 17.1.0、gym-pusht 0.1.5。

## 检查结果

| 检查 | 实际范围与结果 |
| --- | --- |
| 真实窗口 | train episode 0/1、val episode 5/22，各 5 个窗口，共 20 个；全部通过 |
| 动作索引 | 独立按局部帧号计算期望，从原始表格逐帧比对完整动作块；无跨 episode 拼接 |
| 当前观测 | 状态与锚点原始记录一致；独立顺序 PyAV 解码四个选定视频，共 545 帧，20 个目标图像与 Dataset 输出逐像素一致，包括 RGB/CHW 与时间戳 |
| Padding | 检查首、中、最后完整窗、首个补位窗、末帧；补位重复末动作但标记为 True |
| 真实 batch | 每 split 两个 batch，共 4 个，每个 2 样本；每个 batch 包含完整中间窗和末尾补位窗 |
| 归一化 | 按训练 mean/std 独立复算，并检查反归一化；输入和 padding 保持不变 |
| GPU | RTX 5070 Ti Laptop GPU，CUDA 13.0；4×4 矩阵与 CPU 完全一致，另检查一个真实 batch 的 GPU 归一化 |
| 离线测试 | 9 项通过；覆盖 episode 泄漏/重复/缺失、短 episode、补位临界点、统计污染与篡改、零标准差、配置漂移及归一化 |

真实 batch 形状：

| 字段 | 形状 |
| --- | --- |
| `observation.image` | `[2,3,96,96]` |
| `observation.state` | `[2,2]` |
| `action` | `[2,16,2]` |
| `action_is_pad` | `[2,16]` |
| 按 ACT 规则堆叠相机 | `[2,1,3,96,96]`，第二维是相机数量 |

CPU 归一化往返的最大绝对误差：图像 `8.940696716308594e-08`，状态/动作均为 `7.62939453125e-06`；分别低于预声明容差 `1e-6`、`1e-4`。GPU 检查的状态误差为 0，图像/动作最大误差与上述对应值一致。mean/std 标准化不要求输出落在 `[-1,1]`。

episode 0 的长度为 161，锚点 145 的 16 个动作完整；锚点 146 仅最后一个动作补位；锚点 160 有 15 个补位位置。验证子集的 episode 22 在 Dataset 中从位置 157 开始，而原始全局 index 为 2855，已分别检查局部帧号、子集位置和原始全局索引，未混用。

## 可复现命令

仓库根目录、原有本地数据和环境中运行：

```bash
/home/wordwoods/miniconda3/envs/imitation-policy/bin/python -m unittest discover -s act -p 'test_*.py' -v
/home/wordwoods/miniconda3/envs/imitation-policy/bin/python act/inspect_act_dataset.py --device cuda
```

脚本每次新建 `outputs/act/a2_<UTC时间>/report.json`，并记录实际命令、Python 路径、配置/脚本/实现指纹、Git 状态、统计来源、逐窗索引、batch 与数值误差。CPU 模式只可声明数据部分通过。当前环境的沙箱隔离会影响 GPU 访问；本次真实数据/CUDA 检查经授权在沙箱外运行，没有改变驱动或安装依赖。

## 结论边界

运行状态为 `PASS`，表示冻结配置下的有限数据与接口检查通过。全部文件校验不等于全部窗口已解码或全部样本的语义已审计；独立视频参考只覆盖选定四个 episode。未创建 ACTPolicy、执行策略 forward/backward、更新参数、保存恢复 checkpoint 或运行环境 rollout，也没有测试完整 ACT 网络的 GPU 显存与吞吐。

A1/A2 已在当前对话中获准展开，相关机制已有讲解和部分引导式回答；原计划中“A1 正式展开待许可”等文字是此前快照。本轮保留权威计划与 gate 不变，不把代码或检查通过等同于理解验收。后续可基于本报告进入 A3 的定位，再核验实际网络与梯度路径；原有首轮训练/闭环预算不变。
