# 机器人灵巧操作技能专项赛

目标：围绕 **2026 年第十三届华北五省（市、自治区）大学生机器人大赛**，以可复现、合规、可靠的自主得分能力争取获胜。调研不是获胜保证。

截至 **2026-10-12（北京时间）**，本仓库已完成前置研究、离线记分及**可执行的RGB抓放闭环实验、故障恢复、逐帧记录/回放与批量评估**。当前反馈是合成RGB运动学场景；真实官方模板和运行环境仍待登录及费用确认。**本任务未代选组或报名、未连接官方比赛平台、未执行机器人动作、未训练或下载模型权重**。用户实际队伍、组别和报名状态待本人确认。本云工作区只是代码与调研环境。

初步推荐 **仿真垒砖 + 全自主 + 传统视觉/几何/状态机闭环**：先验证每块释放后稳定，再逐步增加形状和颜色分。正式场景、资格、设备与平台资源尚待确认；推荐不代替用户选组。ACT 是有条件的局部技能备选，Diffusion Policy 次之；当前不把 VLA、大规模强化学习或 REA 逆向工具放进关键路径。

## 阅读导航

| 文件 | 用途 |
|---|---|
| [赛事详细要求](docs/competition-requirements.md) | 资格、报名、程序、计分、提交及审计证据整合 |
| [原件精确证据](docs/evidence/official-excerpts.md) | 父任务复核的条款短引文、页码与适用范围 |
| [规则矩阵](docs/rules-matrix.md) | 分组、计分、日期、冲突、正式/演练边界 |
| [来源与证据等级](docs/sources.md) | 一手 URL、访问限制、版本、论文章节、规则页码 |
| [官方 SDK 注意事项](docs/sdk-notes.md) | 接口陷阱、控制输入白名单、待验收适配层 |
| [开源项目比较](docs/open-source.md) | 8 个项目的许可、接口、成本、适配与淘汰原因 |
| [论文解读](docs/papers.md) | 4 篇原始论文与可证伪的本赛实验 |
| [技术路线及首周计划](docs/roadmap.md) | 选组、恢复策略、实验门槛、角色分工 |
| [答疑问题](docs/questions.md) | 按得分/失格风险排序的待确认项 |
| [闭环实验](docs/closed-loop-experiments.md) | 自动执行、RGB后验、恢复、回放、批量评估与反馈层级 |
| [官方接入交接](docs/platform-access.md) | 实际访问、登录/费用边界、所需模板与最小人工步骤 |
| [离线评估说明](docs/offline-evaluation.md) | 非官方记分、最终状态、日志格式与示例 |
| [验证记录](docs/validation.md) | 已跑与未跑分开记录 |
| [来源锁定清单](docs/evidence/source-lock.json) | 仓库 SHA、直接读取的文件 URL 和 SHA-256 |

官方站点在本环境不可达；规则和 SDK 事实采用任务提供的父任务已核实摘录，**本次未读到规则 PDF 全文**。开源项目的 README、许可、关键接口与论文 HTML 关键章节已直接访问。独立规则审计与父任务原件再次核对结果已整合；精确引文/页码见证据表，不能将继承证据写成本工作环境的独立复核。

## 本地启动（仅离线，Python 3.10+，无第三方依赖）

在仓库根目录执行：

```bash
python scripts/check.py
python -m hand_competition.closed_loop run --output runs/my-first-trial
python -m hand_competition.closed_loop replay runs/my-first-trial
python -m hand_competition examples/brick-final-state.json
python -m hand_competition examples/assembly-final-state.json
```

`hand_competition.closed_loop`会在本地合成RGB场景执行内部动作协议并生成图像/日志；`hand_competition`旧入口仍只计算人工/测试标注分数。两者都不是官方成绩。`configs/offline.json`是计划约定，不是平台配置；尚未接入ROS/SDK或实现官方几何判分。实验目录必须不存在，换新目录可重跑。

## 接入与提交边界

进入官方平台前，先核对实际 `agent_system` 模板版本、传感器许可、命名空间、控制频率、收费和算力。当前没有可执行的官方平台适配器。已核实打开agent工程可能启动计费环境，display分享页可能自动运行；先完成登录与资源/费用审查再接入，见交接文档。真机运行、付费资源、报名和作品提交均不在本次授权范围。

未来正式提交需要：创建者从正确工程/分支生成分享链接，锁定代码与配置；补充运行逻辑、实际启动方法、团队分工，声明第三方库、模型、API 与难度等级。作品锁定截止尚未确认，**不能用 10 月 26 日报名截止替代**。不在此仓库保存凭据、视频数据集、第三方代码或模型权重，也未授予本仓库新的开源许可。
