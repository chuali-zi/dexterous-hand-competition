# 机器人灵巧操作技能专项赛

后续开发先读[开发规范](DEVELOPMENT.md)；日常采用本地开发、GitHub 同步、Rabokit 云端测试。具体命令见[云端操作手册](docs/rabokit-workflow.md)。

**2026-10-17 接入更新：** 用户已处理报名并建立平台应用副本；已实际进入装配副本的云 VS Code，安装 SDK 并验证 GitHub 可拉取。日常开发采用「本地修改 → GitHub → 云端拉取测试」，详见[云端开发与测试步骤](docs/rabokit-workflow.md)。下文截至10月13日的离线总结保留为历史状态，不代表当前仍未登录或未创建应用。

**同日垒砖检查：** 最初的真实 ROS 只读探针通过双臂反馈检查，但当时未发现相机；这一阶段记录保留在[垒砖检查与审核点](docs/rabokit-workflow.md#垒砖副本只读检查与审核点2026-10-17)。后续已接入经用户确认许可的 RGB 相机，当前状态见下段。

**2026-10-28 当前状态：** 实际目标仍是 **10 块砖、4 层砌筑**。最新[固定动作与桌面承托试验](docs/simple-table-transfer.md)已在本地模型中完成约99mm脱桌搬运、两砖叠放和撤手保持3秒。第80轮改为先由下层砖承重再推齐，把上下砖中心误差从22.066mm降到5.605mm，撤手后三秒漂移0.008mm；但下层砖也移动约6.968mm。进一步推向相邻两块下层砖的跨砖试验失败，降低推指的对照最终落回桌面。当前没有实现指定区域、正确层级或原生砌筑；局部结果不能当作完整任务成功。

第78–80轮全部在本地完成，未唤醒平台。第80轮四次动力学循环墙钟合计109.250秒，均保留失败条件和完整轨迹；不为5mm局部研究目标继续微调，下一结构优先约束推移时的偏航、侧移及底层随动。第77轮最近一次云端仅做资产只读检查，未启动仿真，启动／检查／休眠窗口约7–8分钟，非账单时长，已休眠。本地100g仍是建模假设，桌边标定、完整机械臂碰撞及原生复现也未解决。历史过程见[方向复盘](docs/grasp-strategy-reset.md)、[第二接触研究](docs/two-contact-study.md)及[RGB接入与接触实测](docs/rabokit-vision.md)。下方10月13日内容属于历史离线阶段，不代表当前平台进展。

目标：围绕 **2026 年第十三届华北五省（市、自治区）大学生机器人大赛**，优先用固定动作和必要反馈完成砌筑，在实际验证基础上提高自动化与可靠性。最终以机器人完成任务和实际得分为准。

赛事日程更新：父任务已核2026-10-13 15:23[官方改期公告](https://robot.bistu.edu.cn/tzgg/b13ac6deec9c497d99bb0c05ddee9ba8.html)，当前决赛为**11月14日**，原11月7日已被替代；报名截止日期不变，既有10月26日17:00节点继续保留。首次答疑仍为10月13日20:00、腾讯会议474-645-560。来源、历史日期和本环境访问限制见[赛事更新](docs/event-updates.md)，本任务未发送新的提醒。

截至 **2026-10-13**，本仓库已完成前置研究、离线记分及**可执行的RGB抓放闭环实验、故障恢复、逐帧记录/回放与批量评估**。固定单块基线已验收；用户随后恢复无需平台的自主推进，本轮新增[多块RGB关联与依赖规划](docs/multi-block-planning.md)，只输出候选任务，不执行多块搬运或计分。官方平台实测继续暂搁。**本任务未代选组或报名、未连接官方比赛平台、未执行机器人动作、未训练或下载模型权重**。用户实际队伍、组别和报名状态待本人确认。本云工作区只是代码与调研环境。

**固定代码362e9a8的纯离线统一独立验收已通过**：R1–R6、N1、M1、C1/C2全部关闭，无阻断本轮离线验收的剩余问题。完整报告、复现入口与后续边界见[最终离线验收](docs/offline-acceptance.md)。比赛工程尚未完成，物理稳定/停止、数据真实性及物理执行验证标志仍为false。

多块模块另有独立验收：固定86e12ae复验通过，MB1–MB3在声明的离线色标/块体代理范围关闭，24项独立近邻正常例通过。[转交结论与精确范围](docs/evidence/multi-block-independent-review.md)单独保存，不延伸到真实执行。候选仍是条件几何建议；相同RGB不能排除两视图实际看见不同同色物体，对应真实性、物理完整性和执行可用性保持false。

本轮已实际核验[现有执行拒绝与接入准备度](docs/readiness-stop.md)：五类请求均拒绝，包括伪造认证标志，未调用SDK。下一有用控制开发需要正式场景与实际模板资料；当前在真实资料门停下，不扩张假设场景、不重复请求登录或付费。

初步推荐 **仿真垒砖 + 全自主 + 传统视觉/几何/状态机闭环**：先验证每块释放后稳定，再逐步增加形状和颜色分。正式场景、资格、设备与平台资源尚待确认；推荐不代替用户选组。ACT 是有条件的局部技能备选，Diffusion Policy 次之；当前不把 VLA、大规模强化学习或 REA 逆向工具放进关键路径。

## 阅读导航

| 文件 | 用途 |
|---|---|
| [开发规范](DEVELOPMENT.md) | 开发位置、代码同步、测试、平台入口与提交收尾的统一约定 |
| [云端操作手册](docs/rabokit-workflow.md) | Rabokit 初始化、日常命令、实测结果与已知问题 |
| [最终离线验收](docs/offline-acceptance.md) | 固定362e9a8的完整独立结论、关闭状态与本轮收束 |
| [多块观测与规划](docs/multi-block-planning.md) | 新阶段：本地ID、颜色/支撑依赖、失败重规划、RGB证据及限制 |
| [多块审查修复](docs/multi-block-review-fixes.md) | MB1–MB3独立反例、先红后绿、空间/支撑/完整性门与观测歧义 |
| [多块独立验收](docs/evidence/multi-block-independent-review.md) | 固定86e12ae的父任务转交结论、24例正常近邻与关闭范围 |
| [接入准备核验与资料门](docs/readiness-stop.md) | 实际执行拒绝、真实缺项、停止依据与最少资料 |
| [赛事详细要求](docs/competition-requirements.md) | 资格、报名、程序、计分、提交及审计证据整合 |
| [原件精确证据](docs/evidence/official-excerpts.md) | 父任务复核的条款短引文、页码与适用范围 |
| [规则矩阵](docs/rules-matrix.md) | 分组、计分、日期、冲突、正式/演练边界 |
| [赛事更新与日程历史](docs/event-updates.md) | 10月13日决赛改期、报名/答疑节点与新增资料检查覆盖 |
| [来源与证据等级](docs/sources.md) | 一手 URL、访问限制、版本、论文章节、规则页码 |
| [官方 SDK 注意事项](docs/sdk-notes.md) | 接口陷阱、控制输入白名单、待验收适配层 |
| [开源项目比较](docs/open-source.md) | 8 个项目的许可、接口、成本、适配与淘汰原因 |
| [论文解读](docs/papers.md) | 4 篇原始论文与可证伪的本赛实验 |
| [抓持方向复盘与新调研](docs/grasp-strategy-reset.md) | 11项相关研究、当前失败假设、实际本地对照与路线调整 |
| [固定动作与桌面承托](docs/simple-table-transfer.md) | 第76–80轮：真实桌边、局部叠放、支撑面推齐与跨砖失败、平台预算 |
| [技术路线及首周计划](docs/roadmap.md) | 选组、恢复策略与有条件的历史首周计划 |
| [答疑问题](docs/questions.md) | 按得分/失格风险排序的待确认项 |
| [离线加固与审查](docs/offline-hardening.md) | 历史审查反例、修复与关闭时间线 |
| [历史噪声定位](docs/noise-robustness.md) | 单帧失败根因、多帧估计、匹配对照、未见夹具及限制 |
| [回放状态/时间修复](docs/replay-state-time.md) | 独立复验状态、完整控制路径、真实缓存包时间与失败依据 |
| [逐帧稳定与M1](docs/motion-stability.md) | 定位/稳定分离、完整真值窗口、运动负例及完成率回落 |
| [审查状态与C1/C2](docs/replay-residuals.md) | 回执年龄/超时回退的修复历史及最终关闭 |
| [证据协议v5](docs/protocol.md) | 时间/来源/schema、阶段门、配置和重建回放 |
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
python -m hand_competition.multi_block blueprint configs/multi-block.json
python scripts/multi_block_validation.py --output runs/my-multi-block-scan
python -m hand_competition.multi_block replay runs/my-multi-block-scan/occlusion
python scripts/multi_block_review_validation.py --output runs/my-review-regressions
python scripts/readiness_audit.py --output runs/my-readiness-audit
```

`hand_competition.closed_loop`会在本地合成RGB场景执行内部动作协议并生成图像/日志；`hand_competition`旧入口仍只计算人工/测试标注分数。`multi_block`独立入口只观察预先指定的RGB快照并重算规划，不调用机器人。全部不是官方成绩。`configs/offline.json`和`configs/multi-block.json`是离线约定，不是平台配置；尚未接入ROS/SDK或实现官方几何判分。实验目录必须不存在，换新目录可重跑。

## 接入与提交边界

进入官方平台前，先核对实际 `agent_system` 模板版本、传感器许可、命名空间、控制频率、收费和算力。当前没有可执行的官方平台适配器。本阶段不依赖官方接入；登录与资源事项保留在后续实测交接文档，当前不请求操作。真机运行、付费资源、报名和作品提交均不在本次授权范围。

未来正式提交需要：创建者从正确工程/分支生成分享链接，锁定代码与配置；补充运行逻辑、实际启动方法、团队分工，声明第三方库、模型、API 与难度等级。作品锁定截止尚未确认，**不能用 10 月 26 日报名截止替代**。不在此仓库保存凭据、视频数据集、第三方代码或模型权重，也未授予本仓库新的开源许可。
