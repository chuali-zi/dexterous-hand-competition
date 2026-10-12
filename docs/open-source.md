# 开源与可读源码项目：针对瓶颈筛选

截至 2026-10-12。已直接读取 8 个原始仓库的 README、许可，及有用的接口源码；[固定版本入口](evidence/project-links.md) 和 [源锁](evidence/source-lock.json) 可追溯。均未安装或运行，不代表在 LBot/O6 上可用。FoundationPose 是有限制的源码发布，不笼统称为宽松开源。

## 候选比较

| 项目/作者或组织/年份 | 许可与代码状态 | 输入→输出、硬件/算力证据 | 本赛瓶颈、改造与结论 |
|---|---|---|---|
| **OpenCV 4.x** / OpenCV团队及贡献者；持续项目，2026读取快照 [R1/D1](sources.md) | 已读 Apache-2.0；C++/Python公开代码，第三方模块另核 | `calibrateCamera`、`solvePnP`、`calibrateHandEye`；角点+内参+几何→外参/姿态；可CPU使用，不要求训练GPU，未测本场帧率 | 优先用于标定、颜色/几何、重投影检验。平面歧义、同色遮挡、镜面工具需多视角/置信度；不是自动生成精确3D真值 |
| **AprilTag 3** / AprilRobotics、密歇根大学体系；持续项目，2026快照 [R2](sources.md) | BSD-2-Clause；C检测库公开，带Python支持 | `apriltag_detector_detect`、`estimate_tag_pose`；灰度+tag尺寸+fx/fy/cx/cy→相机系tag位姿；不要求GPU | 若许可标定板/标签，可做外参锚点；不能未经允许给比赛砖贴标。姿态相机系不是base_link；先核实尺寸和变换方向。条件候选 |
| **MoveIt 2** / MoveIt贡献者，PickNik领导开发；持续ROS2项目，2026快照 [R3](sources.md) | BSD-3-Clause（根许可）；组件依赖须单独保留声明；C++接口公开 | `MoveGroupInterface`的规划/笛卡尔路径接口；URDF/SRDF、碰撞几何、关节状态、规划组和控制器集成；基础规划不要求GPU，未测耗时 | 能帮助肘姿/自碰/路径检查，但没有现成LBot集成证据；SDK若不暴露轨迹执行能力则仅可规划/验证。不要为了“用ROS”重搭整个栈 |
| **ACT** / Tony Z. Zhao等；2023 [R4/P1](sources.md) | MIT；训练/评估、模拟任务、数据加载源码公开，README链接示教 | `ACTPolicy(qpos,image,...)`→动作chunk；HDF5示教；原论文双6DoF臂+夹爪，14维目标；论文单11GB RTX2080Ti约5h训练/约10ms推理（作者设置） | 小模型候选优先级高于VLA；本赛双7轴+双6电机手，动作/观测维度、手状态缺失、同步和示教需重构。不能补零冒充手反馈。仅在固定技能成为瓶颈时小试 |
| **Diffusion Policy** / Cheng Chi等，Columbia/TRI/MIT，仓库real-stanford；2023及2024扩展 [R5/P2](sources.md) | MIT；Dataset/EnvRunner/Policy接口、训练配置和checkpoint链接公开 | `predict_action(obs_dict)`→`action`序列；视觉+本体历史；公开实机Push-T为UR5/UR5e、2×D415、SpaceMouse；Linux NVIDIA GPU，论文3080约0.1s推理（特定配置） | 多峰抓姿/连续接触候选；替换机器人接口、动作归一化、时间戳/延迟。RGB-D采集设备存在不等于本赛允许深度。比ACT更应先量时延；不直接运行实机示例 |
| **OpenVLA** / Moo Jin Kim、Karl Pertsch、Siddharth Karamcheti等；2024 [R6/P3](sources.md) | 代码MIT；README明确模型继承Llama2条款；公开权重入口，本次未下载 | RGB+文本→`predict_action`，原版7维单臂末端动作；7B，bf16推理约15GB/4090约6Hz（论文）；README LoRA示例80GB A100，减batch最低约27GB | 不是双臂灵巧手即插即用；四路图像、动作头、手控制需适配/数据。固定颜色任务没有足够语义收益，当前不进入主线；不外调收费模型 |
| **FoundationPose** / NVIDIA，Bowen Wen等；2023预印本/CVPR2024 [R7/P4](sources.md) | NVIDIA Source Code License §3.3限非商业研究/评估；README称代码/数据均适用；权重/资产需继续逐项核实 | `register(K,rgb,depth,ob_mask,...)`→6D pose，`track_one`→跟踪；需RGB-D、mask、CAD或参考视图；CUDA/PyTorch3D/NVDiffRast；作者3090初始化约1.3s/跟踪约32Hz | 对工具姿态有条件价值；默认4路RGB不满足深度入口。不把“model-free”误解成不需要深度/对象建模。当前搁置，若获合法深度再做感知对照 |
| **REA / Reverse Engineer Anything** / morluto；2026（许可证版权年） [R8](sources.md) | MIT；本次只读README/许可，未安装或检查所有后端 | 面向agent/CLI的本地二进制、JS/Electron、.NET、网页分析；Node22.19+/24.11+/26+及npm；原生分析依赖Hopper/Ghidra/IDA等 | 不能解决标定、接触、抓持、恢复或计分。对获授权且仅有二进制的SDK故障定位最多是间接工具；当前已有公开文档/模板路线，不需要REA，不逆向比赛评分器或读取隐藏状态 |

年份是论文首发/版本或明确的快照年；不伪造持续软件项目的首发年份。上表硬件数字是作者报告，不是资源最低保证，也不是本环境的实测预算。ACT/DP/VLA学习到的行为仍需要失败检测与恢复；没有模型能通过名称自动补出缺失传感器。

## 接口核实细节与最小试用范围

- **OpenCV**：读取了4.x `modules/calib3d/include/opencv2/calib3d.hpp`。PnP需要已知3D点与匹配2D点；手眼标定需要多组姿态，不能只拍一张。先离线验证单位、坐标方向、留出点和颜色鲁棒性，再考虑集成。
- **AprilTag**：读取 `apriltag.h` 与README姿态章节。检测器拿到tag姿态后还需T_base_camera；标签平面歧义与检测噪声不等于机器人标定精度。未直接阅读AprilTag论文，不把README当论文复现。
- **MoveIt**：读取 `move_group_interface.hpp`；能够计算路径不证明SDK能执行路径。先查官方URDF/SRDF、joint names、限位/碰撞模型和控制器，再决定是否值得引入。开发分支快照不是可安装的比赛依赖锁。
- **ACT**：读取 `policy.py`、`constants.py`、`utils.py`。现有ALOHA归一化和夹爪参数不能照搬O6；HDF5应存真实可用观测、指令、图像时间和手命令历史。数据格式借鉴不意味着复制论文的仿真真值脚本。
- **DP**：读取 `base_image_policy.py` 及README接口章节；需要自定义Dataset、EnvRunner和本赛执行适配，动作chunk长度/执行窗口由闭环频率决定，不能只按训练loss选模型。
- **OpenVLA**：读取 `modeling_prismatic.py` 及README；`unnorm_key`决定输出解归一化，`bridge_orig`是示例机器人数据统计，不是LBot的尺度。不要把示例action直接发送给SDK。本文评估原版OpenVLA，不把后续衍生版本性能拼接到它身上。
- **FoundationPose**：读取 `run_demo.py` 的register/track调用；mask来源必须合法，公开数据中的物体pose不能流入控制。比赛用途与非商业限制是否相容尚未判断，当前不集成。

## 采用顺序和退出条件

先使用“标准库记录/测试 → 获准输入下的OpenCV几何 → 单臂状态机与视觉后验”；AprilTag仅在标定物获准时考虑；MoveIt在模型和控制器可接通时采用。ACT/DP仅对明确反复失败的局部技能做预算可控的对照。FoundationPose、VLA和REA不加入默认依赖。

引入任何依赖前固定发布版/提交、校验许可和转发义务、记录实测显存/延迟、审阅下载/安装脚本、在隔离环境小样本验收。当前仓库只引用研究材料，**没有vendoring第三方代码或运行其安装脚本**。
