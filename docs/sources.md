# 来源清单与证据等级

初始检索/读取日期：**2026-10-12**。工作区 UTC 时间约 05:56 起，对应北京时间 13:56 起。2026-10-13另增日程改期维护，详见[赛事更新](event-updates.md)。维护日期不代表官方规则全文重新发布。

- **A：本次直接读取**原始仓库 README、许可及所列源码；或作者论文 HTML 的方法/接口/实验/限制章节。代码未执行，论文实验未复现。
- **B：父任务已核实摘录**。官方站点在本环境无法访问，依赖委托输入，不能声称独立核实全文。
- **C：工程推论/待确认**。推荐、实验门槛、恢复方案与成本判断，不能当官方规定。

不使用搜索摘要或第三方博客作为关键技术结论。论文版权、代码许可、模型权重/基础模型条款分开看；本次没有复制论文全文、第三方实现或权重进本仓库。

## 官方来源（B）

| ID | 原始来源 | 版本/定位与本次访问结果 |
|---|---|---|
| O1 | [10月11日演练与答疑通知](https://robot.bistu.edu.cn/tzgg/3d95cef5dc594ecbbc11333770b5cbe0.html) | 父任务核实发布时间 2026-10-11 15:01；本次 web 不可访问，普通 HTTPS 请求代理返回 403；不是10月13日之后的“最新通知”标签 |
| O2 | [演练手册](https://robot.bistu.edu.cn/docs/2026-10/780da6f7f1834fdcb7f13c226149662a.pdf) | 2026-10 路径；本环境 web 不可访问；后续父核补充共5页，第1页限定仿真、第5页复议条款，见精确证据表 |
| O3 | [报名通知](https://robot.bistu.edu.cn/tzgg/ce6d4ae0373c4e5cb7aa84372e4d96e2.html) | 本次 web 超时；日期/资格摘录来自父任务 |
| O4 | [教委方案](https://robot.bistu.edu.cn/docs/2026-09/f402c4be17fa4ca2ba45f1445abacd80.pdf) | 2026-09 路径；本次 web 不可访问，页码未获 |
| O5 | [2026正式规则PDF](https://robot.bistu.edu.cn/docs/2026-09/7746a8e90714439cad8eea91355e0e06.pdf) | 灵巧操作印刷 pp.100–109；PDF物理 pp.102–111；本次 web 不可访问，HTTPS 代理 403。本环境没有全文散列；后续父核补充关键条款单页及短引文，见精确证据表 |
| O6 | [LBot双臂机器人手册](https://www.rabokit.com/manual/lbot_dual_arm) / [公开只读数据接口](http://www.rabokit.com/v1/api/manual/docs/lbot_dual_arm) | 父任务读 2026-10-10 文档；本次网页不可访问，HTTP接口403；没有绕过限制 |
| O7 | [腾讯问题表](https://docs.qq.com/sheet/DUmdlU3JJSkF6eWJu?tab=BB08J2) | 初期 web 不可访问、父任务401；2026-10-22 通过正常浏览器只读访问，读取第 8–32 行中的部分问题与答复，非全表审核。D10 确认 50 mm 积木、D23 初始摆放不变、D20 相机自行添加、D31 各类传感器有数量限制；D11 与 D16 的 grasp_force 说明相互矛盾，按实际 SDK 和对应手册判断。D30 真值话题问题未答，未找到允许改物理配置的明确答复；详见[第 50 轮记录](rabokit-vision.md#接触部位与摩擦求解诊断2026-10-22第-50-轮仅本地)。本次选定单元格内容属于直接读取 A，不追溯改变其他官方来源的 B 等级 |
| O8 | [后续通知索引](https://robot.bistu.edu.cn/tzgg/index.html) | 作为后续复核入口；本次未读取索引内容，不声称无新通知 |
| O10 | [10月13日决赛改期公告](https://robot.bistu.edu.cn/tzgg/b13ac6deec9c497d99bb0c05ddee9ba8.html) | 父任务核实页面显示2026-10-13 15:23，决赛改为11月14日、报名截止日期不变；本环境web不可取得，普通HTTPS隧道403；全文与页面哈希未取得，证据等级B |
| O11 | [首次答疑官方直播入口](https://meeting.tencent.com/l/GkiFvA6gq8KR) | 父任务转交；10月13日20:00、腾讯会议474-645-560仍依据O1；本轮仅记录入口，未进入/观看直播 |

O6 的相关父任务文档 slug：`home`、`05-interface`、`06-sensors`、`14-controller`、`dev_kit-agent-template`、`rabo_dev_kit`、`18-devenv`、`19-power`。页面与公开接口路径分别为 `https://www.rabokit.com/manual/{slug}` 和 `http://www.rabokit.com/v1/api/manual/docs/{slug}`，目录接口为 `http://www.rabokit.com/v1/api/manual/docs`。本次未逐一读取，不把这些路径当作平台控制 API。

后续能读全文时，保留原始文件散列、下载时间、印刷页与 PDF 页的双重定位；规则/场景/答疑单独做版本，不静默覆盖冲突。

## 开源/可读源码（A，固定快照）

通过 `git ls-remote` 只读获取公开分支 SHA，再直接下载该 SHA 下的原始文件。OpenCV 选 `4.x` 分支以对应 calib3d 接口；其他为读取时 HEAD，**不是建议安装最新开发分支**。实际平台集成还须选择与 ROS/SDK 兼容的发布版。

| ID | 项目与快照 | SHA | 实际成功读取 |
|---|---|---|---|
| R1 | [opencv/opencv](https://github.com/opencv/opencv/tree/802cf95aa3d42816995695bf68a86b3532f12760) | `802cf95aa3d42816995695bf68a86b3532f12760` | `README.md`, `LICENSE`, `modules/calib3d/include/opencv2/calib3d.hpp` |
| R2 | [AprilRobotics/apriltag](https://github.com/AprilRobotics/apriltag/tree/b7c0ebe9aa20f82ec7a828579004f9e706bfecd9) | `b7c0ebe9aa20f82ec7a828579004f9e706bfecd9` | `README.md`, `LICENSE.md`, `apriltag.h` |
| R3 | [moveit/moveit2](https://github.com/moveit/moveit2/tree/a9004a43151b7d1fd50d462b8db6d4d280b0cc21) | `a9004a43151b7d1fd50d462b8db6d4d280b0cc21` | `README.md`, `LICENSE.txt`, `moveit_ros/planning_interface/move_group_interface/include/moveit/move_group_interface/move_group_interface.hpp` |
| R4 | [tonyzhaozh/act](https://github.com/tonyzhaozh/act/tree/742c753c0d4a5d87076c8f69e5628c79a8cc5488) | `742c753c0d4a5d87076c8f69e5628c79a8cc5488` | `README.md`, `LICENSE`, `policy.py`, `constants.py`, `utils.py` |
| R5 | [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy/tree/5ba07ac6661db573af695b419a7947ecb704690f) | `5ba07ac6661db573af695b419a7947ecb704690f` | `README.md`, `LICENSE`, `diffusion_policy/policy/base_image_policy.py` |
| R6 | [openvla/openvla](https://github.com/openvla/openvla/tree/c8f03f48af692657d3060c19588038c7220e9af9) | `c8f03f48af692657d3060c19588038c7220e9af9` | `README.md`, `LICENSE`, `prismatic/extern/hf/modeling_prismatic.py` |
| R7 | [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose/tree/a1b694b83e633c2cb6115b9063d940a687759392) | `a1b694b83e633c2cb6115b9063d940a687759392` | `LICENSE`, `readme.md`, `run_demo.py` |
| R8 | [morluto/rea](https://github.com/morluto/rea/tree/69e097a8ca256ab97ee4b2883739447784a5f90e) | `69e097a8ca256ab97ee4b2883739447784a5f90e` | `README.md`, `LICENSE` |

[机器可读源锁](evidence/source-lock.json) 保存访问时间、URL、字节数、SHA-256 和错误。初次猜测 `LICENSE`（AprilTag/MoveIt）、`README.md`（FoundationPose）返回404，已改用仓库实际文件名并成功读取；404保留在源锁而不作为缺失许可的结论。GitHub API 在环境被403拒绝；使用正常可用的公开 Git/raw 入口读取，没有更改网络或凭据。

## 论文与官方技术文档（A）

| ID | 原文、版本、作者/组织 | 实际读取定位 | 版权/复现边界 |
|---|---|---|---|
| P1 | [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/html/2304.13705v1)，2023-04-23 v1；Tony Z. Zhao、Vikash Kumar、Sergey Levine、Chelsea Finn；Stanford/Meta/UC Berkeley | III、IV（含 IV-C）、V、VI、VII；原文为 HTML，不捏造 PDF 页码 | 页面 arXiv perpetual non-exclusive；代码 R4 MIT，文献版权不等于代码许可 |
| P2 | [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/html/2303.04137v5)，2023首发，2024-03-14 v5；Cheng Chi、Zhenjia Xu、Siyuan Feng、Eric Cousineau、Yilun Du、Benjamin Burchfiel、Russ Tedrake、Shuran Song；Columbia/TRI/MIT/Stanford | §2、§3.4、§6–7、§9；本次读扩展版，非只有2023会议版摘要 | 页面 arXiv perpetual non-exclusive；代码 R5 MIT；仓库README作者列表与v5不同，勿混为同版 |
| P3 | [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/html/2406.09246v3)，2024-09-05 v3；Moo Jin Kim、Karl Pertsch、Siddharth Karamcheti 等18位；Stanford/UC Berkeley/TRI/Google DeepMind/Physical Intelligence/MIT | §3.2–3.5、§4–6；[摘要页含完整作者](https://arxiv.org/abs/2406.09246) | 论文 CC BY 4.0；代码 MIT，权重继承底座条款，不能当 MIT 权重 |
| P4 | [FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects](https://arxiv.org/html/2312.08344v2)，2023预印本，2024-03-26 v2/CVPR 2024；Bowen Wen、Wei Yang、Jan Kautz、Stan Birchfield；NVIDIA | §3.2–3.4、§4.5、附录 Implementation Details / Limitations | 论文 CC BY 4.0；R7代码/数据为 NVIDIA Source Code License，非商业研究/评估限制 |
| D1 | [OpenCV Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html) | 官方 calib3d 文档与 R1 头文件中的 `solvePnP`、`calibrateCamera`、`calibrateHandEye` 接口 | 已读文档/声明，未安装/实测；4.x链接重定向到4.13.0 |

论文 PDF 普通下载请求被本环境代理403拒绝；随后通过可用的 web 原始 arXiv HTML 读取关键章节。没有在仓库伪造 PDF 哈希。论文发表结果只说明作者设置，不作为 LBot/O6 或本赛性能证据。

## 规则审计与原件再次复核（2026-10-12增补）

[完整赛事要求报告](competition-requirements.md)已整合父任务转交的独立规则审计。读取主体必须区分：本环境此前失败不代表父任务没有读过原件；“审计原页/索引正文”指另一次审计的访问，不冒充本工作环境直接访问。

- **B1 父核原件**：保留首轮已读PDF/平台文档事实。父任务原件读取者再次核对PDF及页面图像后，提供[11项精确页码、短引文与适用范围](evidence/official-excerpts.md)。届次为A印刷1/PDF3；自主定义印刷101/PDF103；装配物料印刷102/PDF104；装配次数/处罚/同分印刷104/PDF106；垒砖基础与颜色印刷107/PDF109；最终状态印刷108/PDF110；附表印刷109/PDF111；仿真手册B物理5复议48小时，B物理1限定仿真。没有本地原件，仍不填文件哈希。
- **B2 审计索引正文**：O3报名通知，2026-09-11 09:04，“一”资格/人数/限额/特别推荐，“二”学校赛区报送/电子表公章/人员冻结/截止，“三”邮箱/命名/超额删除，“四”联系方式。独立审计取得索引正文但原页超时；本轮按转交结果整合，未重新宣称直读。
- **B3 审计原页交叉印证**：[北京工商大学通知](https://caai.btbu.edu.cn/tzgg/f2513a151a5f4f75a272e61e80e754f9.htm)“四、报名方式及提交材料”支持盖章；[天津工业大学通知](https://jwc.tiangong.edu.cn/2026/0928/c1328a117927/page.htm)“三、参赛要求”支持共性队伍要求；[太原科技大学转载省赛通知](https://cxcy.tyust.edu.cn/info/1106/1252.htm)“决赛时间”印证原定11月7日，现已被O10改期公告替代。访问主体是独立审计，本环境未重复访问。这些学校截止不适用于本用户。
- **O9 报名表原件**：[2026报名表XLS](https://robot.bistu.edu.cn/docs/2026-09/1c437a5fed2c4f1987364b86613a590b.xls)在独立审计未取得；未核字段/签章栏/命名完整格式，不虚构其内容。

精确证据更新不覆盖开源项目源锁；[规则变更记录](evidence/rules-change-log.md)单独追踪适用范围与修订。用户已有学校背景是北京信息科技大学，实际本次参赛学校/队伍和报名状态待本人确认；不从其他高校通知推断用户资格或截止。
