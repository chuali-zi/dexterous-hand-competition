# 本地开发、Rabokit 云端测试

本页是操作手册与实测记录；后续开发的统一要求见[开发规范](../DEVELOPMENT.md)。

2026-10-17 实际进入用户建立的「电动工具装配 副本」及其云 VS Code。用户确认报名已处理，本页只说明开发链路，不代表已经选定最终参赛小项。

## 三个位置

| 位置 | 用途 |
|---|---|
| 本地 `D:\race\dexterous-hand-competition` | 日常写代码和运行离线检查 |
| 现有 GitHub `chuali-zi/dexterous-hand-competition` 的 `main` | 本地与云端之间同步项目源码 |
| 云端 `/workspace/agent_system` | Rabokit 的控制器工程，平台运行按钮启动这里的 `main.py` |
| 云端 `/workspace/agent_system/competition` | 从上述 GitHub 克隆的项目，在这里拉更新并执行测试 |

云端顶层 `agent_system` 的 `origin` 是 Rabokit 自己的工程仓库，并非 GitHub。保留两者各自的远端，不替换 `origin`。顶层只放启动入口；算法和测试在 GitHub 项目中维护。`competition/` 应加入顶层 `.gitignore`，避免把嵌套 Git 仓库混进平台提交。

## 模板文件究竟做什么

- 顶层 `README.md`：通用 LLM agent 的教程，不是比赛解法或完整机器人操作步骤。
- 原始 `main.py`：默认运行 `example_agent`；本次实测只打印初始化成功并退出，不控制机器人。
- `__init__.py`：Python 包入口和导出约定，不是代码上传入口。
- `test.py`：SDK 双臂/手部动作演示，包含大幅运动，不是单元测试，也不会自动被平台运行按钮执行。其注释中的 `python main.py` 不能用于证明它实际被执行。
- `requirements.txt`：依赖清单。初次进入本次环境时 SDK 尚未安装，执行 `python -m pip install -r requirements.txt` 后安装成功，其中 `rabo_robolib` 为 3.0.0。

## 首次设置

在 Rabokit 云 VS Code 的终端执行（本轮已克隆，不要重复克隆）：

```bash
cd /workspace/agent_system
git clone --depth 1 https://github.com/chuali-zi/dexterous-hand-competition competition
python -m pip install -r competition/rabokit/requirements.txt
```

将项目的 `rabokit/main.py` 复制到顶层 `main.py`，将 `/competition/` 加入顶层 `.gitignore`。这两个平台工程改动需要提交推送到其已有 Rabokit 远端，才能在平台重置后恢复入口。不要把密钥或环境变量导出到源码。SDK 本身通过包仓库安装，不复制到 GitHub。

## 每次开发和测试

本地开发并检查后，正常提交、推送本项目到现有 GitHub。云端只需：

```bash
cd /workspace/agent_system
git -C competition pull --ff-only
python competition/scripts/check.py
```

在平台点击「启动仿真」。仿真环境和控制程序是两个进程：只启动仿真，机器人不会自动完成任务。取消勾选「启动当前智能体工程」可手动分步调试。

仿真就绪后：

```bash
# 只订阅左右臂真实关节消息，列出当前场景相机话题
python competition/scripts/rabokit_probe.py

# 在仿真中将左腕第 7 关节增加 0.08 rad，再返回起始角度
python competition/scripts/rabokit_probe.py --motion
```

平台「执行智能体工程」按钮启动顶层 `main.py`。本次提供的入口默认调用只读探针，不自动运动；终端执行 `python main.py --motion` 可调用同一动作测试。程序输出在云 VS Code 终端查看。

本轮也实际验证了勾选「启动当前智能体工程」的启动流程：平台向已有终端发送激活环境、安装依赖和 `python main.py` 的命令，并打印探针 PASS。多开终端时日志可能落在先前的终端，本次在 Terminal 2，而不是另开的空 Terminal 3。短程序已经退出后，平台顶栏仍可能显示「运行中」；先看终端的 PASS、退出码和 shell 提示符，需要时点「停止智能体工程」清理界面状态，避免只凭顶栏判断代码是否还在执行。

后续抓取调试使用独立命名的 `brick-control` 和 `brick-status` 终端。重连后编号可能变化，按名称选择；保留其他终端的待输入内容。当前页面快捷键映射与常见默认值不同，可从命令面板执行 `Terminal: Copy Last Command and Output` 读取最近命令输出，避免依赖复制快捷键。复制的单行 JSON/BASE64 可能带终端换行，解析时只拼回该单行记录，不混入其他日志。输入输出通路先在仿真停止时核对，稳定后再启动仿真。

第三次拇指准备试验的启动命令前混入字符，首个 shell 计时赋值失败；控制程序仍执行，但终端用时字段无效。该轮已按进程运行时间和原文件时标独立核算，保留错误记录。后续输入采用已验证的页面内会话剪贴板及终端粘贴；计时由 Python 子进程记录 UTC 起止、单调墙钟和返回码，减少对交互 shell 变量的依赖。此记录通路已用无动作子进程验证，没有为该验证启动仿真。

24 轮又出现原生按键/粘贴导致命令重复和中文前缀；首次仿真未执行就绪脚本或控制器，仍消耗了启动时间。当前终端输入改用浏览器 DOM 的实际 Terminal textbox `fill` 后 `press('Enter')`，场景启停也用实际按钮定位。先在仿真停止时用无动作命令验证输入、输出和退出码，再启动；不要把粘贴 API 成功当作 shell 命令正确。输出仍通过 `Terminal: Copy Last Command and Output` 获取，它只返回最近已完成命令，长命令运行时读到旧输出不代表应再次启动。计时继续使用 Python 子进程。

大文件通过编辑器复制时，Quick Open 要等待并点击对应文件结果，不能输入后立即 Enter 命中旧结果。只关闭本轮打开的只读结果编辑器；不要关闭用户未保存文件。复制后校验长度及解码，避免对打开大段 BASE64 的编辑器请求完整 DOM 快照。24 轮旧开发页崩溃后，使用同一浏览器中的原工程地址新建开发页恢复，没有修改浏览器安全设置。

25 轮重载开发页后终端编号发生变化，按新页面可见名称重新绑定；原数据和 Git 版本核实后再继续读取。Quick Open 打开结果文件后先核对焦点确为对应 editor，再全选复制；AX 编辑器容器点击不保证焦点正确，剪贴板仍可能是旧终端输出，必须校验长度与解码后才接受该分段。没有为页面断线重启仿真。

短时手部诊断的命令、监视和本地接收脚本在启动前准备完整。冷启动使用单次有上限的只读就绪订阅，消息齐备即结束；在同一已确认运行的实例内等待连接。活动实例期间只执行准备好的测试与状态查询，进程结束后先停仿真，再做本地分析、代码修改和同步。记录控制墙钟及界面仿真时钟，避免把前者当作全部平台运行时间。

## 本次实际发现和范围

云运行环境为 Python 3.12.3、ROS2 Jazzy。已通过云终端读取 SDK 方法并实际接收左右臂关节角。

`LbotDualArm(mode='sim')` 的部件属性为 `left_arm/right_arm/left_hand/right_hand`。3.0.0 在此环境返回客户端代理，不能把导出类的全部 Node 方法当作实例方法；例如直接对 `left_arm` 调 `get_topic_names_and_types()` 会失败。话题枚举应使用独立的 `rclpy` 节点。所有连接探针都应在 `finally` 中关闭 SDK，以免异常退出遗留后台线程。

本次实际装配副本的 ROS 图没有发现本场景的 Image、CompressedImage 或 CameraInfo 话题，场景编辑器里的双臂也显示传感器数为 0。后续要核实正式场景的相机配置/允许的图像接入方式。界面能看到 3D 画面，不代表控制程序能订阅相机。物体 pose 话题的存在不等于比赛允许用真值做决策；本次探针只用机器人关节反馈。

本次腕关节探针已实际通过：4秒内收到左臂198条、右臂197条 JointState；左腕目标增加0.08 rad，实际增加0.07669 rad，随后返回至起始位置附近（第7关节约0.00241 rad）。另行测试左手 close/open，均收到接受回执，ROS手指关节在闭合/张开后出现对应变化；这只是空手动作，未测试夹持物体。SDK关闭时输出过 `cannot use Destroyable because destruction was requested` 后台清理警告，进程返回0；不把它写成无警告运行。

检查结果：新脚本语法及云端运动/返回、只读入口执行通过；164个相对文档链接通过。云端原有回归单次180秒上限前完成96项，随后单独运行最后的观测模块7项通过（4项重叠），覆盖99个不同测试。单次全套命令未完整退出成功，不写成单次99项通过。Windows探索性全套检查遇到默认临时目录权限问题；改用仓库内临时目录后还发现原有运行记录路径使用反斜杠导致的断言失败，已停止本地全套尝试，本轮不改动离线控制源码。

本页的接通测试不代表抓取、装配或垒砖得分，不把离线合成场景结果当作正式仿真成绩。

## 垒砖副本只读检查与审核点（2026-10-17）

本轮进入用户已有的「华北五省大学生机器人大赛 - 垒砖砌筑 副本」。该应用的云端工程与装配副本独立，顶层模板为 `74657f1`，`main...origin/main` 干净；初始没有 `competition/`，ROS/rclpy 已安装，`rabo_robolib` 未安装。没有修改顶层入口、依赖或场景配置。

为完成这次只读检查，项目提交 `612f9a3` 将 SDK 版本查询改为可选：缺包时记录 `sdk_version: null`，仍可通过 ROS 检查；`--motion` 在缺 SDK 时以参数错误退出，不初始化观测或执行动作。本地语法、`--help`、缺 SDK 的只读路径和运动拒绝检查通过；未重跑无关的离线算法全套。

云端从既有 GitHub 项目临时克隆至 `/tmp/competition-probe-20261017`，确认版本 `612f9a3`，仅启动仿真，取消勾选自动运行当前智能体。临时克隆用于检查，不是持久开发目录；后续正式接入仍按本文的 `competition/` 工作流维护。

```bash
python /tmp/competition-probe-20261017/scripts/rabokit_probe.py
```

实际结果：4 秒内左臂 199 条、右臂 198 条 JointState；`camera_topics: []`；`PASS: Rabokit simulation probe`，退出码 0。第二次运行的完整输出保存在云端 `/tmp/competition-probe-result.txt`。只读探针没有发送运动命令。

场景大纲中，底座、左右臂及全部道具均为「传感器 0」；启动后的接口清单列有关节状态、指令、物体真值位姿和时钟/TF，没有相机接口。没有订阅物体真值位姿，不能把这些接口用于补足视觉决策。本轮结束停止了本轮启动的仿真；控制程序已退出。

本轮直接读取[官方传感器支持清单](https://www.rabokit.com/manual/06-sensors)：场景编辑器可以在零件上添加相机；彩色图像为 `<robot>/<cam>/image_raw`（rgb8），内参为 `camera_info`，相机默认 320×240、共用发布频率默认 15 Hz。此处只是平台能力，不能推出赛事允许增改相机。

**下一审核项：** 是否以垒砖为优先开发方向，在用户的开发副本中添加一台固定 RGB 相机，先验证图像接收和标定，再做单块抓放。建议以 320×240、5 Hz 起步，仅使用 RGB 与相机内参；不启用深度/点云，不把自定义副本结果记为正式成绩。相机安装点和朝向需在实际场景检查后确定，正式场景可否增改相机仍需核实。若优先处理正式输入许可，应先取得官方相机配置或允许的输入说明。本轮在修改场景前交由用户审核。

后续用户已同意按有相机的方案继续，并明确告知相机使用已获许可；已添加相机、收到实际 RGB/内参，并完成 10 块砖的候选检测。该审核点已处理，后续按 [RGB 接入实测](rabokit-vision.md) 的最新配置与实际传输结果继续。
