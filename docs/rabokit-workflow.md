# 本地开发、Rabokit 云端测试

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

## 本次实际发现和范围

云运行环境为 Python 3.12.3、ROS2 Jazzy。已通过云终端读取 SDK 方法并实际接收左右臂关节角。

`LbotDualArm(mode='sim')` 的部件属性为 `left_arm/right_arm/left_hand/right_hand`。3.0.0 在此环境返回客户端代理，不能把导出类的全部 Node 方法当作实例方法；例如直接对 `left_arm` 调 `get_topic_names_and_types()` 会失败。话题枚举应使用独立的 `rclpy` 节点。所有连接探针都应在 `finally` 中关闭 SDK，以免异常退出遗留后台线程。

本次实际装配副本的 ROS 图没有发现本场景的 Image、CompressedImage 或 CameraInfo 话题，场景编辑器里的双臂也显示传感器数为 0。后续要核实正式场景的相机配置/允许的图像接入方式。界面能看到 3D 画面，不代表控制程序能订阅相机。物体 pose 话题的存在不等于比赛允许用真值做决策；本次探针只用机器人关节反馈。

本次腕关节探针已实际通过：4秒内收到左臂198条、右臂197条 JointState；左腕目标增加0.08 rad，实际增加0.07669 rad，随后返回至起始位置附近（第7关节约0.00241 rad）。另行测试左手 close/open，均收到接受回执，ROS手指关节在闭合/张开后出现对应变化；这只是空手动作，未测试夹持物体。SDK关闭时输出过 `cannot use Destroyable because destruction was requested` 后台清理警告，进程返回0；不把它写成无警告运行。

检查结果：新脚本语法及云端运动/返回、只读入口执行通过；164个相对文档链接通过。云端原有回归单次180秒上限前完成96项，随后单独运行最后的观测模块7项通过（4项重叠），覆盖99个不同测试。单次全套命令未完整退出成功，不写成单次99项通过。Windows探索性全套检查遇到默认临时目录权限问题；改用仓库内临时目录后还发现原有运行记录路径使用反斜杠导致的断言失败，已停止本地全套尝试，本轮不改动离线控制源码。

本页的接通测试不代表抓取、装配或垒砖得分，不把离线合成场景结果当作正式仿真成绩。
