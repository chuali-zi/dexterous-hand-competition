# 离线闭环证据契约 v2

本页规定本项目内部数据与策略边界，不是官方SDK/API说明。当前只运行离线合成RGB，官方平台实测暂搁。本契约的强校验、恢复和回放准备不需要登录或实测。

## 观测包

`ObservationGate.accept(raw, now_s)`的顶层字段必须恰为：`schema_version=2`、`producer`、`calibration_id`、`time_s`、`frames`、`robot`。未知/缺字段、重复JSON键、NaN/Infinity、超大整数、错误类型均拒绝。数字单位为米/秒；原始RGB在进程内是`bytes`，不是JSON字符串。

| 字段 | 校验与意义 |
|---|---|
| `producer` / `calibration_id` | 与预先审查配置一致；生产者标签不是来源真实性认证 |
| `time_s` | 有限、非负、严格递增、不得超前于接收时钟；最大年龄默认0.25秒 |
| `frames` | 完整且无重复的已配置相机列表；当前单块策略使用两路针孔RGB |
| 每帧 | 恰含`camera_id/frame_id/sequence/captured_s/width/height/encoding/rgb`；光学frame与相机ID对应；序号与捕获时间递增；rgb8字节长度精确匹配整数尺寸 |
| `robot` | 恰含`frame_id='base_link'/tip_position_m/producer/captured_s/sequence`；位置有限且位于配置边界；独立校验生产者、捕获时间和序号 |
| 对齐 | 相机与机器人样本时间偏差默认≤0.03秒；完整包通过后才原子更新序号状态 |

机器人位置必须是可核实的本体反馈；禁止以命令目标回显替代。手/O6反馈不可填零假造。实际话题、时间转换、标定和数据来路需后续根据真实模板核实；字段校验无法识别生产者伪装的物体真值。

所有相机外参为base→optical的正交旋转行与相机原点；光学轴x右、y下、z前。内参fx/fy以像素计。相机ID采用安全内部标识，防止日志路径混入斜杠；实际ROS话题由未来适配器另做审查映射。相机尺寸/矩阵维数先检查，分数尺寸、非正交旋转、无效焦距均拒绝。

## 配置与策略不变量

[完整默认配置](../configs/closed-loop.json)可通过`validate-config`校验，再用`run/batch --config`加载。未知键拒绝，数组归一化为不可变tuple，关系约束包括：

- 供应区/砌筑区及目标位于工作区；本策略是**桌面单块放置**，目标中心高度须与桌面块中心高度一致，不假称已支持垒高层。
- 安全高度≥水平搬运最低高度，且在工作区内；低高度任何超过数值舍入范围的水平位移被拒绝。
- 稳定窗口≥3秒；稳定、到位和高度容差不能大到掩盖抬升失败；手分离距离大于抓持距离，安全高度须提供对应余量。
- 最大跨帧间隔不得超过样本年龄/稳定窗口；连续稳定以最老相机/本体捕获时间累计，不用调用时钟经过3秒替代图像。

阶段门采用内部`PHASE_ACTIONS`映射：只有规定阶段/动作组合可提交。携带/下降/释放要求此前视觉确认持有；关闭前要求新鲜目标邻近手，释放前要求测量到位，撤手前必须有释放尝试。搬运期间失视预算为**零个缺失样本**：首个失视/持有跟随失效即请求停止，避免直接下降/释放。正常开始遮挡允许原地等新图像；稳定阶段短暂失视或大帧间隔会重开窗口，不盲目重抓已经放好的块。

每个move的complete都核验末端测量到位；complete仍不是抓到/释放成功。进入及保持稳定须同时满足实验目标、桌面高度、手块分离和位置变化阈值。所有阈值是离线工程配置，不是官方几何判分。

submit/observe/status返回后及成功前重查总期限；动作期限从提交请求之前起算，先检查超时再接受complete，同时检查独立墙钟。无法返回/取消的未知SDK调用仍不受Python轮询强制中断；适配器必须真正有界，这里没有声称已实现该SDK能力。终止后同一个Controller不能重跑/再提交动作。

## 记录与回放

v2 `manifest.json`包含完整配置、相机、噪声参数、代码身份和复现命令；另存`config.json`，复现命令引用保存配置，防止自定义参数丢失。每条事件绑定实际manifest文件散列，并串联普通内容哈希；这些不是签名。

事件顶层为`index/kind/time_s/payload/previous_sha256/manifest_sha256/sha256`；kind与payload字段按代码中的白名单校验。必须有唯一最后result、对应停止事件；未知/重复/终止后事件、重用动作ID、重叠动作、无提交状态、晚到complete和无证据失败被拒绝。

回放不导入Controller，独立重建动作生命周期和成功语义：从close/lift顺序、抬升前后RGB高度与手块关系重建抓持，再核release、连续稳定样本和末端反馈。`grasp_verified`标签不能充当证明；与图像矛盾的标签也拒绝。返回标志分别表示视觉重算、控制记录一致性、manifest绑定、机器人时间检查；`sensor_authenticity_verified=false`、`physical_action_execution_verified=false`保持实际限度。

```bash
python -m hand_competition.closed_loop validate-config configs/closed-loop.json
python -m hand_competition.closed_loop run --config configs/closed-loop.json --output runs/my-v2
python -m hand_competition.closed_loop replay runs/my-v2
python -m hand_competition.closed_loop replay runs/my-v2 --expected-runtime-sha256 <可信外部源码指纹>
```

可信源码指纹须来自调用者审查的提交/源码，不能仅从待验证manifest自己取值就称认证。v1只用于明确的历史日志回放，不进入实时观测门；旧记录没有机器人时间/生产者及manifest绑定，返回标志会体现缺口。

原始被拒绝包不落盘；未知后端异常只记录类别/类型。日志写入失败会停止且结果标`trace_complete=false`，回放不认可缺损记录。`stop_request_succeeded`仅为调用成功，物理停止仍未验证。闭环结果仍不自动转换为官方或人工标签分数。
