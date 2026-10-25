# 开源灵巧手项目本地试跑（2026-10-25）

用户授权下载候选项目并实际测试。本轮未打开 Rabokit、未启动云端测试或训练。8 个仓库以浅克隆和稀疏检出保存在 `runs/open-source-75/repos/`；按需下载源码、O6 USD 和预训练权重，没有下载大型训练集。第三方源码、权重、原始输出不进入项目提交。

本机为 Windows、Intel Arc 显卡，Python 3.12.10、PyTorch 2.13.0+cpu，CUDA 不可用。已有 MuJoCo 3.3.7 位于 `runs/mujoco-runtime`。额外小依赖安装到 `runs/open-source-75/deps`，未修改全局 Python。已有 WSL 只有 Python 3.13，没有 torch、Pinocchio、nlopt，不为本轮另装整套 Linux 训练环境。

## 逐项结果

| 项目与检出版本 | 实际执行 | 结果与适用边界 |
|---|---|---|
| [O6 Isaac Sim grasp](https://github.com/MuYCev/linkerhand-o6-isaacsim-grasp)，`1cc0306` | 作者原始轨迹验证；读取原始 USD；移植闭指、撤托、保持和释放到本地 MuJoCo，跑 50 g、100 g 和不闭指对照 | 作者日志通过；本地跨引擎复现两种闭指条件均掉落。不闭指对照也掉落。不是作者原始 Isaac 仿真重跑，也不是桌面抓取成功。 |
| [Linker IsaacLab patch](https://github.com/ZzXwANT/linkerhand_isaaclab_patch)，`8a31655` | 上游 `tests/test_linker_mimic.py` | 1 项关节联动测试通过。完整任务依赖 IsaacLab，尚未跑物理策略；README 的 O6 任务无现成权重。 |
| [GraspXL](https://github.com/zdchan/GraspXL)，`53ab112` | Allegro 预训练 actor 严格加载，CPU 前向 32 个合成归一化观测 | 输出 `32×22`，全部有限。只验证网络运行；没有 RaiSim 物理闭环，不能当作成功抓取。 |
| [DexGrasp-Zero](https://github.com/YliangWu/DexGrasp-Zero-code)，`e80d713` | 多手图网络 actor 严格加载，CPU 前向 32 个合成图 | 原始网络头输出 `32×51`，全部有限。合成图绕过了 URDF 观测构建和真实动作映射；不是 O6 策略。完整执行还缺 RaiSim 和接触/冲量等观测接口。 |
| [DRO-Grasp](https://github.com/zhenyuwei2003/DRO-Grasp)，`07590bd` | v1.0 `model_3robots.pth` 严格加载；O6 与 5 cm 立方体各 512 点、三个随机种子的距离预测及多点测距重建 | CPU 可运行，约 4.5–5.4 s/次；几何距离重建 RMSE 23.3–24.4 mm。不包含关节 IK 或物理抓取，不宜直接上平台。 |
| [dex-retargeting](https://github.com/dexsuite/dex-retargeting)，`3f56141` | 实际导入 `RobotWrapper`，解析并尝试安装二进制依赖 | 缺少 `pinocchio`；当前 Windows/Python 3.12 下 `pin` 无可用匹配 wheel，源码依赖开始构建后停止，避免把时间花在整套 C++ 工具链上。未完成重定向求解。 |
| [DexGraspNet2](https://github.com/PKU-EPIC/DexGraspNet2)，`26ecd76` | 实际导入 `src.network.graspness_sample`，检查运行栈 | 在 `pytorch3d` 导入处失败；MinkowskiEngine、IsaacGym/CUDA 栈亦未就绪。未运行预训练推理或物理验证，不能判断算法失败。 |
| [Linker fixed grasp](https://github.com/linker-bot/linkerhand-fixed-grasp)，`b036bfb` | 本地 CMake 配置；兼容旧 CMake 声明后继续检查 | 缺少 ROS `catkin`，配置失败。源工程针对 RealMan + L10 真机，未运行硬件通信节点，不是现成 O6 仿真解法。 |

## O6 物理复现具体结果

上游示例固定手根，将 50 mm / 50 g 砖预先放在掌前托台上。0–1 s 稳定、1–3 s 闭指、3–4 s 撤托、4–7 s 保持、7–8 s 张手。采用上游平滑插值和手形：四指 MCP 1.3 rad，DIP 1.157 rad；拇指 yaw 1.2、pitch 0.5、IP 0.93 rad。

上游 `verify.py` 对公开日志给出通过：保持窗口 2.066666 s，最大位移约 0.002256 mm，释放后下降约 288 mm。这只是已发布日志的复核。

本地使用已有右手 URDF 凸网格模型；12 个刚体零位原点与原始 USD 最大差约 `1.08e-8 m`，排除了整体原点放错这种解释，但没有证明碰撞壳、惯量、驱动和求解器等价。

| 本地条件 | 闭指结束后的表现 | 4–6.1 s 保持窗口 |
|---|---|---|
| 50 g；kp=3、kv=0.12、最大驱动力矩 0.5；关闭手内自碰撞 | 有手指接触，撤托时滑落 | 砖中心约 24.97 mm，即落地；失败 |
| 100 g；现有驱动 kp=20、kv=1、最大力矩 1；保留当前自碰撞排除关系 | 未能保持 | 砖中心约 24.97 mm，即落地；失败 |
| 50 g，不闭指 | 撤托后掉落 | 负对照符合预期 |

两组都保留本地接触参数 μ=1、solref≈0.01113 s、关节被动阻尼及 MuJoCo 求解器；50 g 组不是完整复制 PhysX 静/动摩擦、接触偏移和惯量。100 g 是现有模型的假设质量，不是本轮重新测得的平台真值。手根固定，没有完整机械臂和邻砖。该结果说明不能直接照搬这个配方，不足以证明上游结果造假或此手形不可能成功。

评估要求高处保持、无托台接触、持续手指接触、低漂移，以及张手后掉落。落地静止即使漂移很小也不算通过。新增 4 个针对这一判定的单元测试，涵盖正常释放、落地、仍有支撑和缺失观测。

## 复现入口与记录

DRO 使用本地 O6 零位网格表面采样和最远点降采样，砖块为中心在原点的 50 mm 立方体表面点云。网络对三个种子均输出有限的 `1×512×512` 距离矩阵；通过上游 `multilateration` 重建点云后，重建距离与网络预测距离的 RMSE 分别为 24.44、24.09、23.33 mm。输入已知正确距离的对照中，位置 RMSE 为 `3.95e-9 m`、距离 RMSE 为 `5.99e-9 m`。这提示当前 O6 输入下的预测几何一致性不足；不是上游支持手型的失败率，也没有证明所有 O6 输入都会失败。尚未进行关节 IK、刚体约束和碰撞检验，不把预测点云当作可执行抓姿。

```powershell
python scripts/test_open_source_o6_hold.py
python scripts/check_open_source_networks.py
python scripts/check_dro_o6_prediction.py --checkpoint runs/open-source-75/dro-model_3robots.pth
python -m unittest tests.test_open_source_hold_evaluation
```

- 物理试验脚本：[test_open_source_o6_hold.py](../scripts/test_open_source_o6_hold.py)。模型依赖本机已存在的 `runs/bimanual-contact-68/local-calibrated-contact.xml` 和手部网格；不是脱离资产即可运行的发布包。
- 网络试跑：[check_open_source_networks.py](../scripts/check_open_source_networks.py)。需要先检出 GraspXL 的 Allegro 权重及 DexGrasp-Zero 的 multi_hand 权重。
- DRO 点云试验：[check_dro_o6_prediction.py](../scripts/check_dro_o6_prediction.py)，用 `--checkpoint` 显式指定完整网络 state dict。
- 原始输出：`runs/open-source-75/o6-physics/`、`neural-smoke.json`、`o6-geometry-check.json`、`runtime-blockers.json`、`fixed-build.log`、`dro-prediction/`。

## 当前取舍

目前没有通过我们物理抓稳标准的开源方案。保留 O6 示例作同款手的接触模型对照，保留 GraspXL 和 DRO-Grasp 作候选动作/几何来源。DexGrasp-Zero 先解决本体与接触观测映射，不能把不存在的冲量输入补零后宣称多模态闭环可用。dex-retargeting 是运动学工具，单独跑通也不提供抓力与稳定性。

下一步应依据候选的实际几何和物理结果决定是否移植；没有本地持握候选前，不为验证导入或模型加载而启用 Rabokit。完整 Isaac/RaiSim 执行、DRO 的关节求解和真实 O6 的抓取闭环仍未完成。
