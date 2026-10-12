# 本次验证记录

日期：2026-10-12。执行环境 `/workspace/dexterous-hand-competition`，Python **3.12.14**。README声明3.10+来自使用的语言特性；**本次只运行了3.12.14**，没有做跨版本矩阵。

## 已执行

| 检查 | 结果/证据 |
|---|---|
| 初始checkout | 分支`work`，起点`5420a93e4ba79facff7fe392dcca720bff6514a6`，只有原始README，工作树干净；远端默认分支`main`同一提交 |
| 工作区规则 | 未发现适用`AGENTS.md`或本地`.agents/skills`文件；没有执行第三方仓库的agent/安装指令 |
| 原始项目读取 | 8仓库，25个成功读取的固定SHA原始文件，下载文件SHA-256与源锁逐项一致；含README/许可及选定接口 |
| 论文读取 | 4篇原始arXiv HTML方法、接口、硬件/实验与限制相关章节；版本与定位见sources |
| `python -m unittest discover -s tests -v` | **18/18通过**。覆盖系数/满分/错色/倒塌最终状态、3秒与释放、只接触不旋入、物料上限、重复ID/JSON键、错误类型/NaN/负数/超时、日志追加、CLI成功及失败 |
| `python -m hand_competition examples/brick-final-state.json` | 合成标注：原始35、系数1.0、结果35；`official_judge=false` |
| `python -m hand_competition examples/assembly-final-state.json` | 合成标注：原始50、系数0.5、结果25；`official_judge=false` |
| JSON与运行时依赖 | 配置/示例/源锁可解析；运行时绝对导入仅argparse/hashlib/json/math/pathlib，均为标准库 |
| 文档内部链接 | 初检仅缺尚未写入的本验证文件；补齐后复检所有相对目标存在（不声称外链都可访问） |
| 差异检查 | `git diff --check`及暂存区同类检查；不包含数据集、权重、第三方源码或凭据 |

测试验证的是离线算术、输入边界与记录语义，**不是视觉精度、机器人成功率或比赛平台验收**。35/25这两个数是测试夹具期望值，不能放进成绩宣传。

## 未执行/仍受限

- 官方规则PDF、演练PDF、通知和SDK页面本次无法读取；使用父任务已核实摘录。未获得全文散列和逐条单页定位。详细错误见sources。
- 未读取腾讯问题表答复，未参加答疑；正式场景、组别、资格、名额、作品锁定时间、输入/算力/费用等仍待确认。
- 未复制比赛应用或运行`agent_system`，未导入ROS/SDK，未验证命名空间、TF、`/clock`、相机标定、运动、碰撞或稳定性。
- 未下载模型权重、训练、安装/执行第三方脚本、购买资源或调用付费平台；未运行真机、报名、联系主办方、提交比赛分享链接或改变仓库权限。
- 未实现几何判分、实际观测白名单隔离、示教采集或模型适配；设计文档不等于已实现功能。
- 父任务另行安排规则只读审计，本提交不声称已经收到或整合其最终结果。

## Git交付核验口径

保留初始提交，直接在既有`work`工作分支创建初始化提交。推送前fetch并检查origin/main没有他人更新；采用普通非强制`HEAD:main`推送。如果远端变更则停止覆盖，先保留并整合他人内容。推送后以`git ls-remote origin refs/heads/main`和本地`git rev-parse HEAD`比对。最终SHA和核验结果记录在任务最终回报；此文件不伪造包含自身的提交哈希。
