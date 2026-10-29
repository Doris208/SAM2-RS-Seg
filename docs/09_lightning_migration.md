# 09 任务02：Lightning底座迁移记录

日期2026-10-29。仅工程迁移准备，不代表模型可运行或GPU环境通过。

## 分支与移除过程

开始时工作区干净，当前分支为doris；它与main均指向
5e154790798fab248819a6dd946cf677bfe7ea45。
检查全部子模块正常后，从main创建refactor/lightning-backend。
原main、doris及其历史保留，origin配置未改。用户随后追加授权：
无需手动审查，由Codex检查后直接提交并同步当前迁移分支，不自动合并main。

移除前确认third_party/mmsegmentation：
无暂存/未暂存/未跟踪/ignored本地文件，HEAD在官方origin/main历史中，
不存在仅在本地分支保存的额外提交。
已使用git submodule deinit -- third_party/mmsegmentation，
再git rm -- third_party/mmsegmentation。
当前工作目录和索引gitlink已移除，.gitmodules及.git/config活动登记已清理。
.git/modules/third_party/mmsegmentation历史元数据保留，不强删缓存或历史。

当前完整保留：

| 子模块 | 固定Commit |
|---|---|
| SAM2 | 2b90b9f5ceec907a1c18123530e92e794ad901a4 |
| GeoSeg | 9453fe48209c4626b29e35e61bab93b61212c4b1 |

两者checkout均干净，未重新克隆、升级、安装或修改核心源码。
third_party/versions.json仅保留这两个版本记录。

## 实际修改范围

- 运行依赖和安装计划切换为PyTorch、lightning.pytorch、TorchMetrics、官方SAM2。
  旧框架的注册/训练/结果对象接口已从当前自有代码和活动设计中移除。
- 环境脚本默认只读元数据，runtime/GPU显式可选，不再固定Python3.10、
  torch2.5.1/cu121；候选下限检查不是兼容性证明。
- 两子模块部署脚本保留版本、哈希、文件导出/校验、Windows安全链接和长路径处理。
- 标签代码datasets/label_contract.py及原tests/test_initialization.py逐字保持原状，
  六/七类别、ignore255、nearest和评价划分约定不因迁移改变。
- 新增Potsdam Hiera Tiny JSON计划文件、engine/evaluation接口说明，
  明确模型/数据/Trainer/指标类尚未实现。
- 删除旧configs/_base_三个占位文件，以及已有真实计划文件的potsdam/.gitkeep。
- README与01–06活动文档更新；07原初始化记录加历史说明而保留原结果。
  08及源码清单按当前两子模块实际文件重新生成。

完整逐文件修改/删除/新增列表保存于outputs/task02/change_inventory.json与change_list.txt；
git status和暂存/未暂存diff是检查依据；git rm暂存删除属于规范移除过程。
按追加授权直接完成迁移提交与推送，不再等待用户在SourceTree手工提交。

## 实际检查结果

- 13项CPU unittest全部通过：原8项保留，新增无训练依赖盘点、
  显式runtime/GPU开关、候选版本不硬锁、计划/标签/子模块一致性、
  两子模块导出链接物化与SHA往返。
- 17个自有Python文件AST解析通过，14个自有命名空间导入通过，
  JSON计划可解析。
- 本地环境盘点成功；缺少torch/Lightning/TorchMetrics等如实报告，
  没有安装任何训练依赖或执行GPU测试。
- 无活动旧框架运行导入；历史记录中仍保留旧框架名称是有意保留事实，
  不能要求全仓库文本完全无旧名而改写历史。
- 原main/doris Hash未变，保留子模块Hash/内部status正常，
  标签代码和原测试与main的diff为空。
- 迁移未提交阶段，真正的--require-clean导出曾返回1并报告未提交修改，未创建输出目录；
  这是历史上的dirty保护检查，不等于干净提交的完整导出验收。

新增导出单测对Git状态使用模拟干净fixture，不创建Git提交；
实际文件复制、链接解析和哈希验证是真实执行。
它不代替迁移提交后的正式--require-clean端到端验收；实际命令结果以提交后执行为准。

## 提交后验收和后续GPU工作

1. 按追加授权完成迁移提交，再正式验证两子模块清单和源码导出，
   核验完整文件数量、两个gitlink、dirty=false及SHA-256。
2. 正常推送refactor/lightning-backend并核验远端Hash，保留main原历史。
3. 并行智算云PyTorch25.03实际镜像盘点、Lightning/TorchMetrics候选依赖解析，
   CPU/CUDA导入/反传/合成指标及Trainer smoke。
4. Hiera实际forward/权重/梯度、真实Dataset、语义头、Lightning训练闭环、
   完整指标/Checkpoint/ID与RGB PNG、AMP/DDP、滑窗和大图拼接。

候选Lightning2.5.5/TorchMetrics1.7.4仅基于声明调查，
不固定平台Python/CUDA/torch/Lightning，不把NGC原版内容当成平台镜像实测。
后续顺序见05_implementation_steps.md，GPU计划见03_gpu_setup_plan.md。

任务01的deploy/与outputs/快照保留为历史，不用于新的Lightning部署。
本轮按追加授权提交并同步迁移分支，不进行训练、远端安装、权重下载或main合并。
