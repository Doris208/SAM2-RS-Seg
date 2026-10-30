# SAM2-RS-Seg

基于 **原生 PyTorch + PyTorch Lightning + TorchMetrics + Meta 官方 SAM2/SAM2.1** 的遥感全监督多类别语义分割科研工程。

唯一主工作区：D:\Research\SAM2-RS-Seg。GitHub：https://github.com/Doris208/SAM2-RS-Seg
任务02迁移分支：refactor/lightning-backend；按用户追加授权，检查后直接提交并同步该分支。

**任务03A的三尺度模型及完整梯度已由用户确认通过云端验证。03B新增可配置Lightning训练、评价、恢复与一致性检查代码，03B正式GPU验收待云端执行；没有真实遥感数据训练结果。**

## 技术分工

- 原生PyTorch：Hiera适配、多尺度语义解码、损失和优化器。
- PyTorch Lightning：唯一训练管理框架，统一使用lightning.pytorch命名空间。
- TorchMetrics：累积分类统计量、计算与导出mIoU/逐类IoU/mF1/OA。
- 官方SAM2：固定源码提供Hiera，保留原始Prompt Encoder/Mask Decoder独立扩展。
- GeoSeg：只参考数据、裁片和整图推理，不安装其训练依赖或复制其主框架。

任务02取消MMSegmentation、MMCV、MMEngine运行依赖；原任务01事实保留在历史记录中。

## 模型与数据目标

三通道影像 → 官方SAM2.1 ImageEncoder（Hiera Tiny + FpnNeck）→ 三尺度backbone_fpn对齐拼接 → 轻量PyTorch语义头 → K通道logits。
完整人工GT全监督，Potsdam RGB优先，后续Vaihingen IRRG与LoveDA2021 RGB。
ignore_index=255；Mask最近邻缩放；按原始tile划分后再裁块。
未来实现验证集mIoU保存最佳模型、类别ID PNG、RGB图、滑窗logits融合与整图拼接。
提示掩膜分支与多类别语义预测保持独立，不在本阶段加入创新模块。

## 工程入口

| 路径 | 当前职责 |
|---|---|
| configs/ | smoke/task03b.yaml驱动合成训练；.plan.json仍仅设计 |
| projects/sam2_rs_seg/models/ | SAM2ImageEncoder、SemanticSegmentationHead、SAM2SemanticSegmentor；prompt_adapters预留 |
| projects/sam2_rs_seg/datasets/ | 合成Dataset和通用DataModule；真实数据读取留待03C |
| projects/sam2_rs_seg/engine/ | LightningModule、优化器/调度器、Trainer实验组合 |
| projects/sam2_rs_seg/evaluation/ | 累计混淆矩阵、统一分割指标及状态管理 |
| datasets/label_contract.py | 已实现的通用标签契约，保持原编号不变 |
| scripts/ | 环境盘点、两子模块版本/哈希清单、源码导出与校验 |
| tests/ | 无GPU基础测试 |
| data/、weights/、outputs/、deploy/ | Git忽略的运行文件位置 |

## 保留的两个子模块

| 源码 | 固定Commit |
|---|---|
| third_party/sam2 | 2b90b9f5ceec907a1c18123530e92e794ad901a4 |
| third_party/GeoSeg | 9453fe48209c4626b29e35e61bab93b61212c4b1 |

实际完整源码已经存在，不重新克隆或升级。第三方checkout未修改。
移除的旧子模块内部历史元数据保留在.git/modules中，不是当前受控源码，不上传SFTP。
从未来已提交的迁移版本重新克隆后才执行git submodule update --init --recursive。

## 本地检查

~~~powershell
python scripts/check_environment.py --json outputs/lightning_environment.json
python -m unittest discover -s tests -v
~~~

默认环境盘点只读取包元数据，不导入深度学习框架；缺少依赖/无GPU时仍可完成盘点。
--runtime显式启用CPU导入/反传/合成指标；--require-gpu显式启用CUDA，未在本机验证。
--candidate仅检查候选最低声明，不再锁死Python/CUDA/PyTorch/Lightning精确版本。

任务03A模型接口、源码参考、测试记录及云端命令见 [任务03A报告](docs/task03a.md)。
`configs/task03a_model.json` 仅供随机初始化模型构建；`.plan.json` 仍不能启动训练。
GPU验证必须显式传入官方checkpoint；脚本不下载权重，不自动回退到随机初始化。
三尺度修正版Smoke逐一检查Hiera四阶段/全部blocks和Neck四个投影分支；旧版单尺度PASS不等同于完整多尺度验收。

03B说明与云端命令见[任务03B报告](docs/task03b.md)。正式运行全部安排在云端RTX 4090；
本地只完成代码、静态检查和验收脚本，不要求安装或升级训练环境。
`scripts/smoke_03b.py`验证短训练、完整恢复、重复验证、测试及原生PyTorch/Lightning单步一致性。

五层架构审查及必要补充见[03B架构审查](docs/task03b_architecture_audit.md)：
独立CE/多类别Dice损失、epoch/step训练控制和明确的调度周期；原SAM2三尺度结构保持不变。

03B指令提供的云端环境为PyTorch2.7.1+cu128、Lightning2.6.6；实际版本写入运行报告。
不自动升级或降级已有PyTorch、CUDA、torchvision、Lightning或SAM2。

## 文档与版本管理

- [03A模型](docs/task03a.md)、[03B训练框架](docs/task03b.md)

SourceTree继续管理同一路径。main及其历史保留；origin沿用现有GitHub配置。
git rm规范移除旧子模块；提交前检查差异、测试和子模块状态，不要求用户手动审查。
**正式--require-clean源码导出基于干净的迁移提交验收；旧deploy/快照仅作历史。**
GeoSeg保留GPL-3.0许可证；未来如复制实现须考虑其分发义务。
