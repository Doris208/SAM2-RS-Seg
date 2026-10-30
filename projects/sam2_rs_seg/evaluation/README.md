# TorchMetrics评价契约（03B）

`SegmentationMetrics`直接使用TorchMetrics MulticlassConfusionMatrix累计完整数据集统计量。
行是GT、列是预测，只有GT=ignore_index的像元被忽略；类别0正常参与。
train/val/test持有独立状态，每次epoch开始reset，再由累计矩阵计算IoU/F1、mIoU/mF1/OA。
默认空类标NaN并从宏平均排除；可显式配置zero。预测中出现但GT缺失的类仍计入宏平均。
JSON中的NaN导出为null，内部指标范围0–1。整套评价无有效GT时明确报错，不选择best。
val/test损失按有效像元数加权，最终指标不取batch指标平均。

03B只支持单设备。多卡损失归约与真实影像结果导出留待后续，不以当前接口声称DDP已验证。
