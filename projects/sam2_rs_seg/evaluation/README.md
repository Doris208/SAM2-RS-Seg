# TorchMetrics评价契约（设计，未实现）

val/test各自持有TorchMetrics MulticlassConfusionMatrix，ignore_index=255。
在全epoch全rank累积并同步后的矩阵上计算每类IoU/F1、mIoU、mF1及OA。
完全空类分母为0时标NaN并从宏平均排除；有GT无预测时F1=0。
没有有效GT的评价集报告不可评估，不更新best。内部0–1，aAcc是OA别名。

ModelCheckpoint监控val_mIoU、mode=max，仅验证集选择best。
计划导出每类结果、类别平均范围、分母和ID/RGB PNG；实际指标模块尚未实现。
环境脚本可选的合成指标检查不是完整训练评价验收。
