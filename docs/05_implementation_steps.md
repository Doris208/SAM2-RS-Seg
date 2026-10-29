# 05 迁移后的实现顺序

当前只完成任务02底座迁移，不启动以下开发或训练。

1. **实际容器环境验证**：优先并行智算云PyTorch25.03，盘点现有Python/torch/
   torchvision/CUDA，再评估Lightning与TorchMetrics。保留原容器，审阅dry-run，
   导入、反传、合成指标通过后形成容器专用约束/freeze。不要把候选标作已通过。
2. **数据小闭环**：保持现有label_contract，按原始tile固定Potsdam RGB
   train/val/test，再实现显式转换、原生Dataset与nearest变换。
   核验单通道ID、RGB、ignore255、对齐、nodata和每类计数。
3. **原生模型小闭环**：官方Hiera Tiny适配、多尺度通道/步长/顺序实测、
   权重覆盖率、单次归一化、简单多尺度解码头输出K通道logits。
   先合成输入forward/loss/backward，核验可训练参数和显存。
4. **Lightning小闭环**：实现LightningModule、LightningDataModule与统一Trainer入口；
   选择配置解析方式，拒绝status=design_only_not_runnable的计划文件。
   先单卡小样本FP32，训练/验证/测试状态分离。
5. **指标/Checkpoint/导出**：TorchMetrics累积混淆状态，测试ignore、空类、
   有GT无预测、OA与宏平均；val_mIoU保存best，test只最终评估。
   导出逐类JSON/CSV、原尺寸ID PNG与独立RGB PNG。加入源码版本记录Callback。
6. **扩展**：验证AMP、resume、DDP状态同步，再做滑窗logits融合、GeoTIFF拼接、
   Vaihingen IRRG与LoveDA2021。原始Prompt Encoder/Mask Decoder经独立接口接入，
   不改变当前语义头输出契约。

每一步在当前本地主工作区开发、检查并提交后再部署。
数据、权重和输出保持独立。不得默认添加创新模块或自动push。
