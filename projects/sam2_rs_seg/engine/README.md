# Lightning引擎接口（设计，未实现）

未来SegmentationModule继承lightning.pytorch.LightningModule，封装普通PyTorch模型，
负责各step、loss、configure_optimizers及val/test TorchMetrics状态。
SegmentationDataModule继承LightningDataModule，负责已有数据与DataLoader；
prepare_data不自动下载大型数据集。

训练入口显式组合模型、DataModule、ModelCheckpoint、logger及Trainer。
源码版本通过后续Callback写入日志/实验目录，不创建另一套训练主循环。
当前没有上述类、训练入口、权重或可运行配置。
