# Lightning训练引擎（03B）

`SegmentationLightningModule`持有普通nn.Module，管理CE损失、各阶段步骤及指标生命周期。
`optim.py`直接构建AdamW和按optimizer step更新的PolynomialLR，检查参数分组覆盖。
补充审查后，损失从`../losses/`构建；PolynomialLR可显式按step或epoch更新，缺省仍为step。
训练可通过正的max_steps或max_epochs控制；原03B默认step配置与CE评价协议保持不变。
`experiment.py`组合DataModule、CSVLogger、ModelCheckpoint和Trainer，保存生效配置与环境。
完整恢复使用`Trainer.fit(..., ckpt_path=...)`；fit结束后通过公开API补存最终完整last.ckpt。
Dataset/DataModule位于现有`../datasets/`命名空间，没有另建重复data目录。

没有自定义训练循环、真实数据读取器或DDP实现。
运行命令、恢复语义和待云端执行的检查见[03B报告](../../../docs/task03b.md)。
