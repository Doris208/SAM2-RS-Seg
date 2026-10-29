# 01 Lightning主工程架构

状态：任务02接口规划，未实现模型训练闭环。唯一训练调度为lightning.pytorch.Trainer。

## 职责与数据流

~~~text
Dataset + nearest label transforms
  -> LightningDataModule (train/val/test split isolation)
  -> LightningModule
       -> SAM2HieraBackbone (official Hiera trunk wrapper)
       -> SimpleMultiScaleDecoder (native PyTorch, K-channel logits)
       -> CrossEntropyLoss(ignore_index=255)
       -> TorchMetrics confusion state (val/test separate)
  -> ModelCheckpoint(monitor="val_mIoU", mode="max")
  -> ID PNG / separate RGB PNG / per-class JSON and CSV
~~~

以上类均为未来接口名称，目前目录只有命名空间与说明。
不依赖第三方注册系统；配置入口显式组合原生对象，不导入GeoSeg训练框架。

## 模型契约

models/backbones/未来适配torch.nn.Module：
forward(images[B,3,H,W])返回按高到低分辨率排列的特征tuple。
优先官方SAM2.1 Hiera Tiny，trunk来自固定源码和官方YAML子树，不复制或修改核心实现。
预计通道(96,192,384,768)、步长(4,8,16,32)，必须forward核验。
Hiera.channel_list为逆序元数据，实际forward按阶段正序；不能混用。
优先pad到32倍数，保留原尺寸、padding、scale_factor，单独验证非方形输入。

官方完整ImageEncoder还包含FpnNeck，scalp=1会丢弃最低分辨率层。
当前基线选择trunk，并不删除未来完整image_encoder路径。
权重核验model包装、image_encoder.trunk.前缀、shape与missing/unexpected keys，
输出加载覆盖率，禁止静默空加载。模型训练不可套官方predictor的inference_mode。

models/decode_heads/未来SimpleMultiScaleDecoder：
逐层投影、对齐到高分辨率、融合和K通道分类，恢复输入空间尺寸。
不含提示生成；对连续特征/logits可双线性插值，对类别Mask仅nearest。
模型forward输出[B,K,H,W]logits；semantic_predict另外执行argmax并恢复原尺寸。

图像读取明确RGB/BGR、uint8/float与取值范围。归一化一次：
若RGB输入0–1，候选使用SAM2 mean=[.485,.456,.406]/std=[.229,.224,.225]；
如果读取0–255则先缩放或等价换算。Vaihingen IRRG独立标明通道语义。
实际归一化需与官方预训练输入核对后验收。

## Lightning管理契约

engine/未来实现：

- SegmentationModule(LightningModule)：forward、training_step、validation_step、
  test_step、predict_step、configure_optimizers；不另写手动训练主循环。
- SegmentationDataModule(LightningDataModule)：只负责已有数据的setup和DataLoader；
  prepare_data不得自动下载完整数据集。train/val/test划分文件及随机种子显式记录。
- Trainer配置：设备、precision、分布式策略、日志、callback、resume路径均由配置显式提供；
  max_epochs、学习率和batch等尚未选定，不编造最优参数。
- 默认不自动使用DDP或AMP。后续先单卡FP32小样本验收，再验证混合精度。
- 标签long[B,H,W]，ignore=255。全ignore批次应显式跳过/处理，避免空分母NaN损失。
- forward、loss和metric分开，使train/eval与推理恢复尺寸一致。

## 指标与Checkpoint

evaluation/未来用TorchMetrics MulticlassConfusionMatrix(num_classes=K, ignore_index=255)
维护整个epoch的累积状态；验证与测试实例分离，每个epoch reset。
分布式通过TorchMetrics同步统计量后计算，不对各batch或各rank的mIoU作简单平均。
不要把已同步的全局指标再次作为各rank局部均值重复sync_dist平均。

从累计混淆矩阵计算IoU_c、F1_c、OA和宏平均。K=6/7及类别顺序保持数据协议。
IoU分母=0或F1分母=0的完全空类为NaN并排除宏平均；有GT但无预测的类F1=0。
这是计划采用的显式公式，未来用合成矩阵测试，不能无条件用TorchMetrics默认macro替代。
整个评价集无有效像元时报告不可评估，不更新best。

统一内部范围0–1；日志val_mIoU、val_mF1、val_OA，aAcc是OA的兼容名称，不是mAcc。
显示百分比仅在导出时显式转换。保留逐类IoU/F1和参与平均的类别数量。
ModelCheckpoint(monitor="val_mIoU", mode="max", save_top_k=1, save_last=True)为未来方案，
仅验证集选择best，test不能调参。

## 提示分支与输出契约

models/prompt_adapters/保留原始Prompt Encoder/Mask Decoder：

- encode_image(images)可保留官方image_encoder返回值和位置编码。
- encode_prompts(points, boxes, masks, image_meta)生成官方提示嵌入。
- decode_prompt_masks(image_embeddings, prompts, high_res_features, image_meta)
  返回prompt mask logits与质量分数。
- image_meta记录原始/输入尺寸、缩放、padding和提示坐标变换。
- 提示mask[N,H,W]及quality不是语义K通道logits，保持独立API，不强行互换。
- 自动提示仅保留扩展空间，本轮不实现；无LoRA/MoE/光谱/伪标签/半监督模块。

ID PNG为原尺寸uint8单通道，RGB PNG按palette另存。
滑窗/整图拼接未来融合重叠logits后argmax，处理边缘、覆盖计数、nodata与地理参考。
GeoSeg仅供工程参考，不能顺带改变类别、ignore、划分或指标协议。
