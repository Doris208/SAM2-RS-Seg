# 01 架构与 SAM2-Hiera 接口设计

状态：设计方案，未实现或运行模型。唯一训练/验证/测试调度为 MMSegmentation 1.x + MMEngine。

## 第一阶段数据流

```text
RGB image + single-channel GT IDs
  -> native dataset / transforms / SegDataPreProcessor
  -> EncoderDecoder
       backbone: SAM2HieraBackbone (future MODELS registration)
       decode_head: native FPNHead + multiscale resize/concatenation
  -> K-channel logits -> cross entropy(ignore_index=255)
  -> predict: SegDataSample.pred_sem_seg
  -> ID PNG + independent RGB palette visualization
```

先采用 Hiera Tiny 与原生简单多尺度头。首版不需要额外自定义复杂解码器；
可用 FPNHead 的多尺度分支，需设置与 Hiera 一致的 `in_channels`、`feature_strides`，
不把 `FPNHead` 误当作必须先接官方 SAM2 FpnNeck。
图像编码器使用预训练参数，语义头随机初始化；冻结/解冻策略以后在配置中明确。

## 图像编码器适配契约

位置：`projects/sam2_rs_seg/models/backbones/`，未来继承 `mmengine.model.BaseModule`，
使用 `@mmseg.registry.MODELS.register_module()`。根包最终显式导入各注册模块，
配置通过 `custom_imports` 加载。本轮 __init__.py 不导入尚未实现的类。

建议初始化参数：
`sam2_config`、`checkpoint`、`out_indices=(0,1,2,3)`、`feature_source='trunk'`、
`frozen_stages`、`init_cfg`。严格校验配置和权重属于同一代 SAM2 / SAM2.1 与同一型号。
本轮不下载 checkpoint，不硬编码绝对路径。

输入 `Tensor[B,3,H,W]`；输出高分辨率到低分辨率的 `tuple[Tensor[B,Ci,Hi,Wi],...]`，
不返回预测 Mask，不使用 predictor 的 `no_grad/inference_mode` 包装。
trunk 路径直接调用官方 `sam2.modeling.backbones.hieradet.Hiera.forward`；
通过官方 Hydra 配置中的 image_encoder.trunk 子树实例化，而非复制 Hiera 源码。
避免仅为 trunk 构建完整视频模型、提示分支或执行 `torch.compile`。

源码核查得到 Tiny / Small 四阶段输出通道预计为 `(96,192,384,768)`，
步长预计为 `(4,8,16,32)`，必须以实际 forward 核验。
特别注意 `Hiera.channel_list` 是逆序通道元数据，但 `forward` 输出按阶段正序；
不能直接将 channel_list 当作正序输出通道。
输入优先 pad 到 32 的倍数，支持 512/1024 与非方形尺寸须分别测试，不强制把大图拉伸为方形。

权重适配：官方 checkpoint 常有顶层 `model` 与
`image_encoder.trunk.` 前缀，提取后检查每个 key/shape；
记录 missing/unexpected keys，拒绝意外空加载。不得用 `strict=False` 静默掩盖不匹配。
第三方参数只在适配器中映射，不更改原文件。

## 官方 image_encoder 与未来提示接口

官方 `ImageEncoder` 由 Hiera trunk + FpnNeck 组成，返回
`vision_features`、`vision_pos_enc`、`backbone_fpn`。
当前 SAM2.1 配置有 `scalp: 1`，会舍弃最低分辨率特征，
所以完整 image_encoder 输出不能未经核验就假设为四级 Hiera 原始特征。
第一阶段以 trunk 特征为清晰基线；未来如选 FPN 特征，通道、层数和预训练加载应单独配置。

独立保留 `models/prompt_adapters/`：

- `encode_image`：可保留完整官方 image_encoder 输出及位置编码。
- `encode_prompts(points, boxes, masks, image_meta)`：包装原始 SAM2 Prompt Encoder。
- `decode_prompt_masks(image_embeddings, prompt_embeddings, high_res_features, image_meta)`：
  包装原始 SAM2 Mask Decoder，返回 mask logits 与质量估计。
- `image_meta` 必须带 original_shape、input_shape、padding、scale_factor，
  保证提示坐标与输出坐标一致。多类别语义 logits 的契约与提示 mask 输出分开。

目前这些是接口设计，没有空实现伪装为可运行功能。首版配置不实例化提示模块，
以后以独立 adapter/segmentor 注册，仍用同一套 Runner、数据接口和源码工程。

## 预处理、损失与评价

只有 SegDataPreProcessor 做一次归一化。SAM2 常见 mean/std
为 RGB 的 `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`；
MMSeg 0–255 输入对应乘 255 的 mean/std，需根据真实读取范围核实。
mmcv 默认读 BGR 时显式 `bgr_to_rgb=True`；若读取器已输出 RGB 则关闭，防止二次换序。
Vaihingen 为 NIR/R/G，单独设置通道语义，不能把它称作自然 RGB。
标签缩放用 nearest，图像与标签共享几何变换，标签 padding=255。

拟用 `IoUMetric(iou_metrics=['mIoU','mFscore'], beta=1, ignore_index=255)`：
各类 IoU/F1 在表中输出，`mFscore` 即 mF1，`aAcc` 即 OA（不是 mAcc）。
MMSeg 返回值为百分数。后续结构化导出每类结果，清楚记录缺失类的 NaN 处理规则。
默认以验证集 `mIoU`、rule='greater' 保存最佳权重，测试集不参与模型选择。

## 推理与拼接扩展

小图：原生 inferencer / inference_model，恢复原尺寸后导出 uint8 单通道类别 ID，
另按 palette 导出 RGB 图，两个文件不可混用。
未来先验证原生 `test_cfg.mode='slide'` 的 crop/stride，
再对 GeoTIFF 做窗口读取、元数据保留与分块写回。
重叠区融合 logits/probabilities，最后 argmax；禁止对类别 ID 做双线性融合。
处理最后一个窗口、padding、nodata 和每像元覆盖计数，评价时避免重叠像元重复计数。

源码依据：固定子模块中的 `hieradet.py`、`image_encoder.py`、
`sam2/configs/sam2.1/sam2.1_hiera_t.yaml` 和 MMSeg `iou_metric.py`。
