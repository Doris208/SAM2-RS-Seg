# 05 后续实现与验收顺序

任务 01 只建立源码、Git、文档和少量 CPU 工具。下列均为未来阶段，不能视为已实现。
后续在当前本地项目继续开发，不新建另一套工程，不把远端作为唯一源码位置。

## 阶段 1：远端依赖验证

按 03 文档部署并检查 PyTorch、MMCV 编译/算子、MMSeg 导入和 SAM2 导入。
只有构建与检查日志证明确认可用后，才能把候选环境升级为“实测通过”；
保存完整 pip freeze、驱动、nvcc、GCC 和源码 manifest。
如出现第三方兼容性错误，先保留日志分析，不私改第三方核心代码。

## 阶段 2：Potsdam RGB 数据接口

拿到真实数据后核验 RGB 与官方 GT 定义，在 `datasets/` 开发显式转换脚本，
在 `projects/sam2_rs_seg/datasets/` 注册必要接口。
先固定 tile 划分，再切块；每类计数、未知像元、对齐、nearest 缩放、ignore 排除、
训练/验证集合隔离和单通道 PNG 全部验收。将 schema 和划分列表提交 Git。

## 阶段 3：Hiera + 简单语义头

在本地实现 `SAM2HieraBackbone`，不复制 Hiera；使用 MODELS Registry。
首先核验 Tiny 四阶段特征、通道/步长/梯度、预训练加载率与 RGB 归一化。
然后通过原生 EncoderDecoder + FPNHead 建立最小 Potsdam 配置，
不添加 LoRA、MoE、跨注意力、光谱先验或半监督部件。
原始 Prompt Encoder / Mask Decoder 扩展命名空间保留，第一版不启用。

验收：有损失的 forward/backward、编码器可训练参数检查、非方形输入、
ignore 255 与 padding、CUDA显存、小 batch 可运行、状态保存/恢复。

## 阶段 4：训练、验证、测试与导出闭环

复用原生 Runner，CheckpointHook 显式 save_best='mIoU'、rule='greater'，
明确评估 interval、保留数量、随机种子与数据划分，最优模型由 val 选择。
使用合成混淆矩阵校验 mIoU/IoU/mF1/OA，注意零预测类别 F1 和空类 NaN。
结构化导出逐类结果，避免只有终端表格。
端到端小样本 smoke 后核对 ID PNG 的 mode、尺寸、值域与 RGB palette 图。
输出目录独立记录版本、完整配置与日志；测试集不参与超参数选择。

## 阶段 5：整图推理与扩展数据

先验证 MMSeg slide 推理，再实现窗口读取/拼接、重叠 logits 融合、
边缘覆盖、nodata、地理参考与内存限制，参考 GeoSeg 而不引入其 Lightning 主框架。
然后按各自已核验协议扩展 Vaihingen IRRG 与 LoveDA2021。
原始提示/掩膜分支的后续接入以独立 adapter 为入口，仍复用该工程的训练与数据系统。

每阶段只在有证据时标记通过；任务01结束后停止，等待进一步指令。
