# 03B补充：五层架构审查与必要完善

## 1. 审查结论

本轮保留Meta官方SAM2 + PyTorch Lightning Trainer路线，没有重建Registry、Trainer或DataModule。
Models、Datasets、Evaluation的核心边界已具备；立即补充独立损失层，以及epoch训练/调度配置。
推理目前可直接调用普通nn.Module，没有发现必须新增推理包装器的耦合问题。

本地仅进行源码审查、语法编译、YAML/配置检查、CLI及差异检查。
**本轮新增loss、CUDA梯度、epoch训练/恢复及03A/03B运行回归均未在本地执行，正式结果待云端RTX 4090验收。**
不将静态通过或已存在的历史结果记为本轮运行PASS。

审查起点：doris分支，HEAD `1e64c75f1824d2c2914a42fce302213eb7d17b57`。
原03B文件已暂存，保留该暂存状态；本轮只修改工作区，没有执行git add/commit/push。
修改前备份：`outputs/task03b_architecture_backup_20261030_122536/`。

## 2. 五层结果

| 层面 | 已实现与解耦情况 | 实际问题/硬编码判断 | 本次处理与后续范围 |
|---|---|---|---|
| Models | 独立ImageEncoder、三尺度语义头、Segmentor；普通nn.Module返回[B,K,H,W]；独立模型/头工厂检查通道、类别、尺寸 | 工厂仅正式注册SAM2，并声明3通道/32倍数/256通道特征，这些是该模型约束，不是Lightning内部硬编码；已有显式替换头入口 | 保持全部模型文件不变。新模型可注册自身构造器/输入约束并复用Trainer；其他模型内部头适配随模型接入再做 |
| Datasets | 纯PyTorch合成Dataset与Lightning DataModule分离；image/mask和可选sample_id；独立固定split、metadata、参数化通道/类别/ignore；支持注入外部Dataset且不静默回退合成数据 | 没有Potsdam固定路径/类别名，没有模型计算。当前不做几何变换，因此没有图像/mask不同步问题；尚无真实读取器/增强Pipeline | 保持不变。Dataset独立定义classes/palette/label_map；未来original_size、spatial_metadata和异构metadata的collate由数据接入阶段处理 |
| Losses | 原有CE位于LightningModule，未污染SAM2模型 | CE被构造和评价代码直接指定；配置只接受CE，不能只换配置使用Dice | 新增独立CE/Dice模块与轻量构建函数、独立标签工具；Lightning仅调用统一loss接口 |
| Training | 自动优化、参数组覆盖、梯度累积/裁剪、CSV、checkpoint完整恢复均已有；不依赖SAM2内部结构 | 配置强制max_steps>0且调度interval固定step，只有max_epochs参数并不等于真正支持epoch-only配置 | 开放有限epoch或step控制，明确scheduler.interval；默认CE/step不变。没有引入Fabric或另一套训练循环 |
| Evaluation / Inference | 独立TorchMetrics累计矩阵；行GT列预测；0有效、255忽略；缺失类策略明确；三阶段独立reset；模型可不经Trainer前向 | 标签校验原放在评价模块，供loss复用会引入不必要依赖；没有正式大图/GeoTIFF能力 | 将同一校验函数移至通用utils并保持旧导入兼容，指标算法不变。不新增推理壳；分块、滑窗、拼接、GeoTIFF和地理参考恢复留到后续 |

没有在本轮新增真实Dataset、图像增强、正式大图推理或研究损失。
当前无几何变换；未来任何空间操作必须同时作用于image/mask，mask采用nearest，标签映射显式且不默认reduce_zero_label。

## 3. 独立损失契约

入口：`projects.sam2_rs_seg.losses.segmentation.build_loss(config, num_classes)`。
返回普通nn.Module，不导入Lightning、TorchMetrics、SAM2或Trainer内部接口。

- `forward(logits, target)`返回标量训练loss。logits为原始[B,K,H,W]浮点值，target为[B,H,W] long。
- 类别0参与监督；ignore_index可配置，必须是与0..K-1不重合的整数，默认255。
- 非法标签、错误形状/类型、非有限logits明确报错，不对非法ID截断或偏移。
- `evaluation_terms(logits,target)`返回可累计的 `(numerator, normalizer)`；全忽略批次返回可反传的零numerator和零normalizer。
- standalone forward的全忽略输入明确报错。Lightning仍在训练forward前拒绝全忽略批次；整套评价无有效GT也继续报错。
- 以后添加普通研究loss，在loss层实现/登记构造器及其配置校验，遵守forward/evaluation_terms契约；无需修改SAM2或Trainer。

### CrossEntropyLoss

训练调用原生 `torch.nn.functional.cross_entropy(..., reduction='mean')`；
评价返回原生 `reduction='sum'` 和有效像元数，保留03B有效像元加权的val/test损失计算。
新增包装器没有参数/缓冲区，不增加模型checkpoint的Tensor state键。

### 多类别DiceLoss

选用标准一阶soft Dice，不使用二值sigmoid Dice或argmax。
softmax从logits在loss内部计算；float16/bfloat16时统计升为float32，double输入保持double。
先把ignore位置的标签临时替换为0以安全one-hot，再将**概率和one-hot标签同时乘有效像元掩膜**。
对每张图n、类别c，在有效像元上：

`D[n,c] = (2*Σ(p*y) + smooth) / max(Σp + Σy + smooth, eps)`。

先平均该图选中的类别loss `1-D`，再平均具有监督像元的图像。
默认 `smooth=0, eps=1e-7, absent_class='exclude'`：排除该图GT中未出现的类别，类别0出现时绝不排除。
可显式include全部类别；若GT与概率质量都为0，规定该类Dice=1。
默认exclude仍通过softmax类别竞争对错误概率分配产生梯度，但不把GT缺失类作为单独监督项。

Dice评价返回每图loss之和与有监督图像数，完整数据集val/test loss为**图像平均Dice loss**；
CE仍为**有效像元平均CE**。两者都可跨batch累计，不把batch Dice随意平均成数据集指标。
无论选择何种loss，mIoU/mF1/OA继续独立从原有全数据集混淆矩阵计算。

人工数值案例：两个有效像元GT=[0,1]，三类概率分别为[.8,.1,.1]与[.2,.7,.1]。
exclude应得到loss=19/90，include为64/135。测试据此核对数值，而不是以另一份相同实现当作答案。

### 配置示例

原03B默认不变：

```yaml
loss:
  name: CrossEntropyLoss
  ignore_index: 255
```

切换Dice只需修改loss配置：

```yaml
loss:
  name: DiceLoss
  ignore_index: 255
  smooth: 0.0
  eps: 1.0e-7
  absent_class: exclude
```

## 4. Epoch / Step控制与兼容性

`max_steps=4,max_epochs=-1`保留原按optimizer更新步数控制；
`max_steps=-1,max_epochs=2`支持epoch-only；两者均为正时由Lightning按先达到的上限停止。
两者都为-1、0、布尔或非整数上限拒绝，避免无界训练或含糊配置。

调度配置明确支持：

```yaml
scheduler:
  name: PolynomialLR
  interval: epoch  # 或step；省略时仍按旧版step处理
  total_iters: 2   # epoch时单位是epoch；step时单位是optimizer更新
  power: 1.0
```

LightningModule将interval原样交给官方调度器管理；没有自己调用第二套scheduler循环。
step调度下梯度累积微批次不单独更新LR；epoch调度下，同一epoch的所有optimizer更新使用相同调度档位。
TrainingAudit记录实际epoch/global_step/LR，以相应单位核对序列。

原配置样例仅补充显式 `interval: step`，默认数值行为不变。
resume_signature将旧checkpoint缺少的interval规范为step，旧CE/step完整恢复不会因新增默认字段被误拒绝。
改变loss或把调度从step改为epoch仍视为训练协议变化，不冒充相同协议的完整resume。

原03B Smoke专门验证step续训，明确拒绝epoch调度配置；epoch训练/恢复由本轮小模型CUDA案例验证。
原生/Lightning单步对照同步接收实际loss配置，避免训练使用Dice却只对照CE的误导性结果；默认仍为CE。

## 5. 文件变更与必要性

以下路径相对于 `D:/Research/SAM2-RS-Seg`。

| 文件 | 原因 |
|---|---|
| 新增 `projects/sam2_rs_seg/losses/__init__.py`、`losses/config.py`、`losses/segmentation.py` | 独立CE/Dice构造与数值实现，不引入新依赖或大型Registry |
| 新增 `projects/sam2_rs_seg/utils/labels.py` | 同一标签校验供loss、评价和训练使用，不让独立loss依赖TorchMetrics |
| 修改 `projects/sam2_rs_seg/evaluation/segmentation_metrics.py` | 从通用工具导入并重新导出原validate_masks；原指标算法不变 |
| 修改 `projects/sam2_rs_seg/engine/lightning_module.py` | 注入loss配置、调用通用评价terms、调度interval可配置；不识别SAM2结构 |
| 修改 `projects/sam2_rs_seg/engine/experiment.py` | 将既有配置loss传给LightningModule，没有重建Trainer或DataModule |
| 修改 `projects/sam2_rs_seg/engine/optim.py`、`utils/config.py` | 明确epoch/step周期与有限训练控制，兼容旧step checkpoint签名 |
| 修改 `configs/smoke/task03b.yaml` | 仅显式写出interval=step；保留原CE、类别、数据、训练步数和评价设置 |
| 修改 `scripts/task03b_checks.py`、`scripts/smoke_03b.py` | 诊断记录调度单位，单步对照使用真实配置loss，Smoke限定其原step验收范围 |
| 新增 `tests/test_task03b_architecture.py` | 15项配置、数值/掩膜/梯度、解耦、旧CE状态兼容、epoch恢复及Dice一致性案例 |
| 修改 `README.md`、`projects/sam2_rs_seg/engine/README.md`、新增本文 | 当前能力和范围、使用协议、来源与云端回归命令 |

Models、Registry、Dataset、DataModule、03A检查/测试、原03B测试均保持原内容。
没有更改SAM2/GeoSeg子模块版本或源码，也没有安装/升级依赖或自动提交代码。

## 6. 开源参考与复用

- [PyTorch CrossEntropyLoss 2.7](https://docs.pytorch.org/docs/2.7/generated/torch.nn.CrossEntropyLoss.html)：直接调用成熟CE实现，保持ignore和mean/sum语义。
- [SMP v0.5.0 Dice](https://github.com/qubvel-org/segmentation_models.pytorch/blob/v0.5.0/segmentation_models_pytorch/losses/dice.py)：实际阅读logits转多类别概率、有效像元掩膜与one-hot、缺失GT类别处理。仅参考算法，不安装SMP、不复制完整源码；本项目明确定义逐图归约，区别于该文件的batch联合Dice及缺失类归约方式。
- [MMSeg v1.2.2 Dice](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/mmseg/models/losses/dice_loss.py)：实际阅读Dice两种分母与激活/one-hot接口。没有照搬其ignore处理或二值默认，不引入MMEngine/MMCV。
- 本地Meta SAM2 `build_sam.py`/`modeling/backbones/image_encoder.py`及03A封装：保留既有完整ImageEncoder/官方Neck/三尺度连接，未修改核心代码。
- 本地GeoSeg `geoseg/datasets/potsdam_dataset.py`、`transform.py`：检查image/mask联合几何变换与mask nearest；未移植读取器、旧palette或默认ignore值。
- [Lightning/Fabric 2.6.6](https://github.com/Lightning-AI/pytorch-lightning/blob/2.6.6/src/lightning/fabric/fabric.py)：阅读setup、setup_dataloaders和backward接口；仅确认普通PyTorch组件可复用，没有引入Fabric。

过程使用了PyTorch Lightning和verification-before-completion技能：前者核对公开接口，后者要求验证证据与声明一致。
按Lightning技能要求，软件过程参考补列：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*.
[DOI](https://doi.org/10.48550/arXiv.2609.00065)。已核对arXiv当前记录为v2；这不是Dice算法来源或运行验收证据。

## 7. 云端回归命令

同步本轮新增文件和修改后的工作区源码；仅上传源码/配置，不重复上传权重或本地outputs。

```bash
conda activate doris
cd /root/projects/SAM2-RS-Seg
export PYTHONPATH="$PWD/third_party/sam2${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -c "import torch, lightning.pytorch as pl; print(torch.__version__, torch.version.cuda, pl.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -m unittest tests.test_task03a tests.test_task03b tests.test_task03b_architecture -v
python scripts/smoke_03a.py --device cuda \
  --checkpoint /root/shared-nvme/weights/sam2.1_hiera_tiny.pt \
  --height 256 --width 256 --num-classes 6 --seed 42 \
  --json outputs/task03b_architecture/task03a_regression.json
python scripts/smoke_03b.py --device cuda \
  --checkpoint /root/shared-nvme/weights/sam2.1_hiera_tiny.pt \
  --config configs/smoke/task03b.yaml \
  --output-dir outputs/task03b_architecture/task03b_regression
```

15项新增案例包括：5项配置/结构检查、6项独立loss CUDA数值/梯度检查、4项CUDA Lightning状态兼容/接入/epoch完整恢复/Dice单步对照。
CE与官方PyTorch数值/梯度对齐；Dice人工数值、屏蔽区域不变与零梯度、类别0、非法标签、全忽略和分批归约均有断言。
默认CE的03A/03B整链回归另行执行，不能仅凭新增小模型案例认定官方SAM2通过。
请核对实际运行无失败及无意外skip；如发生skip，不计为PASS。

本地静态检查记录：`outputs/task03b_architecture_static_checks.json`。
云端默认CE完整报告：`outputs/task03b_architecture/task03b_regression/smoke_report.json`。
本轮没有Dice实测loss/梯度值、epoch恢复结果或新的GPU PASS记录，等待人工云端核验后停止于03B补充。
