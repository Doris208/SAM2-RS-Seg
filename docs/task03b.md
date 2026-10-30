# 任务03B：通用Lightning遥感语义分割训练基础

## 1. 实施概况与当前状态

已实现：安全YAML实验配置、轻量模型/已有头构建、合成Dataset、通用DataModule、
LightningModule、累计混淆矩阵评价、AdamW、按optimizer step的PolynomialLR、
ModelCheckpoint、完整恢复与数值一致性验收脚本。

**本地只做源码阅读、编写与静态检查，未执行03B训练/单元运行时测试/GPU Smoke。**
所有正式运行由用户在已有RTX 4090容器执行。不能把静态通过记为训练通过。
03A云端12项测试及深层梯度成功是本次用户指令提供的事实；没有改动03A模型计算结构。

当前分支doris；起点提交 `1e64c75f1824d2c2914a42fce302213eb7d17b57`（03A已完成）。
实施前工作区干净，两个固定子模块干净，未切换版本、未commit/push、未安装或升级依赖。
原README与两份模块说明备份于 `outputs/task03b_backup_20261030_101113/`。

## 2. 实际公开检索、阅读与复用

| 来源 | 实际阅读文件/文档 | 用途 |
|---|---|---|
| 本地固定Meta SAM2，Apache-2.0 | `sam2/build_sam.py`、`modeling/backbones/image_encoder.py`、`hieradet.py` | 确认严格加载、完整ImageEncoder、三尺度连接；通过03A类复用，没有复制或修改第三方代码 |
| [Lightning 2.6.6源码](https://github.com/Lightning-AI/pytorch-lightning/tree/2.6.6/src/lightning/pytorch)，Apache-2.0 | `core/module.py`、`core/datamodule.py`、`callbacks/model_checkpoint.py`、`trainer/trainer.py`、`loops/fit_loop.py`、`loops/training_epoch_loop.py` | 直接调用Module/DataModule/Trainer/ModelCheckpoint/CSVLogger；核对自动优化、恢复先于train_start、step调度跳过累积微批次、监控回调与epoch-end顺序 |
| Lightning官方文档 | [Module](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)、[DataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)、[Checkpoint](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html) | 核对公共API；stable当次显示2.7开发文档，兼容性以另行读取的2.6.6源码为主。2.6.6对应checkpoint文档URL无法访问，未声称读过该版本文档 |
| [MMSegmentation v1.2.2](https://github.com/open-mmlab/mmsegmentation/tree/v1.2.2/mmseg)，Apache-2.0 | `models/segmentors/encoder_decoder.py`、`datasets/basesegdataset.py`、`evaluation/metrics/iou_metric.py`、`registry/registry.py` | 只参考模块分离、metainfo/标签、全数据集统计后计算指标、按名称构建；没有移植MMEngine/MMCV代码或引入运行依赖 |
| 本地固定GeoSeg，GPL-3.0 | `train_supervision.py`、`geoseg/datasets/potsdam_dataset.py`、`geoseg/models/UNetFormer.py` | 只阅读模型/训练封装及数据接口，未复制实现；不采用旧pytorch_lightning命名空间、调色板或类别排除规则 |
| [TorchMetrics v1.7.4源码](https://github.com/Lightning-AI/torchmetrics/blob/v1.7.4/src/torchmetrics/classification/confusion_matrix.py)，Apache-2.0 | MulticlassConfusionMatrix的state/update/compute；[IoU文档](https://lightning.ai/docs/torchmetrics/stable/classification/jaccard_index.html) | 直接累计混淆矩阵；由项目小函数明确处理缺失类NaN策略，不依赖不同版本Jaccard宏平均的隐含默认行为 |
| PyTorch 2.7官方文档 | [PolynomialLR](https://docs.pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.PolynomialLR.html)、[sdpa_kernel](https://docs.pytorch.org/docs/2.7/generated/torch.nn.attention.sdpa_kernel.html) | 直接调用优化器/调度器；一致性对照两条路径固定相同math SDPA后端，避免后端选择差异 |

公开检索用于明确未指定的调度次序、恢复钩子、缺失类别、确定性对照和Last Checkpoint处理。
自行实现的代码仅为配置/接口连接、标签校验、指标归约、记录及验收诊断，没有重新实现训练循环。
没有新增依赖；沿用已有torch、lightning、torchmetrics、PyYAML和SAM2依赖。

## 3. 文件变更

路径相对于 `D:/Research/SAM2-RS-Seg`。

| 文件 | 职责 |
|---|---|
| `projects/sam2_rs_seg/models/registry.py` | 明确的模型/头映射、构造参数与通道/尺寸/类别检查；默认模型不替换03A头 |
| `projects/sam2_rs_seg/datasets/synthetic_dataset.py` | float32影像、long标签、固定且独立的train/val/test样本与metadata |
| `projects/sam2_rs_seg/datasets/datamodule.py` | setup和DataLoader；可注入外部Dataset，保留现有datasets目录而不新建重复data目录 |
| `projects/sam2_rs_seg/engine/lightning_module.py` | 普通模型包装、CE、自动优化接口、三个独立指标状态、epoch归约、checkpoint配置协议 |
| `projects/sam2_rs_seg/engine/optim.py` | AdamW、可选前缀参数组、完整参数覆盖检查、PolynomialLR或无调度 |
| `projects/sam2_rs_seg/engine/experiment.py` | Trainer、Callback、CSVLogger组合，配置/环境记录、最终完整last保存 |
| `projects/sam2_rs_seg/evaluation/segmentation_metrics.py` | TorchMetrics累计矩阵、IoU/F1/OA与缺失类策略、标签合法性检查 |
| `projects/sam2_rs_seg/utils/config.py`、`utils/__init__.py` | safe_load与明确协议检查，不eval、不动态加载任意模块 |
| `configs/smoke/task03b.yaml` | 短训练的全部模型/数据/损失/优化器/调度器/Trainer/评价/checkpoint/seed参数 |
| `scripts/train_03b.py` | 配置训练入口；区分官方初始化与完整resume，支持可选best模型测试 |
| `scripts/smoke_03b.py` | CUDA必需的短训练、保存、精确状态恢复、验证/测试和原生/Lightning一致性验收 |
| `scripts/task03b_checks.py` | 公共钩子观测、LR序列核对、完整状态比较、单步误差计算 |
| `tests/test_task03b.py` | 人工数值案例、模型/头可替换性、数据/标签/优化器、CUDA累积与恢复/一致性测试 |
| `README.md`、`engine/README.md`、`evaluation/README.md`、本文 | 能力与边界、参考、命令与验收状态 |

**没有修改03A的backbone、semantic_head、segmentor、03A检查/测试或官方配置文件。**
没有新增正式对比模型或第二套分割头；Dummy模型/头只存在测试中。

## 4. 架构与协议

`YAML → build_model → ordinary nn.Module → SegmentationLightningModule`
与 `Dataset → SegmentationDataModule → Trainer` 组合，由独立评价器和ModelCheckpoint输出结果。

- 现仅正式注册SAM2SemanticSegmentor，metadata声明3输入通道、尺寸至少32且为32倍数。
- 类别数与data一致，已有头按三个256通道尺度验证；最终实际logits形状再次在Lightning步骤检查。
- 默认头配置null，保留03A原头原参数。显式头配置仅从HEADS构建，测试Dummy不进入生产默认注册表。
- 模型只接收 `batch['image']`，标签来自 `batch['mask']`；sample_id等附加字段不影响模型。
- GT为long，0合法，ignore_index默认255，可配置且不得与有效ID重合。
- 全忽略训练批次在forward前报错，避免NaN和无监督优化。全忽略评价批次可累计零统计量；
  整个评价集没有有效像元时明确报错，不能生成虚假的best指标。
- 每个样本使用 `seed + 3*index + split_id` 的独立CPU Generator，split_id=0/1/2。
  样本内容不受访问顺序/worker影响，不执行图像文件IO、类别偏移或真实增强。
- train/val/test独立矩阵，在各自epoch开始reset；矩阵行GT、列预测。
- IoU=TP/(GT+Pred-TP)，F1=2TP/(GT+Pred)，OA=ΣTP/有效像元总数。
- absent_class=exclude默认：分母为0记NaN、宏平均排除；zero可显式计0。
  预测出现而GT缺失的类分母非零，不能被排除。JSON将NaN写为null。
- val/test损失按有效像元总数归约，指标从累计矩阵计算，不平均batch mIoU。
- 03B只支持单设备；没有以接口预留代替DDP实现/验证。

### 优化器和调度器

AdamW使用官方实现；前缀参数组可指定lr_multiplier，其余参数进入默认组。
检查空组、重叠、重复和遗漏；只包含requires_grad=True的模型参数。

PolynomialLR不含warmup。设u为已经完成的optimizer更新次数，T=total_iters，p=power：
下一次更新使用 `lr(u)=base_lr * max(0,1-u/T)^p`。
先执行optimizer.step，再由Lightning调用scheduler.step；`interval='step', frequency=1`。
梯度累积读取多批数据只产生一次optimizer更新/global_step增长和调度更新。
诊断在 `on_before_optimizer_step` 读取实际使用的LR，并与公式比较，防止更新时机偏移。

### Checkpoint与完整恢复

官方.pt仅用于首次模型初始化，由03A完整严格加载。Lightning.ckpt则用于完整训练恢复。
恢复时显式构建同结构未初始化模型，交给Trainer严格装载完整训练状态；这是明确的resume路径，
不属于预训练加载失败后的随机回退，也不要求旧官方初始化文件仍存在。

ModelCheckpoint监控val_mIoU/max，保存top-1和last；CSVLogger记录主要标量。
另在fit结束后调用公开 `Trainer.save_checkpoint(last.ckpt)`，保证Last覆盖最终optimizer step，
避免max_steps在验证之间停止，以及部分版本仅在top-k写入时更新last的差异。
没有自定义保存格式或第二套训练循环。

Checkpoint保存完整experiment_config；完整resume禁止改变模型、数据、loss、优化器、调度器、评价与seed协议。
可延长trainer.max_steps；PolynomialLR总预算必须保持不变，不在恢复时偷偷重算。
Smoke在完整epoch边界恢复，核对global_step、epoch、全部模型Tensor、完整AdamW状态（含moments/step/LR）
和调度器state；随后核对AdamW计数、调度计数及连续LR序列。
这验证完整状态续训，不额外声称任意中途批次/worker随机状态与不停机训练逐位一致。

## 5. 自动化检查状态

| 检查 | 本地状态 | 云端运行入口 |
|---|---|---|
| Python语法、配置解析、CLI help、差异检查 | 已执行并通过（仅静态） | 无需GPU |
| 模型名称/参数/通道/尺寸/类别、头替换 | 未运行 | tests.test_task03b |
| 合成数据、固定划分、long标签、255和类别0 | 未运行 | tests.test_task03b |
| 人工混淆矩阵、缺失类、跨batch累计、独立reset | 未运行 | tests.test_task03b |
| 非法/全忽略训练批次、logits尺寸 | 未运行 | tests.test_task03b |
| 优化器完整覆盖、重叠组、PolyLR序列 | 未运行 | tests.test_task03b |
| CUDA Tiny模型梯度累积、保存/恢复、重复验证 | 未运行 | tests.test_task03b的CUDA集成类 |
| CUDA Tiny随机层原生/Lightning一致性 | 未运行 | tests.test_task03b的CUDA集成类 |
| 官方预训练SAM2完整03B闭环与数值一致性 | 未运行 | scripts/smoke_03b.py |

没有把跳过、静态通过或03A旧结果记为03B运行PASS；本地没有安装/升级训练依赖。

## 6. 云端RTX 4090验收

先同步本轮全部新增源码和配置；无需上传本地outputs、.git或重复权重。

```bash
conda activate doris
cd /root/projects/SAM2-RS-Seg
export PYTHONPATH="$PWD/third_party/sam2${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -c "import torch, lightning.pytorch as pl, torchmetrics, sam2; print(torch.__version__, torch.version.cuda, pl.__version__, torchmetrics.__version__); print(torch.cuda.get_device_name(0)); print(sam2.__file__)"
python -m unittest tests.test_task03b -v
python scripts/smoke_03b.py \
  --device cuda \
  --checkpoint /root/shared-nvme/weights/sam2.1_hiera_tiny.pt \
  --config configs/smoke/task03b.yaml \
  --output-dir outputs/task03b
```

默认流程：2次optimizer更新→完整保存→恢复到4次→重复验证→测试→原生/Lightning单步对照。
每次Smoke保留独立 `outputs/task03b/run_<UTC时间>_<随机后缀>/`，不会删除旧实验。
最新报告：`outputs/task03b/smoke_report.json`；同一报告也留在本次run目录。
该目录另含requested_config.json、effective_config.yaml、effective_resume.yaml、环境JSON、
logs/version_*/metrics.csv、checkpoints/best-*.ckpt和last.ckpt。
报告只有全部环节通过才为PASS；失败时记录stage、异常与traceback，并返回退出码1。
环境、正式checkpoint加载、训练损失、阶段梯度、参数更新、指标、保存路径、恢复状态、LR和一致性均记录。
合成数据mIoU只能说明接口可计算，不能解释为真实遥感分割性能。

### 单独训练入口（仍然只有合成数据）

```bash
python scripts/train_03b.py --config configs/smoke/task03b.yaml \
  --checkpoint /root/shared-nvme/weights/sam2.1_hiera_tiny.pt \
  --output-dir outputs/task03b_manual --test
```

对尚未跑完原计划预算的中断训练，使用相同配置恢复：

```bash
python scripts/train_03b.py --config configs/smoke/task03b.yaml \
  --resume outputs/task03b_manual/checkpoints/last.ckpt \
  --output-dir outputs/task03b_manual --test
```

`--checkpoint`与`--resume`互斥；后者读取原初始化描述但不重新加载官方.pt。
Smoke内部先以max_steps=2截断，再以max_steps=4恢复，total_iters始终为4。
默认四步预算已完成后不应继续超出total_iters；PolynomialLR到预算末尾后LR为0。
延长截断测试的max_steps需仍在预先声明的调度总预算内，改变调度预算属于新训练策略，不能冒充相同协议恢复。

## 7. 数值一致性检查

对同一模型快照保存全部参数/BN缓冲；原生执行一步AdamW后，严格恢复初始模型，使用全新AdamW
交给Lightning自动优化执行同一批数据。两次均train/float32、同seed、无累积/裁剪/调度/AMP。
输入内容完全相同；随机种子在Lightning实际forward前设置，避免DataLoader/Trainer初始化消耗RNG。
两条路径均使用math SDPA，TF32在Smoke禁用；这是对照的后端控制，不修改03A模型代码。

固定容差 `atol=1e-6, rtol=1e-5`。报告初始logits、loss、**所有**可训练参数梯度和更新后参数的
最大绝对误差、最大相对误差及逐参数结果；相对误差分母下限1e-12仅用于报告。
通过判断使用torch.allclose原始容差，不以报告的分母下限放宽判定，也不会遇错自动扩大容差。
梯度为None即失败，不能把两边都没有梯度当成一致性通过。

当前误差数值：**未运行，尚无实测值**。必须由云端生成，不能用预测值代替。

## 8. Checkpoint恢复当前结果

当前恢复步数、AdamW moments/step、调度器恢复结果：**未运行**。
实现会在恢复后第一次更新前做完整逐Tensor精确比较，再检查global_step从2继续到4、
AdamW计数继续到4、PolynomialLR.last_epoch=4及LR无重置；这些是验收条件，不是已通过结论。

## 9. 后续扩展与范围

- 新模型：注册普通nn.Module构造器及输入通道/尺寸约束，继续复用LightningModule。
- 新头：注册构造器并显式配置decode_head，遵循现有三尺度→logits接口；当前没有新增正式头。
- 新Dataset：实现image/mask协议与metadata后注入DataModule，标签映射在Dataset执行且必须显式。
- 新loss：在LightningModule损失构建位置增加明确名称与测试，不把损失塞进模型。
- 新学习率策略：在optim.py增加官方调度器与数值序列测试，不手写第二套优化循环。
- 对比/消融：复用相同data/loss/evaluation协议，用独立配置/输出目录保存完整记录。

03B不实现真实Potsdam/Vaihingen/LoveDA/Sentinel读取、DDP、滑窗、创新模块或正式性能实验。

## 10. 遗留问题与验收限制

尚未验证的问题均是上表中的云端运行项目；如失败，请返回unittest日志与smoke_report.json，
可定位到具体stage及误差/状态检查。范围止于03B，不自动进入03C或自动commit/push。
自动化测试共18项（16项逻辑/接口数值案例、2项小模型CUDA集成），本地均未执行。
静态检查报告位于 `outputs/task03b_static_checks.json`，与GPU验收报告严格区分。
