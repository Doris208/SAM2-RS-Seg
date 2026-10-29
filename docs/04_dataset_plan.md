# 04 数据集与评价协议

范围：Potsdam RGB 优先，后续 Vaihingen 和 LoveDA **2021**。
本轮只核查公开定义、建立格式契约及合成格式测试，未下载/转换真实数据或确定论文实验划分。

## 统一存储契约

```text
data/<dataset>/
  images/{train,val,test}/
  masks/{train,val}/        # uint8 single-channel IDs, ignore=255
  splits/                  # scene/tile membership; copy final lists into source control
  metadata/                # original scene, crop offset, channels, label schema
```

源标签只读保留，转换产物另存，转换清单记录来源版本、输入/输出哈希、映射表、计数、
原始 scene/tile、坐标及空间分辨率。源码侧最终将划分列表、schema 和转换参数提交 Git；
影像和 Mask 放 data/ 或独立磁盘，不纳入版本库。
几何变换先对齐图像和标签，Mask resize/rotate/remap 只用最近邻；
Mask padding=255，图像可双线性缩放，不能将 RGB 标签图当训练 ID 图读入。
未知颜色/ID 默认报错，不静默并入背景或 ignore。

## Potsdam 与 Vaihingen

官方六类 RGB 标签如下。本项目拟转换成下列训练 ID（这是本工程编号，不声称原始为整数 ID）：

| 本工程训练 ID | 类别 | 官方标签 RGB |
|---|---|---|
| 0 | impervious surface | 255,255,255 |
| 1 | building | 0,0,255 |
| 2 | low vegetation | 0,255,255 |
| 3 | tree | 0,255,0 |
| 4 | car | 255,255,0 |
| 5 | clutter/background | 255,0,0 |

Clutter 是有效类别，第一版训练和六类评价均保留，不能自动当 ignore。
官方另有半径 3 像元圆盘腐蚀边界参考标签；full_reference 与 no_boundary
是不同评价协议，不能混用或声称本地 full GT 评价等于官方排行榜。
统一训练 ignore=255；只有经文件说明核实的 nodata / 边界剔除位置可写成255。
对黑色等未定义颜色先报错，不能未查明文件版本就自动映射。

| 数据集 | 图像读取计划 | 需注意 |
|---|---|---|
| Potsdam | 原始 TOP RGB TIFF，读取 R/G/B，核验 uint8/range；大图未来用窗口读取 | 官方也提供 RGBIR/IRRG；首版不用这两种替代 RGB |
| Vaihingen | 三通道 TIFF 为 NIR/R/G（IRRG），核验通道顺序与 nodata | 无自然蓝色通道，不能将三通道叫 RGB；单独配置归一化/模态名 |

转换后的0–5 Mask由未来原生torch Dataset按ID直接读取，ignore_index=255，
不在加载阶段隐式减1或重排。若数据来自其他工具，先核实1–6或0–5编号约定，
用值域、类别计数、转换清单和可视化验证，不自动猜测编号。
GeoSeg 的展示 palette 里 car 可为 [255,204,0]，不能据此替代官方 GT 的 [255,255,0]。

划分先按原始 tile 分 train/val/test，再切块；同一场景和重叠 crop 不可跨集合。
不自动采纳第三方把测试区命名为 val 的目录或划分。
实际拿到授权数据后明确可用 GT 与官方隐藏测试区；从训练区留验证集选择 best，
隐藏测试区只用于官方提交，本地 hold-out 与官方分数分开报告。

## LoveDA 2021

官方确认原始 PNG：0=no-data(ignore)，1=background，2=building，3=road，
4=water，5=barren，6=forest，7=agriculture。background 为有效类别。
图像为三通道 RGB PNG，Train/Val/Test 下分别有 Urban 和 Rural。
文件标识必须包含 split/domain，避免两域同名文件发生覆盖。

本工程推荐显式预处理：raw 0→255，raw 1–7→train 0–6；
原生Dataset按转换产物直接读取。只有明确调用
`loveda2021_raw_to_train_id` 才转换，禁止扫描目录后自动猜测是否需要减1。
转换前核验原始值域，转换清单区分 raw / converted。
如以后选择原始0–7存储，也必须显式定义一次转换并记录raw/converted，不能二次转换。
未来官方提交如需1–7输出，应另做显式导出映射并核验最新提交协议，
不要修改内部训练 ID PNG。

全监督任务使用官方 Train 的 Urban+Rural 训练、Val 验证；
不做源/目标域伪标签或 UDA。Test 无公开 GT 时只预测并按官方平台提交。
本项目锁定 2021 版本，不自动换成 EarthVQA/EarthVL 或其他衍生数据。

## 指标约定

在 GT !=255 的像元累积整个评价集的混淆矩阵，然后计算：

- IoU_c = TP_c / (TP_c + FP_c + FN_c)。
- F1_c = 2TP_c / (2TP_c + FP_c + FN_c)，mF1 为约定类别上的宏平均。
- OA = sum(TP_c) / 有效 GT 像元数；保留aAcc为OA别名，不是mAcc。
- mIoU 为类别 IoU 宏平均；主基线 ISPRS 六类、LoveDA七类。

不按小块先算百分比再简单平均。TorchMetrics累积整个epoch的混淆状态，
验证/测试分别reset，分布式时同步统计量后计算。
保持完全空类NaN/nanmean宏平均约定并报告有效类别数；
有GT无预测时按上述公式F1=0，分母为0的完全空类为NaN。
内部指标0–1，显示百分比时显式乘100。后续以合成混淆矩阵验收，
不把TorchMetrics默认macro或缺失类填0静默替代约定。
ISPRS 如另报去 clutter 五类均值，明确标作附加指标，写明混淆矩阵和 OA 的统计范围，
不替换六类结果。边界剔除结果另列。

## 原始来源与本地对照

- [ISPRS 六类与边界评价说明](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx)
- [Potsdam RGB / IRRG / RGBIR](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)
- [Vaihingen IRRG 定义](https://isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)
- [LoveDA 官方 README（2021 标签定义）](https://github.com/Junjue-Wang/LoveDA/blob/master/README.md)
- [LoveDA 2021 数据记录](https://doi.org/10.5281/zenodo.5706578)
- 本地 GeoSeg `geoseg/datasets/`、`tools/` 与 `inference_huge_image.py`，
  仅作参考，不直接复制默认划分、ignore、类别编号或评价平均范围。

最初访问 ISPRS 旧 education 链接遇到超时/404，已通过当前 resources 路径检索定义。
数据到手后的真实文件值域、压缩包版本、nodata、许可、划分和官方提交格式仍需逐项验证。

任务02仅更换训练底座：label_contract.py、类别编号、ignore255及nearest协议保持不变。
TorchMetrics评价模块和真实数据Dataset尚未实现。
