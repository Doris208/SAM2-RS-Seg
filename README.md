# SAM2-RS-Seg

基于 **MMSegmentation 1.x + Meta 官方 SAM2-Hiera** 的遥感全监督多类别语义分割科研工程。
唯一主工作区：`D:\Research\SAM2-RS-Seg`。工作方式为本地开发和 Git 提交、Linux RTX 4090 上运行。

**当前阶段：任务 01 初始化完成范围。模型集成、GPU 环境安装、训练和推理均待后续实现与验证。**
本轮不包含数据集或预训练权重，不包含可运行的训练配置。

## 目标与边界

第一阶段：三通道影像 → 预训练 Hiera 多尺度特征 → 简单语义解码器 → 类别预测。
优先 Potsdam RGB，然后 Vaihingen 和 LoveDA **2021**。
未来目标包含 train/val/test、mIoU、各类 IoU、mF1、OA、最佳 checkpoint、
单通道 ID PNG、RGB 语义图，以及滑窗和整图拼接。
原始 SAM2 Prompt Encoder / Mask Decoder 的独立接口在架构中保留。

MMSegmentation 是唯一主训练框架；SAM2 提供基础模型；GeoSeg 仅供数据与大图推理参考。
不复制 SAM2-UNet 或 MmSAM，不引入额外创新模块，不修改第三方核心源码。

## 源码与文档

| 位置 | 用途 |
|---|---|
| `configs/` | 原生配置的占位目录 |
| `projects/sam2_rs_seg/` | 自定义 Registry 模块命名空间，当前仅占位 |
| `datasets/` | 数据准备源码与最小标签契约 |
| `tools/` | 未来必要的实验辅助入口 |
| `scripts/` | 环境盘点、版本清单、源码导出和文件校验 |
| `tests/` | 合成数据格式和部署完整性测试 |
| `third_party/` | 三个完整官方源码子模块及版本记录 |
| `outputs/`, `data/`, `weights/`, `deploy/` | 本地运行目录，Git 忽略 |

- [架构与接口](docs/01_architecture.md)
- [依赖兼容性与官方依据](docs/02_environment_compatibility.md)
- [租赁 GPU 后的安装命令](docs/03_gpu_setup_plan.md)
- [数据集、标签及评价协议](docs/04_dataset_plan.md)
- [后续实现步骤](docs/05_implementation_steps.md)
- [Git / FileZilla 部署一致性](docs/06_deployment.md)
- [本轮核验记录](docs/07_validation.md)
- [完整项目目录树与第三方文件清单](docs/08_directory_tree.md)

## 当前固定源码

| 子模块 | 版本 / Commit |
|---|---|
| MMSegmentation | v1.2.2 / `c685fe6767c4cadf6b051983ca6208f1b9d1ccb8` |
| SAM2（含 SAM2.1） | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| GeoSeg | `9453fe48209c4626b29e35e61bab93b61212c4b1` |

三者实际源码位于 `third_party/`，由主仓库 gitlink 固定版本；
准确来源另见 `third_party/versions.json`。这些是源码快照，**尚未通过联合环境运行验证**。
已有本地完整源码，无须再次下载。以后从其他位置克隆本工程才需要：

```bash
git submodule update --init --recursive
git submodule status
```

## 本地安全检查

从项目根目录执行，无需 GPU，不安装依赖：

```powershell
python scripts/check_environment.py --json outputs/local_environment.json
python -m unittest discover -s tests -v
python scripts/deployment_manifest.py create --require-clean --output outputs/version_manifest.json
```

默认环境检查只读取安装元数据。缺少 GPU 框架会报告缺失，盘点成功不表示模型环境通过。
`--runtime --require-gpu --candidate` 仅供远端安装完成后使用。

候选环境为 Ubuntu 22.04、Python 3.10、PyTorch 2.5.1/cu121、torchvision 0.20.1、
MMEngine 0.10.7、MMSegmentation 1.2.2、**从源码构建 MMCV 2.1.0**。
当前没有确认可直接使用的匹配 MMCV wheel；必须先通过远端构建及算子检查。

## SourceTree

在 SourceTree 选择“添加 / Add”已有本地仓库，路径填入
`D:\Research\SAM2-RS-Seg`，确认后打开。主分支为 `main`，
子模块为有意固定的 detached HEAD，不要对其直接提交改动。
也可通过“文件 → 打开”选择此目录；无需创建远端仓库。
主仓库未设置 GitHub remote，未执行 push。

后续只在此工程继续开发。科研数据、权重、输出和凭据不得加入源码提交。
第三方各自许可证随源码保留；GeoSeg 为 GPL-3.0，未来如复制其实现应另行检查分发义务。
