# 02 Lightning环境兼容性与候选

日期2026-10-29。已核查官方声明，未安装GPU依赖或连接服务器。
MMSegmentation/MMCV/MMEngine已退出当前运行依赖；旧约束不再决定镜像选择。

## 优先验证并行智算云PyTorch25.03镜像

用户指定该镜像为优先候选，先实测镜像本身，不强制锁定Python、CUDA、PyTorch或Lightning精确版本。
平台镜像可能重新封装，不能从25.03名称或截断的Ubuntu标签推断内容。

作为对照，NVIDIA原版25.03声明Ubuntu24.04、Python3.12、CUDA12.8.1和PyTorch
2.7.0a0+7c8ec84dab，并带/etc/pip/constraint.txt。
这不证明并行智算云镜像与原版相同，更不证明它与本工程完全兼容。
不要沿用任务01中强制Python3.10/cu121的安装路线，也不自动替换镜像预装torch。

## 声明、候选与未验证项

| 组件 | 依据或待调查候选 | 当前结论 |
|---|---|---|
| Python | 固定SAM2声明>=3.10；Lightning2.5.5声明>=3.9 | 满足下限后先保留镜像版本 |
| torch | 固定SAM2声明>=2.5.1 | 先核验镜像build，不强制2.5.1 |
| torchvision | 固定SAM2声明>=0.20.1，与torch配套 | 需导入及算子检查；不要独立升级 |
| Lightning | 先考察2.5.5，使用lightning.pytorch | requirements/pytorch/base.txt列torch>=2.1,<2.9用于CI |
| TorchMetrics | 先考察1.7.4 | base.txt列torch>=2.0,<2.8用于CI |
| CUDA/驱动 | 以torch.version.cuda、GPU和服务商驱动实测 | 不硬编码12.1；不自行替换系统驱动 |
| SAM2 | 固定2b90b9f5ceec907a1c18123530e92e794ad901a4 | 源码已在本地，GPU运行未验证 |
| NumPy/Pillow/Hydra/iopath | SAM2声明>=1.24.4 / >=9.4.0 / >=1.3.2 / >=0.1.10 | 先盘点镜像版本与ABI，不强制精确pin |
| GeoSeg | 仅源码参考 | 不安装其Lightning/timm等依赖 |

上表Lightning/TorchMetrics的上界在上游文件注释中说明用于CI稳定，发行安装时可能放宽；
不能把源码CI范围当作对每种容器/开发版torch的兼容保证。
NGC开发版torch即使满足数值下限，仍须实际导入、反传和Trainer smoke测试。

requirements/constraints-gpu-candidate.txt仅有候选最低范围，
requirements/gpu-bootstrap.txt列需要评估的运行依赖，二者都不是完整锁文件。
本轮未调用pip安装，更不会自动解析“最新版本”替代验证。
未来先在实际镜像选定版本、审阅dry-run，生成容器专用约束；
通过检查后再保存全量freeze和源码清单，形成稳定组合。
lightning发行包可能额外安装pytorch-lightning发行包；代码只用lightning.pytorch，
记录二者版本，避免混用对象。--candidate只检查声明下限，不验证精确版本配对。

## 编译和运行风险

- Hiera-only基线不需要当前已移除框架的自定义CUDA算子。
  SAM2_BUILD_CUDA=0可跳过SAM2可选connected-components后处理。
  无nvcc不应自动否决Hiera-only环境；未来启用该后处理扩展才核验Toolkit与torch runtime匹配。
- nvidia-smi显示驱动能力；torch.version.cuda是torch构建runtime；
  nvcc是可选扩展编译工具链，三者不是同一概念。
- NVIDIA原版25.03有默认pip约束；平台是否继承须实际检查。
  不直接编辑全局约束或用不匹配wheel覆盖；优先隔离环境保留可恢复基线。
- TorchMetrics可选音频/图像包、torchvision算子、NumPy ABI可能导致导入失败；
  需根据具体报错分析，而非无条件升级所有包。
- 4090的SDPA、AMP/bfloat16、Hiera非方形尺寸、训练梯度和显存均未验证。
  DDP与混合精度属于后续扩展，不是此次迁移的通过项。

## 来源

- [固定SAM2依赖](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/setup.py)
- [固定SAM2构建声明](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/pyproject.toml)
- [Lightning2.5.5依赖](https://github.com/Lightning-AI/pytorch-lightning/blob/2.5.5/requirements/pytorch/base.txt)
- [Lightning2.5.5打包/Python声明](https://github.com/Lightning-AI/pytorch-lightning/blob/2.5.5/src/lightning/__setup__.py)
- [TorchMetrics1.7.4依赖](https://github.com/Lightning-AI/torchmetrics/blob/v1.7.4/requirements/base.txt)
- [NVIDIA25.03对照说明](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-03.html)
- [PyTorch官方版本配对](https://pytorch.org/get-started/previous-versions/)
