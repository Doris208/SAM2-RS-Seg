# 03 未来 Linux GPU 安装计划

以下命令**本轮均未执行**。仅供租赁 Ubuntu 22.04 / RTX 4090 24GB 后使用。
先完成部署文件哈希核验，再创建独立环境。当前仍无模型配置，环境检查后不能直接启动本项目训练。

## 1. 选择并检查服务器

优先选择已带 NVIDIA 驱动、CUDA Toolkit **12.1 devel**、GCC11 的镜像。
只有 CUDA runtime 的镜像缺少 nvcc，不满足 MMCV 源码构建要求。
驱动由服务商/系统管理，不在这里执行自动系统级升级。
远端绝对路径由你选择，下面以 `/srv/sam2-rs/releases/<版本目录>` 为示例。

```bash
cd /srv/sam2-rs/releases/<版本目录>
python3 scripts/deployment_manifest.py verify --root . --manifest source_manifest.json
nvidia-smi
nvcc --version
gcc --version
```

确认 GPU 型号、显存、驱动支持 CUDA12.1、nvcc12.1、GCC11；
任一不符先调整镜像或工具链。不要将 nvidia-smi 的 CUDA Version 当成 nvcc 版本。

## 2. 创建环境并安装候选依赖

以下使用 conda 仅管理 Python，PyTorch 统一通过 pip 安装，避免重复的 conda/pip torch。
服务器需已有 conda/Miniconda；否则先按其官方安装方式准备。

```bash
conda create -n sam2-rs python=3.10 -y
conda activate sam2-rs
python -m pip install pip==24.3.1
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -c requirements/constraints-gpu-candidate.txt -r requirements/gpu-bootstrap.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

约束文件是候选直接依赖约束，bootstrap 中辅助包并非全部固定。
这有助于首次构建，仍须对最终解出的全部依赖执行 pip check、记录 freeze；
在实测前不能将此文件称为完整可复现环境锁。

## 3. 显式从源码构建 MMCV 2.1.0

```bash
mkdir -p outputs/setup
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.9"
set -o pipefail
MMCV_WITH_OPS=1 FORCE_CUDA=1 MAX_JOBS=4 python -m pip install -v --no-build-isolation --no-binary=mmcv -c requirements/constraints-gpu-candidate.txt mmcv==2.1.0 2>&1 | tee outputs/setup/mmcv_build.log
python -c "import mmcv; from mmcv.ops import nms; print(mmcv.__version__)"
```

这是待验证构建步骤，可能因 MMCV2.1 与 torch2.5 的 C++ API 不兼容而失败。
失败立即停止后续安装，保存日志并回到本地工程研究；不要自动换成 2.2.0、
删除版本断言、安装 torch2.1 wheel 或修改第三方源代码。
CUDA_HOME 示例必须和实际安装路径一致；限制 MAX_JOBS 可降低编译内存占用。

## 4. 从已上传本地源码安装

```bash
python -m pip install -e third_party/mmsegmentation -c requirements/constraints-gpu-candidate.txt
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation -e third_party/sam2 -c requirements/constraints-gpu-candidate.txt
python -m pip check
python scripts/check_environment.py --runtime --require-gpu --candidate --json outputs/setup/gpu_environment.json
python -m pip freeze > outputs/setup/pip_freeze.txt
python scripts/deployment_manifest.py verify --root . --manifest source_manifest.json
```

不安装 GeoSeg 的训练依赖，不安装 SAM2 notebooks/dev/demo 额外包。
SAM2_BUILD_CUDA=0 只跳过 SAM2 可选 mask 后处理扩展；MMCV 完整 ops 仍必须可用。
环境脚本执行导入、Tiny tensor backward、CUDA NMS 和注册模块加载，
**不构建实际 Hiera 网络，不加载权重，不证明联合模型训练通过**。

## 5. 环境通过后的下一轮验收

在本地实现适配代码并提交、重新导出和上传新版本之后，才执行：

1. 配置解析与 Registry 构建；Hiera 多尺度 shape/顺序与权重加载率核验。
2. 合成输入单步 forward/loss/backward；核实编码器梯度和冻结范围。
3. 小样本完整 GT 数据的 train/val/test smoke，验证 ignore、指标、best checkpoint 和 PNG。
4. 记录显存、逐类指标、样本输出；才开始完整实验。

未来原生入口格式（当前不存在实验配置，不可直接照跑）：

```bash
# <experiment.py> 待下一阶段在本地开发、提交并部署
PYTHONPATH="$PWD:${PYTHONPATH:-}" python third_party/mmsegmentation/tools/train.py configs/potsdam/<experiment.py> --work-dir /srv/sam2-rs/outputs/<run_id>
PYTHONPATH="$PWD:${PYTHONPATH:-}" python third_party/mmsegmentation/tools/test.py configs/potsdam/<experiment.py> /srv/sam2-rs/outputs/<run_id>/<best_checkpoint.pth>
```

代码版本文件、环境报告、配置快照、种子、数据划分及标签 schema 与每个实验一起保存；
训练日志源码版本机制见部署文档。禁止只在远端修改并保留唯一源码。
