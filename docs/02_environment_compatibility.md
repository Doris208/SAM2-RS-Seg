# 02 环境兼容性调研

核查日期：2026-10-29。已核对官方源码声明和部分官方 wheel 索引；
**未在 Linux/RTX 4090 上安装、编译或运行，下面是待验证候选，不能称为已通过。**

## 源码事实与候选选择

| 组件 | 核查依据 / 限制 | 本工程候选 |
|---|---|---|
| 系统 | SAM2 INSTALL 推荐 Linux，Windows 推荐 WSL Ubuntu | Ubuntu 22.04 x86_64 |
| Python | SAM2 setup.py 要求 >=3.10 | 3.10 独立环境 |
| PyTorch | 当前固定 SAM2 setup.py 和 pyproject.toml 均要求 >=2.5.1 | 2.5.1 + cu121 |
| torchvision | SAM2 >=0.20.1；PyTorch 官方给出配对 | 0.20.1 + cu121 |
| CUDA runtime | 官方提供 torch 2.5.1 的 cu118/cu121/cu124 wheel | 12.1（随 PyTorch wheel） |
| CUDA Toolkit | 编译扩展需要 nvcc，与 torch.version.cuda 匹配 | 12.1，远端 devel 环境 |
| NVIDIA driver | 驱动必须支持所选 runtime 与 RTX 4090 | 建议 >=535，远端核验 |
| MMCV | MMSeg 1.2.2 运行时断言 >=2.0.0rc4 且 <2.2.0 | 2.1.0，完整 ops 源码构建 |
| MMEngine | MMSeg 断言 >=0.5.0 且 <1.0.0 | 0.10.7 |
| MMSegmentation | 固定稳定 tag，不追随 main | 1.2.2 / c685fe6767c4cadf6b051983ca6208f1b9d1ccb8 |
| SAM2 / SAM2.1 | 同一官方源码树含两代配置与模型支持 | 固定 2b90b9f5ceec907a1c18123530e92e794ad901a4 |
| NumPy | SAM2 >=1.24.4；避免旧扩展与 NumPy 2 ABI 风险 | 1.26.4 |
| GeoSeg | requirements含 timm 0.9.16、lightning 2.0.0、pytorch-lightning 2.3.0 | 仅阅读源码，不安装 |

MMEngine 0.10.7 满足声明区间，但与这组 torch 的运行行为仍需实测。
SAM2 的 Python 包元数据版本为 SAM-2 1.0，不能依赖它区分模型 2 与 2.1；
必须记录 Git SHA、模型 YAML 和权重哈希。
SAM2 早期版本的历史低版本 PyTorch 教程不适用于当前固定源码的安装要求。

## 关键：PyTorch 与 MMCV wheel

1. 官方 `cu121/torch2.1/index.html` 可查到 Linux cp310 的 MMCV 2.1.0 wheel。
   这是为 torch 2.1 构建的扩展，不能拿到 torch 2.5.1 环境中硬装。
2. 官方 `cu121/torch2.3/index.html` 此次列出的为 MMCV 2.2.0 wheel；
   MMSegmentation 1.2.2 会拒绝 2.2.0，不能靠删除版本断言规避。
3. 尝试读取 `cu124/torch2.5/index.html` 未成功，本次未确认
   torch 2.5.1 + MMCV <2.2 的官方匹配 wheel。访问失败不等于证明所有 wheel 都不存在。
4. 因此主候选保留 SAM2 当前官方依赖，从源码构建 MMCV 2.1.0。
   这在声明层面无直接版本区间冲突，但 C++/CUDA API 是否适配 torch 2.5.1
   **尚未验证，可能构建失败**。构建和算子测试是后续工作的第一道关口。
5. 不用 MIM 自动挑选未知 wheel，不允许安装时悄悄升级 torch/mmcv，
   不把 mmcv-lite 当作完整 mmcv 的等价替代（MMSeg 会导入 ops）。
   mmcv / mmcv-lite 共用命名空间，不应同时安装；MMCV 1.x 的 mmcv-full 也不适用。

候选选择用 SAM2 的最低当前官方 torch 要求限制变动范围，
用 MMSeg 的严格上限选择 MMCV；不是“最新版本拼接”，也不是可复现运行锁文件。
远端实测失败后应保存日志，再研究官方版本组合；本轮不降级 SAM2 依赖或修改第三方源码。
可另建纯 MMSeg 验证环境（torch2.1.x + 配套 MMCV2.1 wheel）帮助隔离问题，
但它不符合当前 SAM2 的官方要求，不能标注为本项目联合环境。

## CUDA 与系统风险

- `nvidia-smi` 显示的是驱动支持能力，不等于已安装 Toolkit。
  `torch.version.cuda` 是 wheel runtime，`nvcc --version` 才是 Toolkit。
- RTX 4090 的计算能力为 8.9。源码构建可设 `TORCH_CUDA_ARCH_LIST=8.9`；
  Ubuntu 22.04/GCC11 为候选，仍需核验 nvcc、GCC、内存及编译依赖。
- 第一阶段只用 Hiera，远端先设 `SAM2_BUILD_CUDA=0` 跳过 SAM2 可选的
  connected-components 后处理扩展；这不解决 MMCV ops 的构建需求。
  未来使用官方 mask 后处理时再验证 SAM2 扩展。
- SDPA、变长输入、AMP/bfloat16、位置嵌入插值、原生 Runner 的训练反传和
  4090 24GB 显存占用仍未测试。首版建议 Tiny、512 crop、batch=1 做 smoke，
  具体训练设置必须另行决定。
- requirements 中固定 NumPy 1.26.4 与 OpenCV 4.10.0.84，防止新 OpenCV 依赖
  引入 NumPy 2；其余辅助包仍是候选，最终以 pip check 和远端 freeze 为准。

## 官方来源

以下仓库链接使用本次固定 SHA，可直接与本地文件对照：

- [SAM2 setup.py](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/setup.py)
- [SAM2 pyproject.toml](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/pyproject.toml)
- [SAM2 INSTALL](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/INSTALL.md)
- [MMSeg 运行时版本断言](https://github.com/open-mmlab/mmsegmentation/blob/c685fe6767c4cadf6b051983ca6208f1b9d1ccb8/mmseg/__init__.py)
- [MMSeg 安装依赖](https://github.com/open-mmlab/mmsegmentation/blob/c685fe6767c4cadf6b051983ca6208f1b9d1ccb8/requirements/mminstall.txt)
- [GeoSeg requirements](https://github.com/WangLibo1995/GeoSeg/blob/9453fe48209c4626b29e35e61bab93b61212c4b1/requirements.txt)
- [PyTorch 官方历史版本与配对命令](https://pytorch.org/get-started/previous-versions/)
- [MMCV 官方源码构建指南 v2.1.0](https://github.com/open-mmlab/mmcv/blob/v2.1.0/docs/en/get_started/build.md)
- [MMCV cu121 / torch2.1 索引](https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html)
- [MMCV cu121 / torch2.3 索引](https://download.openmmlab.com/mmcv/dist/cu121/torch2.3/index.html)
- [未成功访问的 cu124 / torch2.5 索引](https://download.openmmlab.com/mmcv/dist/cu124/torch2.5/index.html)
- [NVIDIA CUDA GPU 能力表](https://developer.nvidia.com/cuda-gpus)

本地默认检查脚本只读取元数据与 nvcc/nvidia-smi 的版本信息，不安装任何软件。
深度检查可选启用，成功也仅代表少量导入/算子测试，不代表模型训练完成。
