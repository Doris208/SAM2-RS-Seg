# 任务03A：官方SAM2.1图像编码器与多类别语义头

## 当前结论

已实现 `SAM2ImageEncoder → SemanticSegmentationHead → SAM2SemanticSegmentor`。
按补充修订保留完整官方Hiera、FpnNeck、位置编码与 `scalp=1`，不添加第二套FPN。
现已针对云端Smoke发现的单尺度梯度断流问题，改为三个 `backbone_fpn` 尺度对齐、拼接和卷积融合。
用户反馈初版云端官方checkpoint严格加载、特征尺寸、六类别logits、CE及backward通过，
但深层Hiera和大部分Neck投影没有梯度；旧版PASS不能作为完整结构通过的证据。
本轮只修改代码并做静态检查，不在本地进行CPU或随机初始化替代验收。
修正版最终结果等待云端RTX 4090使用官方预训练权重重新验证；初版CPU记录仅作历史。
本轮Python语法编译、配置JSON解析和 `git diff --check` 已通过；未执行模型forward/backward或运行时测试。

## 1. 工程检查与保护

- 检查日期：2026-10-29。
- 初始分支：`doris`；HEAD：`812340116d39484511363cb85e5fdf3947ebe62c`。
- 初版03A修改前主仓库、SAM2、GeoSeg工作区均干净；本轮修正前已有初版暂存改动，保留暂存内容、不提交。
- SAM2实际HEAD：`2b90b9f5ceec907a1c18123530e92e794ad901a4`。
- GeoSeg实际HEAD：`9453fe48209c4626b29e35e61bab93b61212c4b1`。
- 既有模型目录仅为占位命名空间；保留标签契约、任务01/02测试、部署脚本和训练设计状态。
- 修改过的原文件已备份至 `outputs/task03a_backup/`（Git忽略）。
- 本轮修正另有完整文件备份：`outputs/task03a_multiscale_backup_20261029_224647/`。
- 没有修改子模块、切换分支、执行commit/push、下载权重或启动正式训练。

默认Python为3.11.7，没有PyTorch。另一个已有环境 `D:/Anaconda3/envs/aglldiff/python.exe`
提供Python3.8.20、PyTorch2.3.0 CPU、torchvision0.18.1a0，没有CUDA、SAM2、Hydra或iopath。
此环境只用于独立PyTorch模块检查，不能认证官方SAM2环境。
官方 `third_party/sam2/INSTALL.md` 声明Python≥3.10、PyTorch≥2.5.1及匹配的torchvision，推荐Linux。
曾尝试把缺失的轻量依赖单独安装到 `outputs/task03a_deps/`，由于PyPI TLS连接失败未完成；
没有升级或修改原有Conda环境、PyTorch、CUDA或torchvision。

后续检查发现本地已有 `shared-nvme/weights/sam2.1_hiera_tiny.pt`（156,008,466字节），
无需重新下载。使用已有CPU PyTorch的 `torch.load(..., weights_only=True)` 已成功读取：
包含 `model`、471个state张量，其中162个属于image_encoder，含全部8个neck权重/偏置张量。
SHA-256：`7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`。
这只是文件内容盘点，不等同于官方模型严格加载成功；记录在 `outputs/task03a_checkpoint_inventory.json`。
本地完整模型验证的限制是缺少兼容运行依赖及CUDA，而不是缺少该文件。

## 2. 实际阅读与复用的源码

| 来源 | 实际阅读的文件 | 用途与适配方式 |
|---|---|---|
| 固定版本Meta SAM2，Apache-2.0 | `sam2/build_sam.py` | **直接调用** `build_sam2`；使用完整模型checkpoint严格加载，然后取 `image_encoder`。官方 `load_state_dict` 默认strict=True，缺失、多余或尺寸错误均抛出异常 |
| 同上 | `sam2/modeling/backbones/hieradet.py` | 核对Hiera按高到低分辨率返回四级特征，`channel_list`反向排列；不重写其结构 |
| 同上 | `sam2/modeling/backbones/image_encoder.py` | **直接复用**ImageEncoder和FpnNeck；核对返回字典、scalp裁剪和各层top-down行为 |
| 同上 | `sam2/configs/sam2.1/sam2.1_hiera_t.yaml`、`sam2/__init__.py`、`INSTALL.md` | 使用官方配置包路径和Hydra初始化，保留 `scalp=1`、256通道、`fpn_top_down_levels=[2,3]`、nearest内部插值及位置编码 |
| [SAM2-UNet](https://github.com/WZH0120/SAM2-UNet/blob/main/SAM2UNet.py) | `SAM2UNet.py` | **只参考组织方式**：官方builder取编码器再接语义网络。不移植冻结、Adapter、RFB、U形解码器和单类输出；其Large配置不适用于当前Tiny |
| [MmSAM](https://github.com/W-qp/MmSAM) | `model/Build_models.py`、`model/Multimodal_functions.py`、`fine_tuning.py` | **只参考接口**：完整image_encoder、`backbone_fpn`、可配置类别数与外置损失。其原SAM2配置、1024重采样、多模态、LoRA/Adapter和训练流程均不移植 |
| 固定版本GeoSeg，GPL-3.0 | `train_supervision.py`、`geoseg/datasets/potsdam_dataset.py`、`geoseg/models/UNetFormer.py` | **只阅读接口**：普通网络与训练封装分离、原尺寸logits。没有复制实现、导入旧pytorch_lightning或采用其标签忽略约定 |
| 原生PyTorch | `nn.Module`、`Conv2d`、`BatchNorm2d`、`ReLU`、`Sequential`、`functional.interpolate`、`CrossEntropyLoss` | **直接调用组件**；项目自行实现必要的连接、合成测试和诊断代码 |

SAM2-UNet与MmSAM文件通过GitHub连接实际读取；未复制其第三方模型实现。
torchvision独立FPN要求已被补充指令取消，本次没有读取或使用其FPN源码。
没有开发Lightning或TorchMetrics模块，也没有把未读取的相关仓库/文档计入参考来源。

## 3. 代码变更清单

以下路径均相对于 `D:/Research/SAM2-RS-Seg`。

| 文件 | 职责 |
|---|---|
| `projects/sam2_rs_seg/models/backbones/sam2_image_encoder.py` | 官方builder适配；校验预训练参数；保留整个image_encoder与原始返回字典 |
| `projects/sam2_rs_seg/models/decode_heads/semantic_head.py` | 三尺度双线性对齐、cat为768通道、768→128的3×3卷积、BN、ReLU、128→K的1×1分类卷积及原图尺寸插值 |
| `projects/sam2_rs_seg/models/segmentor.py` | 可配置普通nn.Module，将全部三个 `backbone_fpn` 尺度交给语义头，返回单个logits Tensor |
| `projects/sam2_rs_seg/models/__init__.py`、`backbones/__init__.py`、`decode_heads/__init__.py` | 将原占位命名空间改为导出实际模型类；SAM2依赖在编码器构造时延迟导入 |
| `scripts/task03a_checks.py` | 标签/合成数据检查；新增完整梯度连接、四阶段/全部block、四个Neck投影和关键权重的严格验收 |
| `scripts/smoke_03a.py` | hooks记录实际三尺度和拼接尺寸；采用严格梯度策略 `three_scale_all_stages_v2`；失败退出码1 |
| `tests/test_task03a.py` | 同步三尺度输入、全部尺度输入梯度、深层和Neck断流/全零梯度拒绝检查；本轮未执行运行时测试 |
| `configs/task03a_model.json` | 随机初始化模型构造参数示例，不是训练配置 |
| `configs/potsdam/hiera_tiny_lightning.plan.json` | 同步已实现的编码器/语义头接口与三个输出尺度；训练状态保持design_only_not_runnable |
| `configs/README.md`、`README.md`、`docs/task03a.md` | 当前能力、边界、参考及复现命令 |

没有改变既有标签和部署接口。未创建FPNSegmentationHead、Registry、Factory、LightningModule、Dataset或正式训练入口。

## 4. 模型接口与结构检查

```python
from projects.sam2_rs_seg.models import SAM2SemanticSegmentor

model = SAM2SemanticSegmentor(
    num_classes=6,
    head_channels=128,
    pretrained=True,
    checkpoint_path=checkpoint_path,  # caller-provided full official checkpoint
    sam2_config="configs/sam2.1/sam2.1_hiera_t.yaml",
    device="cuda",
)
logits = model(images)  # float RGB [B,3,H,W] -> raw logits [B,K,H,W]
```

默认 `pretrained=True`，必须提供存在的checkpoint；随机测试必须显式 `pretrained=False`。
禁用预训练但仍传checkpoint也会报错，避免使用者误以为已经加载。
加载失败不会回退；`pretrained_loaded`只在官方builder成功返回后设置。
构造时完整官方SAM2被实例化并加载，最终模型仅保留ImageEncoder，不保留Prompt、Mask Decoder或Memory模块。
没有重置编码器权重，没有 `no_grad`、`detach` 或冻结操作；全部编码器参数保持可训练。

输入已由调用方排列通道、转换浮点并归一化，模型内部没有再次归一化。
当前适配器明确要求H/W均为32的正整数倍且至少32；没有自动补边、改官方配置或掩盖尺寸限制。
语义头的插值固定 `mode="bilinear", align_corners=False`，不输出概率或argmax。

| 位置 | 256×256输入的形状 | 本地验证状态 |
|---|---|---|
| Hiera Trunk四级输出 | `[1,96,64,64]`、`[1,192,32,32]`、`[1,384,16,16]`、`[1,768,8,8]` | 源码推导的预期，尚未通过实际官方前向验证 |
| 官方FpnNeck，scalp前 | `[1,256,64,64]`、`[1,256,32,32]`、`[1,256,16,16]`、`[1,256,8,8]` | 源码推导的预期，待Smoke打印 |
| ImageEncoder.backbone_fpn | 前三个尺度：64×64、32×32、16×16，均256通道 | 源码推导的预期，待官方前向验证 |
| 新语义头输入 | 上述三个 `backbone_fpn` 张量 | 修正版待云端验证 |
| 对齐拼接后的特征 | `[1,768,64,64]` | 修正版待云端验证；Smoke记录为 `semantic_fusion` |
| 语义头及完整模型logits | `[1,6,256,256]` | 修正版待云端验证 |

第二、第三尺度先按 `bilinear, align_corners=False` 对齐到第一个尺度，再按原顺序沿通道拼接。
拼接后通过原有卷积、BN、ReLU和分类层；只把第一层卷积输入由256改为768，保持中间宽度128。
语义头参数量按结构计算为 **885,766**，不是本轮运行时实测；Smoke将重新统计各部件参数量。
初版295,942参数与CE=1.860856294631958仅对应旧单尺度头，不能作为当前版本结果。

### 三尺度特征流与梯度验收

官方Tiny配置仅在第2/3级执行top-down，第0/1级直接采用本层横向投影。
旧版只选择 `backbone_fpn[0]`，损失只连接第一阶段和 `convs[3]`。
修正版联合三个尺度：第0/1尺度分别接入 `convs[3]`、`convs[2]`；第2尺度包含
`convs[1]` 与最低分辨率 `convs[0]` 的top-down结果，因此应使全部四阶段与四个投影参与损失。
没有修改官方scalp、配置、内部FpnNeck或预训练参数。

诊断分别列出 `frozen`、`none`、`zero`、`nonfinite`、`nonzero` 的参数名和数量。
新的PASS条件同时满足：

- Hiera、Neck、语义头全部参数可训练、有梯度且梯度有限；None、冻结或NaN/Inf均失败。
- 根据实际 `stage_ends` 检查四阶段及每个block；Tiny对应0、1–2、3–9、10–11。
- 每个block的 `attn.qkv.weight` 有有限非零梯度，明确包含 `blocks.9`、`blocks.11`。
- 四个Neck投影均有完整梯度，其各自的 `conv.weight` 必须有限且非零；包括通过top-down参与的 `convs[0]`。
- patch embedding权重及语义头融合/分类卷积权重均有限且非零。

JSON的 `gradients` 包含 `hiera_stages`、`hiera_blocks`、`fpn_neck_branches`、
`required_weights` 和 `failures`，总体 `ok=false` 时Smoke不得输出PASS。
个别参数全零梯度仍逐项列出；关键权重全零一律失败，不把正常的个别零梯度与未连接混淆。

## 5. 初版历史测试记录（不作为三尺度修正版验收）

### 默认Python基础回归

```powershell
python -m unittest discover -s tests -v
```

实际结果：24项中 **13通过、11跳过**；退出码0。
原有任务01/02测试13项通过；新增测试因没有torch而跳过，没有计为PASS。
日志：`outputs/task03a_default_tests.txt`。

### 已有CPU PyTorch环境

```powershell
& 'D:/Anaconda3/envs/aglldiff/python.exe' -m unittest tests.test_task03a -v
```

实际结果：11项中 **8通过、3跳过**；退出码0。
首次在沙盒执行时，仅临时目录清理因访问权限失败；随后在正常临时目录权限下重新执行上述测试通过。
日志：`outputs/task03a_cpu_tests.txt`。

通过项目：

1. 合成256×256输入/标签，覆盖0..5和255；语义头六类输出、有限CE、输入特征与语义头有效梯度。
2. 两类输出及矩形目标尺寸，类别数未写死。
3. 原始logits与指定双线性插值，输出不经过概率激活。
4. 忽略像元不影响有效像元平均损失且梯度为零；类别0梯度非零。
5. 全255批次、非法ID、错误mask dtype拒绝。
6. 梯度诊断能区分被冻结、None、全零、NaN/Inf、有限非零。
7. 缺失checkpoint/参数矛盾直接报错，不进入随机回退。
8. 替身编码器连接测试：segmentor选择特征并恢复原图尺寸；不代表官方编码器已验证。

三项官方依赖测试跳过：完整随机模型forward/backward与两类输出、官方不完整checkpoint严格拒绝、builder失败传播。
当前CPU解释器低于官方Python要求且缺少官方依赖。

### 随机初始化完整模型Smoke

```powershell
& 'D:/Anaconda3/envs/aglldiff/python.exe' scripts/smoke_03a.py `
  --device cpu --random-init --json outputs/task03a_random_cpu.json
```

**未通过**，退出码1：`ModuleNotFoundError: No module named 'sam2'`。
报告中 `checkpoint_loaded=false`，未产生官方feature shapes、完整模型loss或编码器梯度结果。
该执行没有使用替身编码器或把随机权重标为预训练。

### 静态与命令行检查

```powershell
python -m compileall -q projects scripts tests
& 'D:/Anaconda3/envs/aglldiff/python.exe' scripts/smoke_03a.py --help
```

已执行，通过。两个JSON配置也通过解析检查。
最终 `git diff --check` 通过；两个子模块仍干净且HEAD与固定版本一致。

### 官方预训练GPU Smoke

**未执行**：本地没有CUDA及可用的官方运行环境，没有连接云端。
本地现有checkpoint仅完成只读内容检查，没有被完整模型严格加载。
未下载重复权重，没有将云端预期结果记为通过。

| 检查 | 当前证据 |
|---|---|
| 完整官方checkpoint严格加载，含Hiera和FpnNeck | 加载路径按官方builder实现；实际成功加载待云端验证 |
| 完整官方模型forward与有限CE | 待云端验证 |
| Hiera有效梯度 | 待云端验证 |
| 官方FpnNeck有效梯度 | 待云端验证 |
| 新语义头有效梯度 | 独立CPU测试通过；完整官方链条待验证 |

## 6. 云端核验命令

以下命令仅供人工在已有云端RTX 4090环境执行，不安装或升级依赖；本轮不以本地CPU测试替代验收。
先确保本次源码已同步到云端；现有源码导出脚本基于Git跟踪文件，未提交新增文件不能依赖旧的干净提交导出。

```bash
cd /root/projects/SAM2-RS-Seg
conda activate doris
export PYTHONPATH="$PWD/third_party/sam2${PYTHONPATH:+:$PYTHONPATH}"
git -C third_party/sam2 rev-parse HEAD
python -c "import sam2, torch; print(sam2.__file__); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python scripts/smoke_03a.py --device cuda \
  --checkpoint /root/shared-nvme/weights/sam2.1_hiera_tiny.pt \
  --height 256 --width 256 --num-classes 6 --seed 42 \
  --json outputs/task03a_pretrained_cuda_multiscale.json
```

SAM2 HEAD须为上文固定值；`sam2.__file__`应指向该仓库 `third_party/sam2/sam2/`。
GPU脚本batch固定1、float32、ignore_index=255，不执行optimizer.step。
需要返回终端日志及 `outputs/task03a_pretrained_cuda_multiscale.json`：
确认 `gradient_policy=three_scale_all_stages_v2`、`status=PASS`、`checkpoint_loaded=true`、
三尺度与拼接尺寸、有限loss、四阶段/全部blocks/四个Neck分支检查通过及实际参数量。
若失败，报告会记录异常并返回非零退出码，不自动更换配置、设备或权重。

## 7. 后续接口与任务边界

当前模型为普通 `nn.Module`，`forward(images)`返回一个K通道logits Tensor，
类别数、语义头宽度、官方配置、预训练开关、权重路径和设备均可配置。
可以由后续LightningModule持有并调用，也可在03B再添加最简注册/构建机制。
损失、数据、指标、优化器和保存逻辑都在模型之外。
本次没有提前开发上述后续框架，也没有实施真实Potsdam训练或科研创新模块。

三尺度修正版代码已完成；本轮仅执行静态检查，最终预训练验收等待人工云端核验，范围仍止于03A。
