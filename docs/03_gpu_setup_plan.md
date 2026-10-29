# 03 并行智算云PyTorch25.03 GPU验证计划

本轮不连接服务器、不安装依赖、不下载权重、不运行真实训练。
优先候选是用户指定的PyTorch25.03镜像，版本以实际容器为准。

## 前置：先提交本地迁移，再导出

正式--require-clean导出必须基于干净的迁移提交，不能绕过dirty保护。
按用户追加授权，Codex完成检查后直接提交并同步迁移分支；部署仍按06文档生成新的两子模块release。
旧deploy/initialization-final属于任务01三子模块历史，不能用于新的Lightning部署。

## 第一步：保持原容器不变，只读盘点

上传新release后执行（示例路径需替换）：

~~~bash
cd /srv/sam2-rs/releases/<release_id>
python3 scripts/deployment_manifest.py verify --root . --manifest source_manifest.json
mkdir -p outputs/setup
python --version
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python scripts/check_environment.py --json outputs/setup/container_inventory.json
python -m pip list --format=json > outputs/setup/container_packages.json
python -m pip freeze > outputs/setup/container_freeze_before.txt
~~~

查看/etc/os-release、实际torchvision/Lightning/TorchMetrics/SAM2元数据，
检查是否存在/etc/pip/constraint.txt及环境的约束路径。不要自动改写全局约束。
nvcc --version仅用于确认可选扩展工具链；Hiera-only模式可跳过SAM2 CUDA后处理构建。
RTX4090型号、实际分配显存和驱动以nvidia-smi核验。

## 第二步：根据盘点结果决定稳定版本

先保留现有torch及配套torchvision，考察Lightning2.5.5/TorchMetrics1.7.4，
检查Python要求、nightly标签、torchvision算子、依赖约束和可选包冲突。
这些是调查候选，不是默认无条件安装的版本。
如需隔离环境，选择与实际镜像相容的Python，不固定3.10或CUDA12.1，
也不要默认将已工作的容器torch换成PyPI其他版本。

人工/下一轮检查后将选定版本及现有torch配对写入
outputs/setup/reviewed-constraints.txt，先执行解析预演：

~~~bash
python -m pip install --dry-run --report outputs/setup/install_plan.json -c outputs/setup/reviewed-constraints.txt -r requirements/gpu-bootstrap.txt
~~~

审核预演是否改变torch/torchvision、CUDA相关包与NumPy。出现意外替换时停止，
不要用忽略依赖/删除约束掩盖问题。此处reviewed-constraints.txt尚不存在，
必须依据实际容器生成，不能拿最低范围文件冒充审核后的锁文件。

## 第三步：仅在环境选择确认后安装和检查

以下只是后续命令模板，不是本轮执行记录：

~~~bash
python -m pip install -c outputs/setup/reviewed-constraints.txt -r requirements/gpu-bootstrap.txt
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation -e third_party/sam2 -c outputs/setup/reviewed-constraints.txt
python -m pip check
python scripts/check_environment.py --candidate --runtime --require-gpu --json outputs/setup/gpu_smoke.json
python -m pip freeze > outputs/setup/container_freeze_after.txt
python scripts/deployment_manifest.py verify --root . --manifest source_manifest.json
~~~

不安装GeoSeg训练requirements。环境脚本成功只证明少量导入/反传/合成指标，
没有实际Trainer训练、Hiera forward、预训练权重或数据加载验收。
失败保存完整日志回本地分析，不在远端保留唯一代码修改。

## 后续而非本轮的模型验收

在本地实现并提交适配器、语义头、Dataset和Lightning训练模块后：
先合成图像forward/backward，再小样本train/val/test；
验收TorchMetrics的ignore/空类/逐类指标、val_mIoU最佳权重与ID/RGB PNG，
最后扩展AMP、DDP及整图推理。当前没有可执行train.py，不能照旧框架命令训练。
