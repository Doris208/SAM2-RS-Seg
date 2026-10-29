# 07 初始化核验记录

日期：2026-10-29。范围仅任务01；不包含模型集成或训练。

## 已核实的本地状态

- 工作目录：D:\Research\SAM2-RS-Seg；开始时为空，无父仓库继承。
- Git：本机 D:\Git\mingw64\bin\git.exe，2.23.0.windows.1。
  已配置用户身份，未创建虚构身份；main 为项目分支。
- 三个官方仓库为实际完整检出（不是空目录或仅有链接），非浅克隆命令获取，
  .gitmodules 登记官方地址，主仓库用 gitlink 管理。
- MMSeg 固定 v1.2.2；SAM2 / GeoSeg 固定当前检出的 SHA；
  三个子模块均 detached HEAD，未修改第三方核心源码。
- 子模块受控文件数：SAM2 569，MMSegmentation 1758，GeoSeg 67。
  完整相对路径清单见 source_inventory.txt。
- SourceTree.exe 实际存在，版本 3.4.27.0；主仓库是标准 .git 仓库，可按 README 的路径添加。
  本轮未操作 SourceTree 图形界面，不声称已完成 GUI 添加。
- 最初查询主仓库无远端；最终检查时已有
  `origin=git@github.com:Doris208/SAM2-RS-Seg.git`，保留该设置。
  本轮未执行配置远端、创建 GitHub 仓库或 push。

Git 子模块命令最初缺少 Git 自带 shell 的 PATH；在当前进程补充后恢复。
网络直连 GitHub 超时，使用系统已配置的本地代理后源码获取成功。
未改变系统代理、Git 全局设置或官方源码。

## 环境盘点

执行 `python scripts/check_environment.py --json outputs/local_environment.json`。
本机 Python3.11.7（Anaconda），NumPy1.26.4、Pillow10.2.0；
未检测到已安装的 torch、torchvision、mmcv、mmengine、mmsegmentation、SAM-2。
PATH 未发现 nvidia-smi/nvcc。盘点退出0只表示脚本完成，不代表 GPU 环境通过。
本轮未安装任何 Python/CUDA 训练依赖。

## CPU 检查

`python -m unittest discover -s tests -v` 使用合成样本，覆盖：

1. 六类 ISPRS 精确 RGB 映射和未知颜色拒绝（包含 GeoSeg 展示色差异）。
2. LoveDA2021 raw0忽略、raw1–7转换，拒绝非法值。
3. class0有效、ignore255与类别区间隔离，拒绝浮点ID。
4. 单通道 PNG 往返、nearest resize 后标签集合不变。
5. 部署文件哈希一致，以及篡改/缺失文件检测。
6. 清单路径不能越过源码根目录。
7. SAM2 Windows Git链接占位文本解析成真实 YAML、拒绝仓库外目标。
8. 超过260字符的部署路径创建、读写和哈希核验。

实际结果：8项测试全部通过，16个自有 Python 文件 AST 解析通过。
这些不是实际数据集或模型测试；通过不代表训练损失的 ignore 已验收。
自有 Python 文件执行 AST 解析，第三方代码不执行不改写。

## 初始提交后的部署验收

初始 Commit 已创建：
`70b7da5e90a0b56c046bce1a9ed4e222bc5e93c7` / `Initial project structure`。
该版本导出2436个受控源码文件全部通过SHA-256核验，四个SAM2 YAML链接已物化；
导出中无Git内部文件、data、weights或outputs目录，三个子模块均干净且gitlink一致。
实际结果保存在被忽略的 `outputs/initialization_validation.json`。
最终文档更新另作提交，最终版本以新清单的main.commit为准。

以后每次提交后重新执行（<new_release_id>必须是尚不存在的目录）：

```powershell
python scripts/deployment_manifest.py create --require-clean --output outputs/version_manifest.json
python scripts/deployment_manifest.py create --require-clean --export-dir deploy/<new_release_id>
python scripts/deployment_manifest.py verify --root deploy/<new_release_id> --manifest deploy/<new_release_id>/source_manifest.json
```

检查主 SHA、三个子模块 SHA、dirty=false；主仓库与子模块状态均应干净。
导出应有全部受控源码、四个物化 YAML alias、无 .git 内部文件、无数据/权重/输出。
版本清单在提交后生成，初始 Commit 的实际 SHA 请用 `git log --oneline` 或清单查询，
不把尚未产生的 SHA 预写入其自身提交。
项目完整目录与源码清单见 08 文档。

## 尚未验证

- Linux CUDA12.1 工具链、MMCV2.1.0 对 torch2.5.1 的源码编译和 CUDA算子。
- SAM2/Hiera 与 MMSeg 的实际联合构建、权重加载、前后向、AMP、显存。
- 真实数据版本、标签分布、场景划分、训练/验证/测试与指标闭环。
- best checkpoint、ID/RGB PNG、小图及滑窗拼接、大图地理参考。
- SFTP 实际传输、远端文件校验、远端安装和 SourceTree GUI 操作。
- 训练日志自动写入源码版本的 Hook（目前只有设计）。

未租赁或连接 GPU；未下载公开数据集或大型权重；未执行训练。
