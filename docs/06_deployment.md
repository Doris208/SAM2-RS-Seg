# 06 本地 Git、FileZilla SFTP 与远端版本一致性

本地主工作区固定为 `D:\Research\SAM2-RS-Seg`。
主仓库提交自定义代码和三个子模块 gitlink；第三方修改不直接发生在子模块中。
不自动创建远端仓库或 push。SFTP 部署可以完全不依赖远端 Git。

## 提交与固定版本

PowerShell 可在本次已验证的 Git 路径执行：

```powershell
$env:PATH = 'D:\Git\mingw64\bin;D:\Git\usr\bin;' + $env:PATH
git status --short --branch
git diff --check
git diff --submodule=short
git submodule status
git add <本轮明确审核的文件>
git commit -m "<变更说明>"
git log -1 --format=fuller
```

本机 Git 2.23 的 cmd 包装程序存在路径问题，当前命令需要时用
`& 'D:\Git\mingw64\bin\git.exe'` 替代 git。
运行 git submodule 还需要上述 Git 自带 usr/bin 的当前会话 PATH；
不需要永久修改系统 PATH 或 Git 全局配置。
此次直连 GitHub 超时，通过系统现有 `127.0.0.1:7898` 代理下载成功；
该代理仅作为本次命令参数，不写入项目或远端安装设置。

子模块以 detached HEAD 固定，有意不跟随上游分支自动升级。
`git submodule status` 正常应为前导空格；`+` 表示实际 checkout
与父仓库记录不一致，`-` 表示未初始化，`U` 表示冲突。
更新第三方版本必须显式评估、提交 gitlink、同步 versions.json 与兼容性文档后再部署。

## 生成版本清单与安全上传目录

版本清单脚本只需 Python 标准库，create 需要本地 Git，
verify 只需文件和 Python，不依赖 Git 或 GPU。
记录主 SHA、子模块 SHA、主/子模块未提交状态、子模块是否等于主仓库记录、
全部受控文件的长度与 SHA-256。

```powershell
# 可盘点 dirty 状态；部署必须加 --require-clean
python scripts/deployment_manifest.py create --output outputs/version_manifest.json
python scripts/deployment_manifest.py create --require-clean --output outputs/version_manifest.json
# <release_id> 用主仓库短SHA或日期+短SHA替换，每次使用新的空目录
python scripts/deployment_manifest.py create --require-clean --export-dir deploy/<release_id>
python scripts/deployment_manifest.py verify --root deploy/<release_id> --manifest deploy/<release_id>/source_manifest.json
```

export 默认强制主仓库和三个子模块干净，并检查子模块 gitlink 一致；
目标只能是本工程被忽略的 deploy/ 下新目录，已存在时拒绝覆盖。
按 Git 实际跟踪文件复制主仓库与三个子模块**完整工作树**，
所以不是仅复制三个空 gitlink。拒绝未初始化、嵌套未知子模块和指向仓库外部的符号链接。
SAM2 有四个旧版 YAML 路径的 Git 符号链接，Windows 默认检出为目标路径文本；
导出时根据 Git 记录解析仓库内部目标，把链接物化为真实 YAML 文件。
清单记录原始 git_mode、目标路径和物化状态；第三方 checkout 不作任何修改。
普通 create 清单记录原工作树字节；export 清单记录部署字节，二者不能混用。
Windows 上第三方文件可能是 CRLF，
不修改第三方文件以统一换行，传输时保持二进制字节即可。
清单记录 git_mode；如需要可执行 shell 脚本，未来远端应据此恢复执行权限；
当前工具均由 python 显式调用。

source_manifest.json 在提交后生成并随上传目录保存，不写入自身所属 Git 提交，
避免 Commit 自引用。不要编辑或重建上传后的清单来“消除”差异；
任何修改回本地提交、生成新目录再上传。

## FileZilla 操作

1. 站点管理器使用 **SFTP**，输入你自己的服务器和认证；本轮不连接。
   密码、SSH 私钥和站点凭据不保存到项目。
2. 先在本地完成清单生成和 export。FileZilla 本地选择 `deploy/<release_id>/`
   的内容，远端选择新建的 `/srv/sam2-rs/releases/<release_id>/`。
3. 传输类型设为 **二进制 / Binary**，保留相对路径和大小写，
   包括 `.gitmodules`、`.gitignore`、各级 `__init__.py`、YAML 配置和清单。
4. 必须上传 README、configs、projects、datasets（代码）、tools、tests、docs、
   scripts、requirements、third_party 的所有受控文件，以及 source_manifest.json。
   `third_party/sam2/sam2/`、`third_party/mmsegmentation/mmseg/`、
   `third_party/GeoSeg/geoseg/` 必须实际存在且有源码文件。
5. 只上传导出目录，不直接同步整个开发根目录。导出中没有主/子模块 `.git`
   内部目录或 `.git` 指针文件，也没有 untracked/ignored 数据、缓存或日志。
   上游自带的公开示例资源随其受控源码保留；不另外下载数据/权重。
6. 远端源码目录与大文件目录分开，禁止在 FileZilla 对项目父目录做“镜像删除”。
   不覆盖现有 release；先上传新版本、校验成功后再显式选择它运行。

推荐远端布局：

```text
/srv/sam2-rs/
  releases/<release_id>/   # 只有源码和清单，每个版本独立
  data/                   # 大型数据，独立管理
  weights/                # 权重，独立管理
  outputs/<run_id>/       # checkpoint / log / prediction
```

## 上传后的完整性与版本核验

```bash
cd /srv/sam2-rs/releases/<release_id>
python3 scripts/deployment_manifest.py verify --root . --manifest source_manifest.json
python3 -c "import json; m=json.load(open('source_manifest.json')); print(m['main']); print(m['submodules']); print('dirty=', m['dirty'])"
```

发现缺失或字节变化时 verify 返回非0，停止安装/实验，修正传输。
校验覆盖清单所列全部文件，但不会拒绝额外文件；使用新的专属 release 目录
避免旧源码混入。SHA-256 用于完整性，不是对来源的数字签名。
远端没有 `.git` 时只能依据随附清单与哈希核验版本，不能执行 git rev-parse；
这与带 Git 的克隆模式不同，不要在远端 git init 伪造历史。
如未来改用 Git 部署，再按指定主 SHA checkout、submodule update 并逐一核对。

## 实验日志中的源码版本

后续本地实现一个 MMEngine HOOKS 模块，在 before_run 阶段读取
release 根目录 source_manifest.json，将主/子模块 SHA、dirty、manifest 哈希
写入 MMLogger 和 MessageHub，并复制清单到外部 outputs/<run_id>/。
缺清单、dirty=true 或文件校验失败应阻止正式科研训练；开发 smoke 可另标 dirty。
同时保存最终配置、数据 schema/划分哈希、种子、权重哈希、环境 freeze 和 GPU信息。
该 Hook 尚未实现；本轮已实现的是清单、导出与校验工具，不声称日志自动接入完成。
