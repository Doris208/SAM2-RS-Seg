# 06 两子模块版本管理、源码导出与FileZilla部署

本地主工作区D:\Research\SAM2-RS-Seg，远端GPU只运行已提交的本地版本。
当前运行路线为PyTorch/Lightning/TorchMetrics/SAM2；GeoSeg仅参考。

## Git与SourceTree

SourceTree继续打开现有主目录。main历史与GitHub原配置不变。
任务02变更在refactor/lightning-backend，用户已追加授权检查后直接commit和push该分支。
规范git rm移除旧子模块；提交前核查全部差异及测试，不要求用户手动审查。

~~~powershell
$env:PATH = 'D:\Git\mingw64\bin;D:\Git\usr\bin;' + $env:PATH
git status --short --branch
git diff --check
git diff --submodule=short
git diff --cached --submodule=short
git submodule status
~~~

本机Git可用D:\Git\mingw64\bin\git.exe，子模块shell还需要上述当前进程PATH。
检查文件、测试和删除清单后创建有意义的迁移提交，SourceTree继续用于查看和管理。
不重写main、不强推；本次明确同步授权只推送迁移分支，不自动合并main。

当前仅保留两个gitlink及.gitmodules登记：

- third_party/sam2：2b90b9f5ceec907a1c18123530e92e794ad901a4
- third_party/GeoSeg：9453fe48209c4626b29e35e61bab93b61212c4b1

两者detached HEAD是有意固定，不直接编辑或升级。
移除子模块在.git/modules中的历史缓存保留，便于检出旧main；
它不是当前工作树源码，不上传。后续子模块变更需单独审查版本记录。

## 正式清单/导出必须先提交

存在未提交修改时--require-clean应拒绝，不能忽略dirty。
完成迁移提交后，release_id使用尚不存在的新目录：

~~~powershell
python scripts/deployment_manifest.py create --require-clean --output outputs/version_manifest.json
python scripts/deployment_manifest.py create --require-clean --export-dir deploy/<release_id>
python scripts/deployment_manifest.py verify --root deploy/<release_id> --manifest deploy/<release_id>/source_manifest.json
~~~

create无--require-clean可盘点dirty，不是正式部署验收：
未跟踪的新文件不会被Git受控文件流程收录，必须先审核并提交。
清单在提交后生成，不纳入自身Commit，避免自引用。
记录主SHA、dirty、两子模块SHA/状态/与主提交gitlink匹配及文件SHA-256/长度。
只复制受控源码，不复制ignored/untracked数据或.git内部文件。
导出目录须为deploy/下新位置；失败产生的副本不能直接上传。

Windows保留扩展长路径处理。SAM2四个旧YAML alias是Git符号链接，
Windows可能检出为目标路径文本，导出时根据Git目标物化为真实YAML，
记录git_mode/link_target/link_materialized，不修改官方checkout。
拒绝越过仓库边界的链接。普通清单记录工作树字节，export记录部署字节，不能混用。
清单记录执行位；未来shell入口可能需要Linux chmod，当前工具由python调用。

## FileZilla SFTP

1. 使用用户自己的SFTP认证，密码/Token/SSH私钥不进入源码。
2. 从新deploy/<release_id>/上传到新/srv/sam2-rs/releases/<release_id>/。
3. Binary传输，保留大小写、相对路径、隐藏文本配置和实际子模块源码。
4. 上传README、configs、projects、datasets（代码）、tools、scripts、tests、
   requirements、docs、third_party/sam2、third_party/GeoSeg及versions.json，
   以及source_manifest.json。
5. 不上传开发根目录/.git、子模块.git指针、.git/modules、本地数据、权重、
   outputs、deploy缓存、环境或凭据。
6. 不覆盖旧release，不对父目录做镜像删除，权重与结果独立存放。

~~~text
/srv/sam2-rs/
  releases/<release_id>/    # 源码和清单
  data/                    # 真实数据
  weights/                 # 权重
  outputs/<run_id>/        # 日志、checkpoint、预测
~~~

任务01旧deploy/initialization-final和outputs/task01_handoff属于历史产物，
可能包含旧路线源码/说明；不得当作任务02release继续部署。
保留历史，不原地篡改成Lightning“已通过”。

## 远端核验和实验版本记录

~~~bash
cd /srv/sam2-rs/releases/<release_id>
python3 scripts/deployment_manifest.py verify --root . --manifest source_manifest.json
python3 -c "import json; m=json.load(open('source_manifest.json')); print(m['main']); print(m['submodules'])"
~~~

verify只检查清单文件，不拒绝额外文件，因此用新建独立目录。
缺失/哈希变化返回非0，停止安装或实验。哈希证明完整性，不是数字签名。
无.git副本依据manifest核验，不在远端git init伪造历史。
未来Git部署须检出指定迁移Commit并初始化两个子模块。

后续实现Lightning Callback(on_fit_start/on_test_start等)，校验manifest，
写主/子模块版本与清单哈希到logger/实验目录，同时保存最终配置、freeze、
数据schema/划分、种子和权重哈希。Callback尚未实现，现有工具不自动接入Trainer。
