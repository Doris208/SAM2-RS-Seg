# 08 实际项目目录树

下面展开全部自有受控源码；第三方源码共2394个文件，完整展开见 [full_directory_tree.txt](full_directory_tree.txt)，逐文件路径见 [source_inventory.txt](source_inventory.txt)。

排除 Git 内部目录、Python缓存及运行产物。data/weights/outputs/deploy 为 Git 忽略的运行位置；deploy 在导出时创建。源码子模块实际位于本地，并非空占位。

```text
SAM2-RS-Seg/
|-- configs/
|   |-- _base_/
|   |   |-- datasets/
|   |   |   `-- .gitkeep
|   |   |-- models/
|   |   |   `-- .gitkeep
|   |   `-- schedules/
|   |       `-- .gitkeep
|   |-- loveda/
|   |   `-- .gitkeep
|   |-- potsdam/
|   |   `-- .gitkeep
|   |-- vaihingen/
|   |   `-- .gitkeep
|   `-- README.md
|-- data/
|   `-- [ignored]
|-- datasets/
|   |-- __init__.py
|   |-- label_contract.py
|   `-- README.md
|-- deploy/
|   `-- [ignored]
|-- docs/
|   |-- 01_architecture.md
|   |-- 02_environment_compatibility.md
|   |-- 03_gpu_setup_plan.md
|   |-- 04_dataset_plan.md
|   |-- 05_implementation_steps.md
|   |-- 06_deployment.md
|   |-- 07_validation.md
|   |-- 08_directory_tree.md
|   |-- full_directory_tree.txt
|   `-- source_inventory.txt
|-- outputs/
|   `-- [ignored]
|-- projects/
|   |-- sam2_rs_seg/
|   |   |-- datasets/
|   |   |   `-- __init__.py
|   |   |-- engine/
|   |   |   `-- __init__.py
|   |   |-- evaluation/
|   |   |   `-- __init__.py
|   |   |-- models/
|   |   |   |-- backbones/
|   |   |   |   `-- __init__.py
|   |   |   |-- decode_heads/
|   |   |   |   `-- __init__.py
|   |   |   |-- prompt_adapters/
|   |   |   |   `-- __init__.py
|   |   |   `-- __init__.py
|   |   `-- __init__.py
|   `-- __init__.py
|-- requirements/
|   |-- constraints-gpu-candidate.txt
|   `-- gpu-bootstrap.txt
|-- scripts/
|   |-- __init__.py
|   |-- check_environment.py
|   `-- deployment_manifest.py
|-- tests/
|   |-- __init__.py
|   `-- test_initialization.py
|-- third_party/
|   |-- GeoSeg/
|   |   `-- [official pinned source; see full tree]
|   |-- mmsegmentation/
|   |   `-- [official pinned source; see full tree]
|   |-- sam2/
|   |   `-- [official pinned source; see full tree]
|   `-- versions.json
|-- tools/
|   `-- README.md
|-- weights/
|   `-- [ignored]
|-- .gitattributes
|-- .gitignore
|-- .gitmodules
`-- README.md
```
