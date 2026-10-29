# 08 当前Lightning主工程目录

任务02工作树包含自有文件43个、SAM2 569个、GeoSeg 67个受控文件；官方源码合计636个，当前源码文件合计679个。

源码清单按当前迁移工作树统计；正式导出数用干净提交生成的manifest重新核验，不沿用任务01三子模块计数。

完整展开见[full_directory_tree.txt](full_directory_tree.txt)，官方文件路径见[source_inventory.txt](source_inventory.txt)。排除.git、缓存及运行产物；不把历史.git/modules缓存当当前源码。

```text
SAM2-RS-Seg/
|-- configs/
|   |-- loveda/
|   |   `-- .gitkeep
|   |-- potsdam/
|   |   `-- hiera_tiny_lightning.plan.json
|   |-- vaihingen/
|   |   `-- .gitkeep
|   `-- README.md
|-- data/
|   `-- [ignored runtime]
|-- datasets/
|   |-- __init__.py
|   |-- label_contract.py
|   `-- README.md
|-- deploy/
|   `-- [ignored runtime]
|-- docs/
|   |-- 01_architecture.md
|   |-- 02_environment_compatibility.md
|   |-- 03_gpu_setup_plan.md
|   |-- 04_dataset_plan.md
|   |-- 05_implementation_steps.md
|   |-- 06_deployment.md
|   |-- 07_validation.md
|   |-- 08_directory_tree.md
|   |-- 09_lightning_migration.md
|   |-- full_directory_tree.txt
|   `-- source_inventory.txt
|-- outputs/
|   `-- [ignored runtime]
|-- projects/
|   |-- sam2_rs_seg/
|   |   |-- datasets/
|   |   |   `-- __init__.py
|   |   |-- engine/
|   |   |   |-- __init__.py
|   |   |   `-- README.md
|   |   |-- evaluation/
|   |   |   |-- __init__.py
|   |   |   `-- README.md
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
|   |-- test_initialization.py
|   `-- test_lightning_migration.py
|-- third_party/
|   |-- GeoSeg/
|   |   `-- [pinned official source]
|   |-- sam2/
|   |   `-- [pinned official source]
|   `-- versions.json
|-- tools/
|   `-- README.md
|-- weights/
|   `-- [ignored runtime]
|-- .gitattributes
|-- .gitignore
|-- .gitmodules
`-- README.md
```
