# 实验配置入口（待实现）

后续配置使用 MMSegmentation 1.x 的 Python 配置继承与 Registry：

- `custom_imports = dict(imports=['projects.sam2_rs_seg'], allow_failed_imports=False)`。
- `default_scope = 'mmseg'`，训练/验证/测试统一用 MMEngine Runner。
- 第一份完整配置面向 Potsdam RGB、Hiera Tiny、原生简单多尺度解码头。
- 所有数据路径、类别表、ignore_index、标签协议必须显式记录。

当前只有目录占位，没有可运行训练配置；不要把本目录说明当作模型集成完成。
