# 后续实验入口

训练、测试优先复用 `third_party/mmsegmentation/tools/train.py` 和 `test.py`，
不另建训练框架。未来只在此增加必要的推理 PNG 导出、整图拼接辅助入口。
本轮无模型推理工具；辅助环境及部署工具位于 `scripts/`。
