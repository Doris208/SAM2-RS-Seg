# 后续实验入口（尚未实现）

未来在当前工程添加少量入口，统一构建lightning.pytorch.Trainer：
训练、验证、测试及预测。核心业务留在projects/sam2_rs_seg/，不再复制第二套训练循环。

模型forward输出K通道logits；predict返回独立类别ID与RGB可视化；
未来滑窗在logits域融合后argmax。本轮没有可运行训练或推理入口。
现有辅助工具在scripts/，默认环境盘点不导入深度学习依赖。
