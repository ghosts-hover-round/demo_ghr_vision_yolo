# RoboMaster 视觉组第一阶段考核

本仓库为 RoboMaster 2027 视觉组第一阶段考核提交。

## 📂 文件说明
- `train.py`：YOLO11n 模型训练脚本
- `convert.py`：将 `.pt` 模型导出为 `.onnx` 格式的脚本
- `data.yaml`：数据集配置文件（6个类别：blue3, blue1, bluesb, red3, red1, redsb）
- `best.pt` / `best.onnx`：训练完成的最佳模型权重及部署格式
- `results.png` / `confusion_matrix.png`：训练过程指标与混淆矩阵可视化

## 🛠️ 环境依赖
- OS: Windows 11 + WSL2
- PyTorch: 2.8.0+cu128 (NVIDIA RTX 5060)
- Ultralytics: 8.4.199
- Git LFS: 用于管理大模型文件
