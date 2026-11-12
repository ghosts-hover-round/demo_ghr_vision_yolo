# RoboMaster 视觉第二阶段：C++ 部署与中心点提取

本项目为 RoboMaster 2027 视觉组第二阶段考核任务。使用第一阶段训练并导出的 `best.onnx` 模型，结合 C++ 与 OpenCV DNN 模块完成装甲板识别、中心点提取、视频/图片流推理以及 FPS 统计。

## 🛠️ 环境依赖
- **操作系统**: Windows 11 + WSL2 (Ubuntu 22.04)
- **编译器**: g++ (支持 C++14 及以上)
- **构建工具**: CMake (>= 3.10)
- **视觉库**: OpenCV 4.10.0

## 📂 文件结构
```text
detect/
├── CMakeLists.txt      # CMake 构建脚本
├── main.cpp            # 核心推理与处理代码
├── best.onnx           # YOLO11n 导出的 ONNX 格式模型
├── test.jpg            # 测试用图片
├── result.jpg          # 推理结果可视化截图
└── README.md           # 说明文档