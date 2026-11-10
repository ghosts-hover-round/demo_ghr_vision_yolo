from ultralytics import YOLO
model = YOLO("D:/robotmaster/demo_ghr_vision_yolo/best.pt")
model.export(format="onnx")