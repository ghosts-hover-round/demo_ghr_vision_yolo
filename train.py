from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    # Windows 下训练 YOLO 必须加上这行，防止子进程重复执行主程序
    multiprocessing.freeze_support()

    # 加载模型并开始训练
    model = YOLO("yolo11n.pt")
    model.train(data="D:/robotmaster/rm_dataset/data.yaml", epochs=100, imgsz=640)