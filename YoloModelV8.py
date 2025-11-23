from ultralytics import YOLO


model = YOLO("runs/detect/MyYoloV8Models/weights/last.pt")

print(model.info())


model.train(
    data = "SafetyVestsDataset/data.yaml",
    epochs = 6,
    imgsz=640,
    batch = 16,
    device = 0,
    name="MyYoloV8Models"
)