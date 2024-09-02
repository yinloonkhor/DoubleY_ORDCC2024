from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="dataset/data.yaml", epochs=50, task='detect', batch=16, verbose=False)  # train the model
