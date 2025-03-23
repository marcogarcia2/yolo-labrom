from ultralytics import YOLO
import torch
import os

# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 8
IMG_SIZE = 640
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Downloading our model
model = YOLO("yolov8n-seg.pt")

# Using GPU if it is available
model.to(device)
print(f"The model is at {device}.")

# Training model on dataset
model.train(data="dataset/data.yaml", epochs=EPOCHS, batch=BATCH_SIZE, imgsz=IMG_SIZE)

# Salvar o modelo treinado
model.save("models/yolov8n-seg_trained_0.pt")