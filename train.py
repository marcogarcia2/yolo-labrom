from ultralytics import YOLO
import torch
import os

def main():
    # Hyperparameters
    EPOCHS = 50
    BATCH_SIZE = 8
    IMG_SIZE = 640
    PATIENCE = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Downloading our model
    model = YOLO("yolo11s-seg.pt")

    # Using GPU if it is available
    model.to(device)
    print(f"The model is at {device}.")

    # Training model on dataset
    model.train(data="dataset/data.yaml", epochs=EPOCHS, batch=BATCH_SIZE, imgsz=IMG_SIZE, patience=PATIENCE)

    # Salvar o modelo treinado
    model.save("models/yolo11n-seg_trained_0.pt")

if __name__ == "__main__":
    main()
