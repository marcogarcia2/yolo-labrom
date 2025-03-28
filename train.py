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

    model_name = ""
    while True:
        version = input("Type n or s: ")
        if version in ['n', 'N']:
            model_name = "yolo11n-seg.pt"
            break
        elif version in ['s', 'S']:
            model_name = "yolo11s-seg.pt"
            break
        else:
            print("Invalid entry.")

    # Downloading our model
    model = YOLO(model_name)

    # Using GPU if it is available
    model.to(device)
    print(f"The model is at {device}.")

    # Training model on dataset
    model.train(data="dataset/data.yaml", epochs=EPOCHS, batch=BATCH_SIZE, imgsz=IMG_SIZE, patience=PATIENCE)

    model = YOLO("runs/segment/train/weights/best.pt")  # carrega o best
    save_name = "models/" + model_name[:-3] + "-trained.pt"
    model.save(save_name)

if __name__ == "__main__":
    main()
