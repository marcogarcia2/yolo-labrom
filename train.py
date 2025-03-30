from ultralytics import YOLO
import torch
import os

def main():

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
    model.train(
    data="dataset/data.yaml",     # caminho do seu arquivo data
    epochs=100,           # número máximo de épocas
    patience=10,          # early stopping: para se não melhorar por 10 épocas
    batch=16,             # tamanho de batch
    imgsz=640,            # resolução das imagens
    lr0=0.001,            # learning rate inicial
    project="runs/train", # pasta onde serão salvos os resultados
    name="exp",           # nome do experimento
)

    model = YOLO("runs/segment/train/weights/best.pt")  # carrega o best
    save_name = "models/" + model_name[:-3] + "-trained.pt"
    model.save(save_name)

if __name__ == "__main__":
    main()
