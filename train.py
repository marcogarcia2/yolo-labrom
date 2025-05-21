from ultralytics import YOLO
import torch
import os

def train_model(model_name: str, device: str):
    
    # Downloading the model if it doesn't exist.
    model = YOLO(model_name)

    # Sending it to GPU if it is avaliable.
    model.to(device)
    print(f"The model is at {device}.")
    
    # Folder where the results will be saved
    save_folder = "runs/train_"  + model_name[6] # char 's; or 'n'

    # Training model based on these hyperparameters
    
    model.train(
        data="dataset/data.yaml",   # caminho do seu arquivo data
        epochs=300,                 # número máximo de épocas
        patience=50,                # early stopping: para se não melhorar por 10 épocas
        batch=16,                   # tamanho de batch
        imgsz=640,                  # resolução das imagens
        lr0=0.001,                  # learning rate inicial
        project=save_folder,        # pasta onde serão salvos os resultados
        name="exp",                 # nome do experimento
    )
    
    # Saving our trained model in models folder
    model = YOLO(save_folder + "/exp/weights/best.pt")  # carrega o best
    save_name = "models/" + model_name[:-3] + "-trained.pt"
    model.save(save_name)


def main():

    # Checking GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Using custom function to train model N on the available device
    train_model('yolo11n-seg.pt', device)
    
    # Same for model S
    train_model('yolo11s-seg.pt', device)

    

if __name__ == "__main__":
    main()
