from ultralytics import YOLO
from pathlib import Path
import torch

def fine_tune_model(model_name: str, device: str):
    
    # Loading the model from models folder
    model_path = Path("models") / model_name
    model = YOLO(model_path)

    # Sending it to GPU if it is avaliable
    model.to(device)
    print(f"The model is at {device}.")
    
    # Folder where the results will be saved
    save_folder = "runs/fine_tune_"  + model_name[6] # char 's; or 'n'

    # Tuning model based on these hyperparameters
    
    model.train(
        data="fine_tuning/data.yaml",   
        epochs=300,                 # max epochs
        patience=50,                # early stopping: if it doesn't get better in 10 epochs
        batch=16,                   # batch size
        imgsz=640,                  # image size 
        lr0=0.001,                  # initial learning rate
        project=save_folder,        
        name="exp",                 
    )
    
    # Saving our tuned model in models folder
    model = YOLO(save_folder + "/exp/weights/best.pt")  # loads the best
    save_name = "models/" + model_name[:-11] + "-fine-tuned.pt"
    model.save(save_name)


def main():

    # Checking GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Using custom function to fine tune model N on the available device
    fine_tune_model('yolo11n-seg-trained.pt', device)
    
    # Same for model S
    fine_tune_model('yolo11s-seg-trained.pt', device)

    

if __name__ == "__main__":
    main()
