from ultralytics import YOLO
from pathlib import Path
import torch
import colorful as cf

# A função de fine-tuning agora recebe o caminho para a pasta de resultados do treino anterior
def fine_tune_model(previous_training_path: str, model_char: str, device: str):

    model_path = Path(previous_training_path) / "weights/best.pt"
    
    if not model_path.exists():
        print(f"ERROR: Could not find model at: {model_path}")
        return

    model = YOLO(model_path)

    # Ensuring that the loaded model is the one that was trained previously
    print(cf.bold_yellow(f"Modelo carregado de: {model.ckpt_path}"))

    # Sending model to GPU
    model.to(device)
    print(f"The model is at " +  cf.bold_green(f"{device}") + ".")
    
    save_folder = "runs/fine_tune_" + model_char

    # Fine Tuning training
    model.train(
        data="fine_tuning/data.yaml", 
        epochs=300,                 
        patience=50,
        batch=16,
        imgsz=640,
        lr0=0.0001,                 # Smaller initial learning rate for fine tuning
        project=save_folder,        
        name="exp",                 
    )
    
    # Saves the best model
    best_tuned_model_path = Path(save_folder) / "exp/weights/best.pt"
    save_name = f"models/yolo11{model_char}-seg-fine-tuned.pt"
    final_model = YOLO(best_tuned_model_path)
    final_model.save(save_name)


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Fine-tuning for model N
    fine_tune_model("runs/train_n/exp", 'n', device)
    
    # Fine-tuning for model S
    fine_tune_model("runs/train_s/exp", 's', device)

if __name__ == "__main__":
    main()