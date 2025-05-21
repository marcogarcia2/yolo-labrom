from ultralytics import YOLO
import torch

def make_predictions(model_path: str, source: str, device: str):
    
    # Loading model from path
    model = YOLO(model_path)
    model.to(device)
    
    # Predictions
    results = model.predict(
        source=source,
        save=True,
        save_txt=True,
        project="runs",                         
        name="predict_" + model_path[13],       
        exist_ok=True                              
    )


def main():

    # Checking GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Source folder
    source = "my_test_images/"
    
    # Using custom function to make predictions with our trained model N on the available device
    make_predictions("models/yolo11n-seg-trained.pt", source, device)
    
    # Same for model S
    make_predictions("models/yolo11s-seg-trained.pt", source, device)

    

if __name__ == "__main__":
    main()
