from ultralytics import YOLO
import torch
import colorful as cf

def train_model(model_name: str, model_char: str, device: str):
    
    # Download the model if it doesn't exist
    model = YOLO(model_name)

    # Sends it to GPU if it is avaliable
    model.to(device)
    
    # Folder where the results will be saved
    save_folder = "runs/train_"  + model_char # char 's; or 'n'

    # Training model with the following hyperparameters
    
    model.train(
        data="dataset/data.yaml",   
        epochs=300,                 # max epochs
        patience=50,                # early stopping: stop if no improvement after 50 epochs
        batch=16,                   # batch size
        imgsz=640,                  # image size 
        lr0=0.001,                  # initial learning rate
        project=save_folder,        
        name="exp",                 
    )
    
    # Saving our trained model in models folder
    model = YOLO(save_folder + "/exp/weights/best.pt")  # loads the best
    save_name = "models/" + model_name[:-3] + "-trained.pt"
    model.save(save_name)


def main():

    # Checking GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(cf.bold_green("-------------------------------------"))
    print(cf.bold_white(" TRAINING YOLO11 SEGMENTATION MODELS"))
    print(cf.bold_green("-------------------------------------\n"))

    print(cf.bold_orange("Device: ") + cf.bold_white(device) + "\n")

    print(cf.bold_orange("N") + cf.white(" --> Nano"))
    print(cf.bold_orange("S") + cf.white(" --> Small"))
    print(cf.bold_orange("M") + cf.white(" --> Medium"))
    print(cf.bold_orange("L") + cf.white(" --> Large"))
    print(cf.bold_orange("X") + cf.white(" --> Extra Large\n"))
    
    # Taking user's input
    user_input = input(("Enter the model letters to train: "))

    # Treating the input
    models_to_train = "".join(set(user_input.replace(" ", "").upper()))

    # Training all selected models

    if ('N' in models_to_train):
        train_model('yolo11n-seg.pt', 'n', device)
    
    if ('S' in models_to_train):
        train_model('yolo11s-seg.pt', 's', device)
    
    if ('M' in models_to_train):
        train_model('yolo11m-seg.pt', 'm', device)
    
    if ('L' in models_to_train):
        train_model('yolo11l-seg.pt', 'l', device)
    
    if ('X' in models_to_train):
        train_model('yolo11x-seg.pt', 'x', device)

    
    print(cf.bold_green("\nTraining complete. Check the models folder to see the results."))
    


if __name__ == "__main__":
    main()
