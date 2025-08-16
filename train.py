from ultralytics import YOLO
import torch
import colorful as cf

def _print_hyperparameters(model_name: str, epochs, patience, batch, imgsz, lr0):
    print(cf.bold_green(f"\nTraining starting with Model: {model_name}"))
    print(cf.bold_white("Epochs:"), epochs)
    print(cf.bold_white("Patience:"), patience)
    print(cf.bold_white("Batch Size:"), batch)
    print(cf.bold_white("Image Size:"), imgsz)
    print(cf.bold_white("LR0:"), lr0)



def train_model(model_name: str, model_char: str, device: str,
                epochs=300, patience=50, batch=16, imgsz=640, lr0=0.001):
    
    try: 
        # Download the model if it doesn't exist
        model = YOLO(model_name)

        # Sends it to GPU if it is avaliable
        model.to(device)
        
        # Folder where the results will be saved
        save_folder = "runs/train_"  + model_char 

        # Training model with the following hyperparameters
        _print_hyperparameters(model_name, epochs, patience, batch, imgsz, lr0)
        
        model.train(
            data="dataset/data.yaml",   
            epochs=epochs,                 # max epochs
            patience=patience,                # early stopping: stop if no improvement after 50 epochs
            batch=batch,                   # batch size
            imgsz=imgsz,                  # image size 
            lr0=lr0,                  # initial learning rate
            amp=True,
            project=save_folder,        
            name="exp",                 
        )
        
        # Saving our trained model in models folder
        model = YOLO(save_folder + "/exp/weights/best.pt")  # loads the best
        save_name = "models/" + model_name[:-3] + "-trained.pt"
        model.save(save_name)
    
    except Exception as e:
        print(cf.bold_red(f"Exception at training {model_name}:"), e)


def main():

    # Checking GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(cf.bold_white(" TRAINING YOLO11 SEGMENTATION MODELS"))

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
    
    if 'N' in models_to_train:
        train_model('yolo11n-seg.pt', 'n', device) # nano


    if 'S' in models_to_train:
        train_model('yolo11s-seg.pt', 's', device) # small

    
    if 'M' in models_to_train:
        train_model('yolo11m-seg.pt', 'm', device) # medium


    if 'L' in models_to_train:
        train_model('yolo11l-seg.pt', 'l', device, batch=8, imgsz=512) # large


    if 'X' in models_to_train:
        train_model('yolo11x-seg.pt', 'x', device, epochs=400, batch=8, imgsz=512) # extra large


    
    print(cf.bold_green("\nTraining complete. Check the models folder to see the results."))
    


if __name__ == "__main__":
    main()
