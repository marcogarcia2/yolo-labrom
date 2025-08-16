from ultralytics import YOLO
import colorful as cf
import cv2
from pathlib import Path
import torch
import time

def real_time_inference(letter: str, model_path: Path, gpu: bool=False):
    
    # Display which model was selected
    print(cf.bold_white("\nSelected model:"), cf.bold_cyan(f"YOLO11seg-{letter}"))
    model = YOLO(model_path)
    
    # Select device (GPU or CPU)
    device = 'cuda' if gpu and torch.cuda.is_available() else 'cpu'
    model.to(device)
    print("The model is loaded on", cf.bold_green(device))
    
    # Start capturing from camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error opening camera.")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Frames per second
    ti = time.time()
    fps = 0.0

    # Main loop
    while True:

        ok, frame = cap.read()
        if not ok:
            print("Error reading frame from camera.")
            break
        
        # YOLO Model Inference 
        results = model.predict(
            frame,
            imgsz=512,
            conf=0.75,
            iou=0.45,
            stream=False,   # True = gerador, False = retorna lista; aqui 1 frame só
            verbose=False,
        )

        annotated_frame = results[0].plot()

        # Calc. fps
        tf = time.time()
        dt = tf - ti
        ti = tf
        fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # Show FPS on screen
        cv2.putText(annotated_frame, f"FPS: {fps:5.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Show 
        cv2.imshow(f"YOLO11{letter} realtime segmentation", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():

    models_folder = Path("models")
    models_dict = {
        'N': models_folder / "yolo11n-seg-trained.pt",
        'S': models_folder / "yolo11s-seg-trained.pt",
        'M': models_folder / "yolo11m-seg-trained.pt",
        'L': models_folder / "yolo11l-seg-trained.pt",
        'X': models_folder / "yolo11x-seg-trained.pt"
    }

    # Verify if the expected models exist
    for k, path in models_dict.items():
        if not path.exists():
            print(f"[MISSING] {k}: not found -> {path}")

    # Getting model from user's input
    print(cf.bold_white("SELECT MODEL LETTER TO START INFERENCE:"))
    print(cf.bold_orange("N") + cf.white(" --> Nano"))
    print(cf.bold_orange("S") + cf.white(" --> Small"))
    print(cf.bold_orange("M") + cf.white(" --> Medium"))
    print(cf.bold_orange("L") + cf.white(" --> Large"))
    print(cf.bold_orange("X") + cf.white(" --> Extra Large\n"))
    user_input = input()
    letter = user_input.strip().upper()[0]
    
    # Starts inference
    if letter in models_dict:
        real_time_inference(letter, models_dict[letter], gpu=True)

    else: 
        print(f"Model not found: {letter}")


if __name__ == "__main__":
    main()