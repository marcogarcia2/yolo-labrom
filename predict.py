from ultralytics import YOLO
import torch
from pathlib import Path
import cv2
from tqdm import tqdm
import os
import colorful as cf

img_ext = (".png", ".PNG",
           ".jpg", ".JPG",
           ".jpeg", ".JPEG")

video_ext = (".MOV", ".mov",
             ".MP4", ".mp4",
             ".avi", ".AVI")

# Make predictions for all files in source folder, given a model
def make_predictions(model_path: str, source_folder: str, device: str, conf: float):
    
    # Loading model and sending it to GPU
    model = YOLO(model_path)
    model.to(device)
    model_char = Path(model_path).stem.split('-')[0][-1]
    print(cf.bold_yellow(f"--- Using YOLOv11{model_char} ---\n"))
    
    for file in os.listdir(source_folder):
        
        # If it is a video
        if file.endswith((video_ext)):
            source_video = source_folder + file
            make_video_predictions(model, model_char, source_video, conf)
        
        # If it is an image
        elif file.endswith((img_ext)):
            source_img = source_folder + file
            make_img_predictions(model, model_char, source_img, conf)

    print(cf.bold_yellow("Done.\n\n"))


## This function makes predictions on a single image
def make_img_predictions(model: YOLO, model_char: str, source: str, conf: float):
    results = model.predict(
        source=source,
        project="runs",
        name="predict_" + model_char,
        conf=conf, 
        save=True,
        exist_ok=True,
        verbose=True
    )


# This function reads a video file as a sequence of frames and saves a .mp4 
# that shows the predictions of both models
def make_video_predictions(model: YOLO, model_char: str, source: str, conf: float):

    # Getting info about the video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Couldn't open source file: {source}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Defining paths
    output_folder = Path("runs") / f"predict_{model_char}"
    output_folder.mkdir(parents=True, exist_ok=True) # Cria a pasta de saída
    output_video_path = output_folder / (Path(source).stem + ".mp4")

    # Defining codec 'mp4v' for .mp4 file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print(f"Error: Failed to initialize the VideoWriter for: {output_video_path}")
        return

    results_generator = model.predict(
        source=source,
        conf=conf,
        stream=True,  # Processa o vídeo como um fluxo de frames
        save=False,    # IMPORTANTE: Desligamos o salvamento automático do YOLO
        verbose=False
    )

    # Usamos tqdm para criar uma barra de progresso
    for result in tqdm(results_generator, total=total_frames, desc="Prediction Progress"):
        annotated_frame = result.plot()
        
        # Writes the annotated frame in the .mp4 file
        video_writer.write(annotated_frame)

    # Releases memory and saves the video
    video_writer.release()
    print(f"Process concluded! Video saved in: {output_video_path}\n")



def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using: {device}\n")

    model_paths = ["models/yolo11n-seg-trained.pt",
                   "models/yolo11s-seg-trained.pt"]
    
    conf = 0.5
    source_folder = "my_test_images/"

    for model_path in model_paths:
        make_predictions(model_path, source_folder, device, conf)

if __name__ == "__main__":
    main()