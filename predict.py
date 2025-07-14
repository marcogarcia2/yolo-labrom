from ultralytics import YOLO
import torch
from pathlib import Path
import cv2
from tqdm import tqdm
import os

# This function reads a video file as a sequence of frames and saves a .mp4 
# that shows the predictions of both models
def make_predictions(model_path: str, source: str, device: str, conf: float):
    
    # Loading model and sending it to GPU
    model = YOLO(model_path)
    model.to(device)

    
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
    model_name_char = Path(model_path).stem.split('-')[0][-1]
    output_folder = Path("runs") / f"predict_{model_name_char}"
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
    print(f"\nProcesso concluído! Vídeo salvo em: {output_video_path}")



def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using: {device}\n")

    conf = 0.5
    source_folder = "my_test_images/"

    for file in os.listdir(source_folder):
        if file.endswith((".MOV", ".mov", ".MP4", ".mp4", ".avi", ".AVI")):
            
            source_video = source_folder + file

            print("--- Processing video with model YOLOv11n ---")
            make_predictions("models/yolo11n-seg-trained.pt", source_video, device, conf)

            print("\n--- Processing video with model YOLOv11s ---")
            make_predictions("models/yolo11s-seg-trained.pt", source_video, device, conf)

if __name__ == "__main__":
    main()