import cv2
import torch
import numpy as np
from pathlib import Path
import json
from ultralytics import YOLO
import os
from segment_anything_2 import SamModel, SamPredictor, get_image_embedding
import natsort # Biblioteca para ordenação natural de nomes de arquivos

# Instale natsort se não tiver: pip install natsort

# Função auxiliar para converter máscara em polígono (mesma de antes)
def mask_to_coco_poly(mask):
    """Converte uma máscara binária em um polígono no formato COCO."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
    return segmentation


# Uses the YOLO model's predictions to generate annotations
def generate_sam2_annotations_from_yolo_model(
    frames_dir: str,
    yolo_model_path: str,
    sam_checkpoint_path: str,
    sam_config_path: str,
    output_json_path: str,
    confidence_threshold: float = 0.5
):

    # Getting the GPU to work
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Loading YOLO Model
    print("Loading YOLO model...")
    yolo_model = YOLO(yolo_model_path)
    
    # Loading SAM 2
    print("Loading SAM 2...")
    sam_model = SamModel(config=sam_config_path, checkpoint=sam_checkpoint_path).to(device)
    sam_predictor = SamPredictor(sam_model)

    # Setting up frames directory
    frames_path = Path(frames_dir)
    if not frames_path.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {frames_dir}")

    # Gets all .png files and sorts them 
    image_files = natsort.natsorted([p for p in frames_path.glob("*.png")])
    
    # Categories
    categories = [{"id": int(k), "name": v, "supercategory": ""} for k, v in yolo_model.names.items()]


    # Basic structure for COCO JSON file
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    annotation_id = 0

    # Main loop for processing image files 
    for frame_id, image_path in enumerate(image_files):
        print(f"Processing frame {frame_id + 1}/{len(image_files)}: {image_path.name}")

        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"  [WARNING] Couldn't read image: {image_path.name}. Skipping.")
            continue

        # Adds image info to the COCO dict
        height, width, _ = frame.shape
        coco_output["images"].append({
            "id": frame_id,
            "width": width,
            "height": height,
            "file_name": os.path.join(frames_path.name, image_path.name)

        })

        # Uses YOLO model for prediction
        yolo_results = yolo_model(frame, conf=confidence_threshold, verbose=False)

        # For each predicted object in the frame
        for result in yolo_results:
            if result.masks is None:
                continue

            for i in range(len(result.masks)):
                yolo_mask_tensor = result.masks.data[i]
                
                yolo_mask_np = yolo_mask_tensor.cpu().numpy().astype(np.uint8)
                yolo_mask_resized = cv2.resize(yolo_mask_np, (width, height))

                image_embedding = get_image_embedding(sam_model, frame)
                
                sam_input = {
                    "image_embedding": image_embedding,
                    "input_mask": torch.from_numpy(yolo_mask_resized).to(device).unsqueeze(0),
                    "hq_token_only": False,
                }

                sam_output = sam_model.predict_torch(queries=[sam_input], multimask_output=False)
                
                refined_mask = sam_output["masks"][0, 0].cpu().numpy() > 0.5
                refined_mask_uint8 = refined_mask.astype(np.uint8)

                segmentation = mask_to_coco_poly(refined_mask_uint8)
                if not segmentation:
                    continue

                x_coords, y_coords = np.where(refined_mask)[1], np.where(refined_mask)[0]
                x, y, w, h = int(min(x_coords)), int(min(y_coords)), int(max(x_coords) - min(x_coords)), int(max(y_coords) - min(y_coords))

                coco_output["annotations"].append({
                    "id": annotation_id,
                    "image_id": frame_id,
                    "category_id": int(result.boxes.cls[i]),
                    "segmentation": segmentation,
                    "area": int(np.sum(refined_mask_uint8)),
                    "bbox": [x, y, w, h],
                    "iscrowd": 0,
                })
                annotation_id += 1

    # 4. Salvar o arquivo JSON final
    print(f"Saving refined annotations in {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(coco_output, f, indent=2)



# Combine all annotations into one single YOLO dataset
def merge_datasets(json_files: list[str], output_filename: str):

    print(f"\n--- Starting the unification of {len(json_files)} datasets ---")
    
    final_coco = {
        "images": [],
        "annotations": [],
        "categories": [] 
    }

    current_image_id_offset = 0
    current_annotation_id_offset = 0

    for json_file in json_files:
        print(f"Processando arquivo: {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Gather all categories
        if not final_coco["categories"]:
            final_coco["categories"] = data["categories"]

        image_id_map = {}

        # Process and re-index the images
        for image in data["images"]:
            old_id = image["id"]
            new_id = old_id + current_image_id_offset
            image_id_map[old_id] = new_id
            
            image["id"] = new_id
            final_coco["images"].append(image)
        
        # Process and re-index annotations
        for annotation in data["annotations"]:
            annotation["image_id"] = image_id_map[annotation["image_id"]]
            annotation["id"] += current_annotation_id_offset
            final_coco["annotations"].append(annotation)
        
        # Updates the offset 
        current_image_id_offset += len(data["images"])
        current_annotation_id_offset += len(data["annotations"])
    
    # Saves the final unified file
    with open(output_filename, 'w') as f:
        json.dump(final_coco, f, indent=2)
    
    print("\n--- Unification Concluded ---")
    print(f"Total images: {len(final_coco['images'])}")
    print(f"Total annotations: {len(final_coco['annotations'])}")
    print(f"Dataset save at: {output_filename}")



def main():
    
    # Using model S because it has greater precision
    YOLO_MODEL = "models/yolo11s-seg-trained.pt"
    SAM_CHECKPOINT = "caminho/para/seu/sam2.1_hiera_large.pt"
    SAM_CONFIG = "caminho/para/o/repo_sam/configs/sam2.1/sam2.1_hiera_l.yaml"
    
    # All directories that contains the images from all four videos
    frame_directories = [
        "spot_datasets/FrontLeftEye",
        "spot_datasets/FrontRightEye",
        "spot_datasets/LeftEye",
        "spot_datasets/RightEye",
    ]

    generated_json_files = []

    # Process all directories listed above
    for directory in frame_directories:
        print(f"\n--- Processing directory: {directory} ---")
        output_name = Path(directory).name + "_annotations.json" # Ex: FrontLeftEye_annotations.json
        generated_json_files.append(output_name)

        generate_sam2_annotations_from_yolo_model(
            frames_dir=directory,
            yolo_model_path=YOLO_MODEL,
            sam_checkpoint_path=SAM_CHECKPOINT,
            sam_config_path=SAM_CONFIG,
            output_json_path=output_name
        )
    
    # Combine all processed datasets into one for fine tuning
    merge_datasets(generated_json_files, "spot_datasets/merged_spot_annotations.json")


if __name__ == '__main__':
    main()
