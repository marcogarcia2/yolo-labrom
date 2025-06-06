# yolo-labrom

Scientific initiation project at LabRoM exploring YOLO's ability to segment transparent surfaces in images.

## LabRoM and Petrobras
This undergraduate research is conducted at LabRoM (Laboratório de Robótica Móvel), or the Mobile Robotics Laboratory, located in São Carlos, Brazil, at the University of São Paulo (USP). The lab is led by PhD. [Marcelo Becker](https://www.linkedin.com/in/marcelo-becker-761bb524/), and the research is supported by Petrobras, a Brazilian multinational corporation.

<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/EESC-LabRoM" style="text-decoration: underline;">
        <img src="photos/labrom.png" alt="LabRoM" height="275"><br>
        LabRoM
      </a>
    </td>
    <td align="center">
      <a href="https://petrobras.com.br/" style="text-decoration: underline;">
        <img src="photos/petrobras.png" alt="Petrobras" height="275"><br>
        Petrobras
      </a>
    </td>
  </tr>
</table>


## The Project

### Motivation
Autonomous robots rely on various sensors to navigate their environment safely. One of the most commonly used sensors today is LiDAR, a laser-based technology that generates a 3D map of the surroundings by measuring light reflection. However, a significant challenge with LiDAR is its difficulty in detecting transparent surfaces. Since light passes through these surfaces, LiDAR struggles to accurately map or identify them, which can lead to crashes and damage to the robot.

<p align="center">
  <img src="photos/lidar.gif" alt="LiDAR" width="550">
</p>
<p align="center">
  <a style="font-size: 12px; text-decoration: none; color: inherit;">
    Example of how LiDAR works.
  </a>
</p>

[Falar dos problemas da petrobrás.]

### Goals
This project aims to evaluate how YOLO (You Only Look Once), a deep learning real-time object detection model [1], performs in segmenting transparent surfaces. The goal is to integrate YOLO's capabilities with LiDAR to enhance the robot’s navigation system, enabling it to better detect and avoid transparent obstacles.


## YOLO: The State of the Art in Computer Vision
YOLO is currently one of the most widely used models for solving computer vision problems. Its versatility in training with custom datasets makes it particularly suitable for highly specific detection and segmentation tasks. For this reason, YOLO was chosen as the core model for this project.

### Models
For this reason, **YOLOv11**, the latest and most powerful version of the YOLO family, was chosen as the core model for this project. The specific task addressed here is **segmentation**, which involves identifying the exact pixels in an image that correspond to a given object. Among the available YOLO model variants, **Nano** and **Small** were selected due to their compact size and faster inference times, which are important factors for real-time or resource-constrained applications. So, the models that were trained in this project are called `yolo11n-seg.pt` and `yolo11s-seg.pt`

### Dataset
The dataset chosen for this project is Trans10K by Xie et al. [2]. It consists of 10,428 images of transparent objects and surfaces, including items such as cups, bowls, windows, doors, walls, and more. Additional information about the dataset and the project can be found on the [project's GitHub page](https://github.com/xieenze/Segment_Transparent_Objects).

<table align="center" style="border-collapse: collapse;">
  <tr>
    <td style="padding: 0;">
      <img src="photos/4.jpg" width="250" style="display: block; border: none; margin: 0;">
    </td>
    <td style="padding: 0;">
      <img src="photos/43.jpg" width="250" style="display: block; border: none; margin: 0;">
    </td>
    <td style="padding: 0;">
      <img src="photos/11.jpg" width="250" style="display: block; border: none; margin: 0;">
    </td>
  </tr>
  <tr>
    <td style="padding: 0;">
      <img src="photos/10282.jpg" width="250" style="display: block; border: none; margin: 0;">
    </td>
    <td style="padding: 0;">
      <img src="photos/57.jpg" width="250" style="display: block; border: none; margin: 0;">
    </td>
    <td style="padding: 0;">
      <img src="photos/6081.jpg" width="250" style="display: block; border: none; margin: 0;">
    </td>
  </tr>
  <tr>
    <td style="padding: 0;">
      <img src="photos/2670.jpg" width="250" style="display: block; border: none; margin: 0;">
    </td>
    <td style="padding: 0;">
      <img src="photos/5735.jpg" width="250" style="display: block; border: none; margin: 0;">
    </td>
    <td style="padding: 0;">
      <img src="photos/2.jpg" width="250" style="display: block; border: none; margin: 0;">
    </td>
  </tr>
</table>

<p style="text-align: center;">Some images of the Trans10K dataset showing different examples of transparent objects and surfaces.</p>


### Workflow 
Working with YOLO and neural networks typically follows a well-established pipeline. The workflow adopted in this project consists of the following stages:

| **#** | **Step** | **Description** |
|-------------|-------------|-------------|
| **1** | Acquire a dataset | Downloaded the Trans10K dataset [2], which contains 10,428 images of transparent objects and surfaces. |
| **2** | Annotate the dataset | The labels of the dataset were in a different format from what YOLO11 required, so it had to be converted into the appropriate format. |
| **3** | Split the dataset | After labelling, the dataset was split into train, validation and test sets, with the proportion of 80-10-10%, respectively. |
| **4** | Train models | Trained YOLO11seg models (Nano and Small) on the dataset. |
| **5** | Evaluate results | The results after training were compared and discussed, in order to choose the best model that fits the project's purpose. |
| **6** | Embed model | The chosen model was embedded on a real robot on an NVIDIA Jetson Orin %%%. |
<!-- | **7** | Sensor Fusion | The model's outputs were integrated with LiDAR, in order to assist its navigation on complex environments. | -->


## Running this project
1. Downoload the dataset
2. Create python environment
3. Run Jupyter Cells
4. Train and Predict on separate python scripts

### Downloading Dataset

### Python Environment

To set up the environment needed:

```shell
python3 -m venv yolo_env
source yolo_env/bin/activate
```

Installing all dependencies needed:
```shell
pip install -r requirements.txt
```

Exiting the environment:
```shell
deactivate
```

## References
- [1] G. Jocher and J. Qiu, “Ultralytics yolo11,” 2024. [Online]. Available: https:
//github.com/ultralytics/ultralytics
- [2] Xie, W. Wang, W. Wang, M. Ding, C. Shen, and P. Luo, “Segmenting transparent objects in the wild,” arXiv preprint arXiv:2003.13948, 2020.