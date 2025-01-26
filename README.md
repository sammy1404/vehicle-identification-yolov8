#  Vehicle Detection using YOLOv8

This project utilizes the **YOLOv8** object detection model for vehicle detection on both static images and video feeds. The model is trained on a custom dataset of vehicle images and tested on individual images and videos to detect vehicles in real-time.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Setup](#setup)
- [Usage](#usage)
- [Testing the Model on Images](#testing-the-model-on-images)
- [Testing the Model on Video](#testing-the-model-on-video)
- [Results](#results)


---

## Overview

This project demonstrates the use of **YOLOv8**, a state-of-the-art object detection model, for detecting vehicles in images and videos. The following steps outline the key functionalities:

1. **Model Training**: Train a YOLOv8 model on a custom dataset of vehicle images.
2. **Object Detection**: Use the trained model to detect vehicles in static images.
3. **Object Tracking**: Apply the model to video input to track vehicles in real-time.


---

## Dataset

The dataset used in the given project is taken from kaggle:
https://www.kaggle.com/datasets/pkdarabi/vehicle-detection-image-dataset/code


---

## Setup

### Prerequisites

- Python 3.x
- PyTorch
- YOLOv8 (Ultralytics)
- OpenCV
- Matplotlib
- NumPy

---

## Usage

To set up the project, follow these steps:

1. Clone the repository or download the project files.

```bash
git clone https://github.com/sammy1404/car-image-segmentation.git
cd vehicle-detection-yolov8
```

2. Install the required libraries:
```bash
pip install torch opencv-python ultralytics matplotlib numpy
```

3. Change the path to your datasets in the notebook file
Change the path to the root directory such that it points to your dataset.

4. Run the jupyter notebook file
This will **train the model** for 200 epochs on the dataset defined in data.yaml.

5. Testing the Model on Images
change the file path and run **testPicture.ipynb**
<img width="752" alt="Screenshot 2025-01-26 at 9 18 55 PM" src="https://github.com/user-attachments/assets/1c0b0e79-3156-4644-93f4-200a9d45aba4" />


7. Testing the model on Videos
Change the file path and run **testVideo.py**

![1933AB7D-D952-4172-8D5F-7A60118B412F_1_206_a](https://github.com/user-attachments/assets/3a44dd37-2f5d-4cfa-b85a-b0310df2d108)

## Results
Once the detection is complete, the processed videos and images will be saved in the following directories:

runs/track/exp/ - Contains the video with detections.
runs/detect/exp/ - Contains images with detection results.
You can inspect the results in these directories or open the images and videos directly.
