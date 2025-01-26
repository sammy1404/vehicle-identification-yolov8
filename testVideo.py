from ultralytics import YOLO

model = YOLO("/Users/samsusikar/Downloads/car image segmentation/runs/detect/train/weights/best.pt")  

results = model.track(
    source="/Users/samsusikar/Downloads/car image segmentation/archive/Sample_Video_HighQuality.mp4",  
    save=True,                 
    show=True                 
)
