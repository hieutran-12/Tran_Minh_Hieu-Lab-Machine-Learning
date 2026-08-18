#BÀI 3 — LỌC DETECTIONS NHƯ LỌC DATAFRAME

import cv2
import supervision as sv 
from ultralytics import YOLO

from display import show_frame1,close_windows

model = YOLO('yolov8n.pt')

generator = sv.get_video_frames_generator("vehicles.mp4")

image = next(generator)


label_annortor = sv.LabelAnnotator(
    text_scale = 1.5,
    text_thickness = 1,
)

box_annortor = sv.BoxAnnotator(thickness = 1)



for image in generator:
    results = model(image)[0]
    detection = sv.Detections.from_ultralytics(results)
    #lọc với độ tin cậy trên 0.5
    detection = detection[detection.confidence > 0.5 ]
    #lọc với class 2
    detections = detections[detections.class_id == 2]
    # giữ nhiều class 
    import numpy as np
    VEHICLE_CLASSES = [2, 3, 5, 7]
    detections = detections[np.isin(detections.class_id, VEHICLE_CLASSES)]
    labels = [
        f"{results.names[class_id]} {conf:.2f}"
        for class_id,conf in zip(
            detection.class_id,
            detection.confidence
        )
    ]
    annotated = image.copy()
    annotated = box_annortor.annotate(
        annotated,
        detections = detection,
    )
    annotated = label_annortor.annotate(
        annotated,
        detections = detection,
        labels = labels
    )
    
    if not show_frame1(annotated,wait = 1):
        break
close_windows()