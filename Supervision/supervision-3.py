# BÀI 2 — VẼ ĐẸP VỚI ANNOTATORS

import cv2 
import supervision as sv 
from ultralytics import YOLO

from display import show_frame1, close_windows

model = YOLO('yolov8n.pt')

box_annortator = sv.BlurAnnotator()
label_annortanor = sv.LabelAnnotator(
        text_scale = 1.5,
        text_thickness = 1,
        text_position = sv.Position.TOP_LEFT, 
    )

generrator = sv.get_video_frames_generator('vehicles.mp4')

for image in generrator:
    results = model(image)[0]

    detections = sv.Detections.from_ultralytics(results)

    labels = [
        f"{results.names[class_id]} {conf:.2f}"
        for class_id,conf in zip(
            detections.class_id,
            detections.confidence
        )
    ]

    annotated = image.copy()
    annotated = box_annortator.annotate(
        annotated,
        detections = detections
    )

    annotated = label_annortanor.annotate(
        annotated,
        detections = detections,
        labels = labels
    )

    if not show_frame1(annotated,wait = 1):
        break
close_windows()