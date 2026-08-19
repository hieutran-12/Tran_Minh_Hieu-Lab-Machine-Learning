import cv2 
from ultralytics import YOLO
import supervision as sv 
import numpy as np

from display import show_frame1,close_windows

SOURCE_VIDEO = "vehicles.mp4"
TARGET_VIDEO = "bai4_output.mp4"
VEHICLE_CLASSES = [2, 3, 5, 7]

model = YOLO("yolov8n.pt")

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO)

box_annortor = sv.BoxAnnotator(thickness = 2)
label_annortor = sv.LabelAnnotator(
    text_scale = 1.5,
    text_thickness = 1
)

def process_frame(frame: np.ndarray) -> np.ndarray:
    results = model(frame,verbose = False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    detections = detections[(detections.confidence > 0.3) & np.isin(detections.class_id,VEHICLE_CLASSES)]
    labels = [
        f"{results.names[class_id]} {conf:.2f}"
        for class_id,conf in zip(
            detections.class_id,
            detections.confidence
        )
    ]
    annotated = frame.copy()
    annotated = box_annortor.annotate(
        annotated,
        detections
    )
    annotated = label_annortor.annotate(
        annotated,
        detections,
        labels
    )
    return annotated

with sv.VideoSink(target_path = TARGET_VIDEO,video_info = video_info) as sink:
    for frame in sv.get_video_frames_generator(SOURCE_VIDEO):
        annotated = process_frame(frame)
        sink.write_frame(annotated)
        if not show_frame1(annotated):
            print('Nguoi dung nhan Q hoac q')
            break
close_windows()
print('Da luu video tai',TARGET_VIDEO)