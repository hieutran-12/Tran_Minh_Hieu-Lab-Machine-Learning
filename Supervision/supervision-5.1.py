import cv2 
from ultralytics import YOLO
import supervision as sv 

from display import show_frame1,close_windows

SOURCE_VIDEO = "vehicles.mp4"
TARGET_VIDEO = "bai4_output.mp4"
VEHICLE_CLASSES = [2, 3, 5, 7]

model = YOLO("yolov8n.pt")
