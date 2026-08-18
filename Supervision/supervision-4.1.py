import cv2 
import supervision as sv 
from ultralytics import YOLO

from display import show_frame1,close_windows

import numpy as np

#1 load model yolo

model = YOLO('yolov8n.pt')

#2 tạo bộ máy cung cấp frame 

generator = sv.get_video_frames_generator("vehicles.mp4")
image = next(generator)

#3 chạy inference
results = model(image)[0]

# thông tin các vật thể phát hiện và chuyển sang dạng supervision hiểu được
detections = sv.Detections.from_ultralytics(results)

print(detections)
# Chỉ giữ đối tượng confidence > 0.5
detections = detections[detections.confidence > 0.5]

# Chỉ giữ xe hơi (COCO class 2)
detections = detections[detections.class_id == 2]

# Giữ nhiều class: car, motorcycle, bus, truck
VEHICLE_CLASSES = [2, 3, 5, 7]

detections = detections[np.isin(detections.class_id, VEHICLE_CLASSES)]

# Kết hợp nhiều điều kiện
detections = detections[
    (detections.confidence > 0.4) & np.isin(detections.class_id, VEHICLE_CLASSES)
]

# Lọc theo diện tích box (loại box quá nhỏ = nhiễu)
detections = detections[detections.area > 1000]

box_annotator = sv.BoxAnnotator(thickness=2)

before = box_annotator.annotate(image.copy(), detections)                 # chưa lọc
filtered = detections[(detections.confidence > 0.4)
                      & np.isin(detections.class_id, VEHICLE_CLASSES)]
after = box_annotator.annotate(image.copy(), filtered)                    # đã lọc

cv2.putText(before, f"TRUOC LOC: {len(detections)}", (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
cv2.putText(after, f"SAU LOC: {len(filtered)}", (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

side_by_side = np.hstack([before, after])   # ghép ngang 2 ảnh
show_frame1(side_by_side, windows_name="Truoc vs Sau khi loc", wait=0)
close_windows()


# vẽ ảnh box bao quanh vật thể phát hiện 
annotated = sv.BoxAnnotator(thickness=2).annotate(
    image.copy(),
    detections
)
label = sv.LabelAnnotator(text_scale=1.5).annotate(
    annotated,
    detections=detections,
    labels= [
        f"{results.names[class_id]} {confidence:.2f}"
        for class_id,confidence in zip(
            detections.class_id,
            detections.confidence
        )
    ]
)
cv2.putText(annotated,f"Phat hien: {len(detections)} Doi tuong",
            (20,60),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),3)

show_frame1(label,wait=0)
close_windows()

print("Số đối tượng:", len(detections))
print("xyxy (tọa độ box):\n", detections.xyxy)        # ndarray (N, 4)
print("confidence:", detections.confidence)             # ndarray (N,)
print("class_id:", detections.class_id)                 # ndarray (N,)
print("tên class:", detections.data["class_name"])      # ndarray (N,) dạng chuỗi
