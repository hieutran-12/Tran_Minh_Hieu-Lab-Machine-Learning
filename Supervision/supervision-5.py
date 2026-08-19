#BÀI 4 — XỬ LÝ VIDEO: PIPELINE CHUẨN CÓ LIVE PREVIEW
import supervision as sv 
import numpy as np 
from ultralytics import YOLO

from display import show_frame1,close_windows
VIDEO_PATH = 'vehicles.mp4'


# 1 ĐỌc thông tin vidoe 
video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
print(video_info)

frame_generator = sv.get_video_frames_generator(VIDEO_PATH)

# ghi video output

with sv.VideoSink(target_path="output.mp4",video_info=video_info) as sink:
    for frame in frame_generator:
        sink.write_frame(frame)