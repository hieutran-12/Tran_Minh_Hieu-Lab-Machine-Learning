# GIÁO TRÌNH SUPERVISION (Roboflow)
## Từ số 0 đến sản phẩm hoàn chỉnh: Hệ thống giám sát giao thông thông minh

> **Phiên bản thư viện:** supervision ≥ 0.26 (mới nhất hiện tại: 0.29.x)
> **Yêu cầu đầu vào:** Python cơ bản, biết sơ về OpenCV và YOLO là lợi thế
> **Sản phẩm cuối khóa:** Hệ thống đếm xe + phân loại + đo tốc độ + thống kê theo vùng, chạy trên video thực tế
> **🖥️ Điểm khác biệt của giáo trình này:** MỌI bài đều hiển thị kết quả **trực tiếp lên cửa sổ** (`cv2.imshow`) — vừa chạy vừa thấy ngay box, số đếm, tốc độ trên màn hình. Không còn kiểu chạy xong phải đi mở file output mới biết kết quả!

---

## MỤC LỤC

1. [Bài 0 — Supervision là gì, cài đặt, và quy ước hiển thị cửa sổ](#bài-0)
2. [Bài 1 — Đối tượng trung tâm: `sv.Detections`](#bài-1)
3. [Bài 2 — Vẽ đẹp với Annotators](#bài-2)
4. [Bài 3 — Lọc detections như lọc DataFrame](#bài-3)
5. [Bài 4 — Xử lý video: pipeline chuẩn có live preview](#bài-4)
6. [Bài 5 — Tracking đa đối tượng với ByteTrack](#bài-5)
7. [Bài 6 — Đếm đối tượng qua vạch: LineZone](#bài-6)
8. [Bài 7 — Giám sát theo vùng: PolygonZone](#bài-7)
9. [Bài 8 — Đo tốc độ với ViewTransformer](#bài-8)
10. [Bài 9 — SẢN PHẨM CUỐI: Traffic Monitor hoàn chỉnh](#bài-9)
11. [Bài 10 — Nâng cao: InferenceSlicer, Dataset, Metrics](#bài-10)
12. [Bài tập & lộ trình tự luyện](#bài-tập)

---

<a name="bài-0"></a>
## BÀI 0 — SUPERVISION LÀ GÌ, CÀI ĐẶT, VÀ QUY ƯỚC HIỂN THỊ CỬA SỔ

### 0.1. Vị trí của supervision trong pipeline CV

Supervision **không phải model**. Nó là bộ công cụ "hậu kỳ" nằm **sau** model detection:

```
Ảnh/Video → Model (YOLO, RT-DETR, RF-DETR...) → raw output
                                                    ↓
                              ┌─────────────────────────────────────┐
                              │           SUPERVISION               │
                              │  • Chuẩn hóa output → sv.Detections │
                              │  • Lọc / xử lý (NMS, filter...)     │
                              │  • Tracking (ByteTrack)             │
                              │  • Đếm (LineZone, PolygonZone)      │
                              │  • Vẽ (20+ loại Annotator)          │
                              │  • Dataset & Metrics (mAP, F1...)   │
                              └─────────────────────────────────────┘
                                                    ↓
                                    Cửa sổ live preview + file output
```

**Tại sao dùng supervision thay vì tự viết OpenCV?**
- Code tự viết để vẽ box + label + track + đếm xe dễ lên tới 300–500 dòng, nhiều bug.
- Supervision gói tất cả thành API thống nhất, **model-agnostic** (đổi YOLO sang RT-DETR không phải sửa pipeline).
- Được Roboflow maintain, cập nhật liên tục, MIT license.

### 0.2. Cài đặt

```bash
# Tạo môi trường ảo (khuyến nghị Python 3.10+)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install supervision ultralytics
```

Kiểm tra:

```python
import supervision as sv
import ultralytics

print(sv.__version__)          # >= 0.26.0
ultralytics.checks()
```

### 0.3. Chuẩn bị dữ liệu học

Supervision có sẵn bộ video/ảnh mẫu:

```bash
pip install "supervision[assets]"
```

```python
from supervision.assets import download_assets, VideoAssets

# Tải video đường cao tốc để dùng xuyên suốt giáo trình
download_assets(VideoAssets.VEHICLES)
print(VideoAssets.VEHICLES.value)  # "vehicles.mp4"
```

> 💡 Nếu không tải được asset, dùng bất kỳ video giao thông nào quay từ camera cố định (quan trọng: **camera không rung** để bài tracking và đo tốc độ chính xác).

### 0.4. ⭐ QUY ƯỚC HIỂN THỊ CỬA SỔ — dùng cho TẤT CẢ các bài

Video mẫu `vehicles.mp4` có độ phân giải rất lớn (3840×2160), hiện nguyên khổ sẽ tràn màn hình. Ta viết **một hàm hiển thị dùng chung**, lưu vào file `display.py` — mọi bài sau chỉ cần `from display import show_frame`:

```python
# display.py — hàm hiển thị dùng chung cho toàn giáo trình
import cv2

WINDOW_NAME = "Supervision - Live"
MAX_DISPLAY_WIDTH = 1280   # thu nhỏ frame cho vừa màn hình (chỉ để XEM, không ảnh hưởng xử lý)


def show_frame(frame, window_name: str = WINDOW_NAME, wait: int = 1) -> bool:
    """Hiện frame lên cửa sổ. Trả về False nếu người dùng bấm Q/ESC (muốn thoát).

    wait=1  -> dùng cho video (hiện liên tục, không chặn)
    wait=0  -> dùng cho ảnh tĩnh (dừng lại chờ bấm phím bất kỳ)
    """
    h, w = frame.shape[:2]
    if w > MAX_DISPLAY_WIDTH:                      # thu nhỏ để vừa màn hình
        scale = MAX_DISPLAY_WIDTH / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    cv2.imshow(window_name, frame)
    key = cv2.waitKey(wait) & 0xFF
    if key in (ord("q"), ord("Q"), 27):            # Q hoặc ESC -> thoát
        return False
    return True


def close_windows():
    cv2.destroyAllWindows()
```

**Quy ước phím trong mọi bài:**

| Phím | Tác dụng |
|---|---|
| `Q` hoặc `ESC` | Thoát, đóng cửa sổ |
| Phím bất kỳ | (với ảnh tĩnh, `wait=0`) đóng ảnh, chạy tiếp |

> ⚠️ **Lỗi hay gặp:** quên `cv2.waitKey()` sau `cv2.imshow()` → cửa sổ xám xịt/treo, "u u minh minh" không thấy gì. `imshow` chỉ **đăng ký** ảnh, còn `waitKey` mới là lệnh **thực sự vẽ** lên màn hình. Hàm `show_frame` ở trên đã lo trọn việc này.

**✅ Checkpoint Bài 0:** Chạy đoạn test sau, thấy cửa sổ hiện frame đầu tiên của video, bấm phím bất kỳ để đóng:

```python
import supervision as sv
from display import show_frame, close_windows

frame = next(sv.get_video_frames_generator("vehicles.mp4"))
show_frame(frame, wait=0)   # wait=0: dừng chờ xem
close_windows()
```

---

<a name="bài-1"></a>
## BÀI 1 — ĐỐI TƯỢNG TRUNG TÂM: `sv.Detections`

**Mục tiêu:** Hiểu cấu trúc dữ liệu quan trọng nhất — mọi thứ trong supervision xoay quanh nó.

### 1.1. Chạy detection đầu tiên — thấy ngay trên cửa sổ

Tạo file `bai1_detections.py`:

```python
import cv2
import supervision as sv
from ultralytics import YOLO

from display import show_frame, close_windows

# 1. Load model (yolov8n = nano, nhẹ nhất; tải tự động lần đầu)
model = YOLO("yolov8n.pt")

# 2. Lấy 1 frame từ video làm ảnh thử (hoặc cv2.imread("traffic.jpg"))
image = next(sv.get_video_frames_generator("vehicles.mp4"))

# 3. Chạy inference
results = model(image)[0]

# 4. ⭐ Chuyển output của YOLO thành sv.Detections
detections = sv.Detections.from_ultralytics(results)

print(detections)

# 5. 🖥️ Hiện luôn kết quả lên cửa sổ để đối chiếu với số liệu vừa in
annotated = sv.BoxAnnotator(thickness=2).annotate(image.copy(), detections)
cv2.putText(annotated, f"Phat hien: {len(detections)} doi tuong",
            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
show_frame(annotated, wait=0)   # bấm phím bất kỳ để đóng
close_windows()
```

### 1.2. Mổ xẻ `sv.Detections`

Chạy đoạn sau và quan sát kỹ output (đối chiếu với các box đang thấy trên cửa sổ):

```python
print("Số đối tượng:", len(detections))
print("xyxy (tọa độ box):\n", detections.xyxy)        # ndarray (N, 4)
print("confidence:", detections.confidence)             # ndarray (N,)
print("class_id:", detections.class_id)                 # ndarray (N,)
print("tên class:", detections.data["class_name"])      # ndarray (N,) dạng chuỗi
print("tracker_id:", detections.tracker_id)             # None (chưa track — Bài 5)
```

Cấu trúc bên trong:

| Thuộc tính | Kiểu | Ý nghĩa |
|---|---|---|
| `xyxy` | `ndarray (N,4)` | Tọa độ `[x1, y1, x2, y2]` mỗi box |
| `confidence` | `ndarray (N,)` | Độ tin cậy 0–1 |
| `class_id` | `ndarray (N,)` | ID lớp (COCO: 2=car, 3=motorcycle, 5=bus, 7=truck) |
| `tracker_id` | `ndarray (N,)` hoặc `None` | ID theo dõi qua các frame |
| `mask` | `ndarray (N,H,W)` hoặc `None` | Mask segmentation (nếu model hỗ trợ) |
| `data` | `dict` | Metadata phụ, vd `class_name` |

> 🔑 **Tư duy cốt lõi:** `sv.Detections` giống một "bảng" NumPy — mỗi **hàng** là một đối tượng. Vì vậy nó hỗ trợ indexing/slicing y hệt NumPy (Bài 3 sẽ khai thác triệt để).

### 1.3. Model-agnostic — sức mạnh thật sự

Cùng một pipeline, chỉ đổi 1 dòng khi đổi model:

```python
# Ultralytics (YOLOv8/v9/v10/v11...)
detections = sv.Detections.from_ultralytics(results)

# Roboflow Inference (RF-DETR, SAM...)
detections = sv.Detections.from_inference(results)

# Hugging Face Transformers (DETR...)
detections = sv.Detections.from_transformers(results)

# NCNN, EasyOCR, MMDetection... đều có connector tương ứng
```

**✅ Checkpoint Bài 1:** Cửa sổ hiện ảnh với box quanh từng xe kèm dòng chữ "Phat hien: N doi tuong"; console in được tên class của từng xe, và hai con số khớp nhau.

---

<a name="bài-2"></a>
## BÀI 2 — VẼ ĐẸP VỚI ANNOTATORS

**Mục tiêu:** Biến output khô khan thành hình ảnh trực quan chuyên nghiệp — và **so sánh các kiểu vẽ ngay trên cửa sổ**.

### 2.1. Cặp đôi cơ bản: Box + Label

```python
import cv2
import supervision as sv
from ultralytics import YOLO

from display import show_frame, close_windows

model = YOLO("yolov8n.pt")
image = next(sv.get_video_frames_generator("vehicles.mp4"))
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)

# Khởi tạo annotator (tạo 1 lần, dùng lại nhiều lần)
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(
    text_scale=0.5,
    text_thickness=1,
    text_position=sv.Position.TOP_LEFT,
)

# Tạo nhãn tùy biến: "car 0.87"
labels = [
    f"{class_name} {conf:.2f}"
    for class_name, conf
    in zip(detections.data["class_name"], detections.confidence)
]

# Vẽ (luôn copy để giữ ảnh gốc)
annotated = image.copy()
annotated = box_annotator.annotate(scene=annotated, detections=detections)
annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

# 🖥️ Hiện lên cửa sổ xem ngay (bấm phím bất kỳ để đóng)
show_frame(annotated, wait=0)
close_windows()

# (Tùy chọn) muốn lưu lại thì thêm:
cv2.imwrite("bai2_output.jpg", annotated)
```

### 2.2. Bộ sưu tập annotator — duyệt xem từng cái trên cửa sổ

Thay vì sửa code chạy lại từng lần, ta cho **mỗi annotator hiện lên lần lượt** — bấm phím bất kỳ để xem kiểu tiếp theo, Q để dừng:

```python
annotators = {
    "RoundBox": sv.RoundBoxAnnotator(),          # Box bo góc — nhìn hiện đại
    "BoxCorner": sv.BoxCornerAnnotator(),        # Chỉ vẽ 4 góc — phong cách "quân sự"
    "Ellipse": sv.EllipseAnnotator(),            # Ellipse dưới chân — phân tích bóng đá
    "Circle": sv.CircleAnnotator(),              # Vòng tròn bao quanh
    "Dot": sv.DotAnnotator(),                    # Chấm tại tâm
    "Triangle": sv.TriangleAnnotator(),          # Tam giác trên đầu — kiểu game
    "Color": sv.ColorAnnotator(opacity=0.4),     # Tô màu cả vùng box
    "Blur": sv.BlurAnnotator(),                  # Làm mờ đối tượng (che biển số, mặt!)
    "Pixelate": sv.PixelateAnnotator(),          # Pixel hóa đối tượng
    # Cần model segmentation (yolov8n-seg.pt): MaskAnnotator, PolygonAnnotator, HaloAnnotator
    # Cần tracker_id (Bài 5): TraceAnnotator
    # HeatMapAnnotator: bản đồ nhiệt mật độ — hợp với video hơn ảnh tĩnh
}

for name, annotator in annotators.items():
    annotated = annotator.annotate(image.copy(), detections)
    cv2.putText(annotated, name, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
    if not show_frame(annotated, window_name="So sanh Annotator", wait=0):
        break   # bấm Q thì dừng duyệt

close_windows()
```

### 2.3. Tùy biến màu sắc

```python
# Bảng màu tùy chỉnh theo class
box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(["#ff0000", "#00ff00", "#0000ff", "#ffff00"]),
    color_lookup=sv.ColorLookup.CLASS,   # màu theo class
    # color_lookup=sv.ColorLookup.TRACK, # màu theo tracker_id (Bài 5)
)
```

**✅ Checkpoint Bài 2:** Bấm phím lần lượt duyệt hết bộ sưu tập annotator trên cửa sổ, mỗi kiểu có tên hiện góc trên. **Bonus:** dùng `BlurAnnotator` che mờ toàn bộ người đi bộ và xem kết quả trên cửa sổ.

---

<a name="bài-3"></a>
## BÀI 3 — LỌC DETECTIONS NHƯ LỌC DATAFRAME

**Mục tiêu:** Thành thạo kỹ thuật lọc — kỹ năng dùng nhiều nhất trong dự án thực tế.

### 3.1. Cú pháp lọc bằng boolean mask

Vì `Detections` hoạt động như mảng NumPy:

```python
# Chỉ giữ đối tượng confidence > 0.5
detections = detections[detections.confidence > 0.5]

# Chỉ giữ xe hơi (COCO class 2)
detections = detections[detections.class_id == 2]

# Giữ nhiều class: car, motorcycle, bus, truck
import numpy as np
VEHICLE_CLASSES = [2, 3, 5, 7]
detections = detections[np.isin(detections.class_id, VEHICLE_CLASSES)]

# Kết hợp nhiều điều kiện
detections = detections[
    (detections.confidence > 0.4) & np.isin(detections.class_id, VEHICLE_CLASSES)
]

# Lọc theo diện tích box (loại box quá nhỏ = nhiễu)
detections = detections[detections.area > 1000]
```

### 3.2. 🖥️ Xem trực quan "trước – sau" khi lọc trên cùng cửa sổ

Cách học nhanh nhất là **nhìn thấy** bộ lọc làm gì — ghép ảnh trước/sau cạnh nhau:

```python
import cv2
import numpy as np

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
show_frame(side_by_side, window_name="Truoc vs Sau khi loc", wait=0)
close_windows()
```

### 3.3. NMS và các phép xử lý khác

```python
# Non-Max Suppression thủ công (khi model chưa làm hoặc gộp nhiều model)
detections = detections.with_nms(threshold=0.5, class_agnostic=False)

# Gộp detections từ 2 model khác nhau
merged = sv.Detections.merge([detections_yolo, detections_rtdetr]).with_nms(0.5)
```

### 3.4. Lấy tọa độ điểm neo (anchor)

Rất cần cho tracking, zone, đo tốc độ:

```python
# Tâm đáy box (điểm "chạm đất" — chuẩn cho xe cộ, người)
points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
# Các anchor khác: CENTER, TOP_LEFT, BOTTOM_RIGHT...
```

**✅ Checkpoint Bài 3:** Cửa sổ hiện 2 ảnh cạnh nhau "TRUOC LOC / SAU LOC" với số lượng box khác nhau rõ rệt; console in thống kê:
```
car: 12, truck: 3
```
Gợi ý: `zip(*np.unique(detections.data["class_name"], return_counts=True))`

---

<a name="bài-4"></a>
## BÀI 4 — XỬ LÝ VIDEO: PIPELINE CHUẨN CÓ LIVE PREVIEW

**Mục tiêu:** Nắm 3 utility video của supervision, viết được pipeline **vừa xem trực tiếp trên cửa sổ, vừa ghi file**.

### 4.1. Ba công cụ video

```python
import supervision as sv

VIDEO_PATH = "vehicles.mp4"

# (1) Đọc thông tin video
video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
print(video_info)  # width, height, fps, total_frames

# (2) Generator duyệt frame — thay cho vòng while cap.read() truyền thống
frame_generator = sv.get_video_frames_generator(VIDEO_PATH)
# Tùy chọn: get_video_frames_generator(VIDEO_PATH, stride=2, start=100, end=500)

# (3) Ghi video output — tự động khớp codec/fps/size
with sv.VideoSink(target_path="output.mp4", video_info=video_info) as sink:
    for frame in frame_generator:
        # ... xử lý frame ...
        sink.write_frame(frame)
```

> 📌 **Vì sao KHÔNG dùng `sv.process_video`?** Hàm đó tiện nhưng chạy "câm" — cắm mặt ghi file, không thấy gì cho tới khi xong. Ta tự viết vòng lặp để **vừa hiện cửa sổ vừa ghi file**, lại chủ động bấm Q dừng giữa chừng.

### 4.2. ⭐ Pipeline video hoàn chỉnh — BỘ KHUNG cho mọi bài sau

Tạo `bai4_video.py`:

```python
import numpy as np
import supervision as sv
from ultralytics import YOLO

from display import show_frame, close_windows

SOURCE_VIDEO = "vehicles.mp4"
TARGET_VIDEO = "bai4_output.mp4"
VEHICLE_CLASSES = [2, 3, 5, 7]

model = YOLO("yolov8n.pt")
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO)

box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5)


def process_frame(frame: np.ndarray) -> np.ndarray:
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[
        (detections.confidence > 0.3) & np.isin(detections.class_id, VEHICLE_CLASSES)
    ]

    labels = [f"{name} {conf:.2f}" for name, conf
              in zip(detections.data["class_name"], detections.confidence)]

    annotated = frame.copy()
    annotated = box_annotator.annotate(annotated, detections)
    annotated = label_annotator.annotate(annotated, detections, labels=labels)
    return annotated


# ⭐ Vòng lặp chuẩn: XEM TRỰC TIẾP trên cửa sổ + GHI FILE song song
with sv.VideoSink(target_path=TARGET_VIDEO, video_info=video_info) as sink:
    for frame in sv.get_video_frames_generator(SOURCE_VIDEO):
        annotated = process_frame(frame)
        sink.write_frame(annotated)              # ghi file
        if not show_frame(annotated):            # 🖥️ hiện cửa sổ; bấm Q -> dừng
            print("Nguoi dung bam Q - dung som.")
            break

close_windows()
print("Xong! Video da luu tai:", TARGET_VIDEO)
```

Chạy lên là thấy ngay cửa sổ video với box + label chạy realtime. Đây là **bộ khung** mọi bài sau đều dựa vào — chỉ thay phần `process_frame`.

> ⚙️ **Mẹo tăng tốc:** truyền `model(frame, imgsz=640, verbose=False)`; nếu có GPU thì `model.to("cuda")`. Với video 4K, dùng `imgsz=1280` cân bằng tốc độ/độ chính xác. Nếu preview giật vì máy yếu: thêm `stride=2` vào generator (bỏ qua 1 frame lấy 1 frame) — chỉ để học, khi xuất file thật thì bỏ stride.

**✅ Checkpoint Bài 4:** Cửa sổ hiện video với box + label chạy mượt trực tiếp, bấm Q dừng được giữa chừng, và file output vẫn được ghi. Em giải thích được vì sao dùng generator thay vì đọc hết video vào RAM.

---

<a name="bài-5"></a>
## BÀI 5 — TRACKING ĐA ĐỐI TƯỢNG VỚI BYTETRACK

**Mục tiêu:** Gán ID cố định cho từng xe qua các frame — nền tảng của đếm và đo tốc độ. **Nhìn thấy ID bám theo xe ngay trên cửa sổ.**

### 5.1. Vấn đề: detection không có "trí nhớ"

Ở Bài 4, mỗi frame model detect độc lập — chiếc xe ở frame 10 và frame 11 là "hai đối tượng khác nhau". Muốn **đếm** hay **đo tốc độ**, phải biết "đây vẫn là xe số 7". Đó là bài toán **Multi-Object Tracking (MOT)**.

### 5.2. Thêm ByteTrack — chỉ 3 dòng

```python
tracker = sv.ByteTrack(
    frame_rate=video_info.fps,             # QUAN TRỌNG: khớp fps video
    track_activation_threshold=0.25,       # conf tối thiểu để khởi tạo track
    lost_track_buffer=30,                  # giữ track "mất dấu" trong 30 frame
    minimum_matching_threshold=0.8,        # ngưỡng IoU để khớp track
)

def process_frame(frame):
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, VEHICLE_CLASSES)]

    # ⭐ Dòng ma thuật: cập nhật tracker
    detections = tracker.update_with_detections(detections)

    # Giờ detections.tracker_id đã có giá trị!
    labels = [f"#{tid} {name}" for tid, name
              in zip(detections.tracker_id, detections.data["class_name"])]
    ...
    # phần vẽ + return annotated giữ nguyên bộ khung Bài 4
```

Phần vòng lặp cuối file **giữ nguyên bộ khung Bài 4** (VideoSink + `show_frame`) — chạy lên, nhìn cửa sổ sẽ thấy mỗi xe mang một nhãn `#ID` **không đổi** khi di chuyển. Đó chính là tracking đang hoạt động, thấy tận mắt chứ không phải đoán.

### 5.3. Vẽ quỹ đạo với TraceAnnotator

```python
trace_annotator = sv.TraceAnnotator(
    trace_length=video_info.fps * 2,   # lưu vệt 2 giây
    position=sv.Position.BOTTOM_CENTER,
)
# Trong process_frame:
annotated = trace_annotator.annotate(annotated, detections)
```

Đổi màu theo track để nhìn rõ từng xe trên cửa sổ:

```python
box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
```

### 5.4. Smoothing — chống box "giật"

```python
smoother = sv.DetectionsSmoother(length=5)
# Sau tracker:
detections = smoother.update_with_detections(detections)
```

> 💡 Bật/tắt dòng smoother rồi nhìn cửa sổ so sánh: không có smoother box rung lật phật, có smoother box mượt hẳn — hiệu quả thấy ngay bằng mắt.

> 🧠 **Hiểu sâu ByteTrack (đọc thêm):** ByteTrack khớp track bằng IoU qua 2 vòng — vòng 1 với detection confidence cao, vòng 2 tận dụng cả detection confidence thấp (thường là xe bị che khuất) thay vì vứt đi. Đó là lý do nó giữ ID tốt khi xe che nhau. Lưu ý mỗi khi xử lý video mới phải `tracker.reset()`.

**✅ Checkpoint Bài 5:** Trên cửa sổ live: mỗi xe một màu riêng, nhãn `#ID`, có vệt quỹ đạo bám theo. Quan sát trực tiếp thấy ID không đổi khi xe đi qua chỗ che khuất ngắn.

---

<a name="bài-6"></a>
## BÀI 6 — ĐẾM ĐỐI TƯỢNG QUA VẠCH: LINEZONE

**Mục tiêu:** Đếm xe theo hai chiều vào/ra — **nhìn số đếm nhảy realtime ngay trên cửa sổ** khi xe cắt vạch.

### 6.1. Nguyên lý

`LineZone` là một đoạn thẳng ảo. Khi **anchor point** của một track đi từ bên này sang bên kia vạch → tăng bộ đếm `in_count` hoặc `out_count`. **Bắt buộc phải có tracker_id** (Bài 5).

### 6.2. Code

```python
# Vạch ngang giữa khung hình (chỉnh tọa độ theo video của em)
START = sv.Point(0, video_info.height // 2)
END = sv.Point(video_info.width, video_info.height // 2)

line_zone = sv.LineZone(
    start=START,
    end=END,
    triggering_anchors=[sv.Position.BOTTOM_CENTER],  # xét điểm chạm đất
    minimum_crossing_threshold=2,   # cần 2 frame xác nhận — chống đếm trùng do jitter
)

line_annotator = sv.LineZoneAnnotator(
    thickness=2,
    text_scale=0.8,
    custom_in_text="Vao",
    custom_out_text="Ra",
)

def process_frame(frame):
    ...  # detect → filter → track như Bài 5

    # ⭐ Cập nhật bộ đếm
    crossed_in, crossed_out = line_zone.trigger(detections)

    annotated = ...  # vẽ box/label/trace
    # Vẽ vạch + số đếm Vao/Ra realtime ngay trên khung hình
    annotated = line_annotator.annotate(annotated, line_counter=line_zone)
    return annotated

# Vòng lặp: giữ nguyên bộ khung Bài 4 (VideoSink + show_frame)
# -> nhìn cửa sổ thấy số "Vao/Ra" nhảy lên từng xe một khi cắt vạch!

# Sau khi xử lý xong:
print(f"Tổng vào: {line_zone.in_count} | Tổng ra: {line_zone.out_count}")
```

### 6.3. Đếm riêng từng loại xe — hiện bảng thống kê realtime trên cửa sổ

`trigger()` trả về boolean mask cho biết **detection nào vừa cắt vạch**. Thay vì chỉ in ra console cuối video, ta **vẽ thẳng bảng thống kê lên góc khung hình**, cập nhật từng frame:

```python
import cv2
from collections import defaultdict

counts = defaultdict(int)

def update_counts(detections, line_zone):
    crossed_in, crossed_out = line_zone.trigger(detections)
    for name in detections.data["class_name"][crossed_in]:
        counts[f"{name}_in"] += 1
    for name in detections.data["class_name"][crossed_out]:
        counts[f"{name}_out"] += 1

def draw_stats(frame, stats: dict, origin=(20, 50)):
    """Vẽ bảng thống kê nền đen mờ ở góc trên trái — nhìn rõ trên mọi video."""
    x, y = origin
    line_h = 45
    h = line_h * (len(stats) + 1)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 10, y - 40), (x + 420, y - 40 + h), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)   # nền đen mờ 60%
    for i, (k, v) in enumerate(stats.items()):
        cv2.putText(frame, f"{k}: {v}", (x, y + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    return frame

# Trong process_frame, sau khi vẽ box/label/line:
#   update_counts(detections, line_zone)   # (gọi trigger 1 lần duy nhất ở đây)
#   annotated = draw_stats(annotated, counts)
```

### 6.4. Cách tìm tọa độ vạch nhanh — rê chuột đọc tọa độ ngay trên cửa sổ

Không cần mở Paint! Dùng chính cửa sổ OpenCV: click chuột vào đâu, tọa độ in ra đó:

```python
import cv2
import supervision as sv

frame = next(sv.get_video_frames_generator("vehicles.mp4"))
frame = cv2.resize(frame, None, fx=0.33, fy=0.33)   # thu nhỏ; nhớ nhân lại 3 lần khi dùng

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Toa do (da thu nho): ({x}, {y})  ->  goc: ({x*3}, {y*3})")

cv2.namedWindow("Chon toa do - click chuot, Q de thoat")
cv2.setMouseCallback("Chon toa do - click chuot, Q de thoat", on_mouse)
while True:
    cv2.imshow("Chon toa do - click chuot, Q de thoat", frame)
    if cv2.waitKey(20) & 0xFF in (ord("q"), 27):
        break
cv2.destroyAllWindows()
```

(Hoặc dùng công cụ online [polygonzone.roboflow.com](https://polygonzone.roboflow.com).)

**✅ Checkpoint Bài 6:** Trên cửa sổ live: vạch đếm hiện số Vào/Ra nhảy realtime khi xe cắt vạch, góc trái có bảng thống kê `car_in / car_out / truck_in / ...` cập nhật liên tục.

---

<a name="bài-7"></a>
## BÀI 7 — GIÁM SÁT THEO VÙNG: POLYGONZONE

**Mục tiêu:** Đếm/giám sát đối tượng **bên trong một vùng đa giác** — làn đường, bãi đỗ, vùng cấm — số xe trong vùng hiện ngay giữa polygon trên cửa sổ.

### 7.1. Code

```python
import numpy as np

# Đa giác vùng quan tâm (lấy tọa độ bằng tool click chuột ở Bài 6.4
# hoặc polygonzone.roboflow.com)
polygon = np.array([
    [100, 400], [600, 400], [700, 700], [50, 700]
])

zone = sv.PolygonZone(
    polygon=polygon,
    triggering_anchors=(sv.Position.BOTTOM_CENTER,),
)

zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone,
    color=sv.Color.RED,
    thickness=2,
    text_scale=1,
    opacity=0.2,          # tô nền vùng mờ mờ
)

def process_frame(frame):
    ...
    # mask boolean: detection nào đang ở TRONG vùng
    in_zone_mask = zone.trigger(detections)

    # Ví dụ ứng dụng: chỉ vẽ box cho xe trong vùng
    detections_in_zone = detections[in_zone_mask]

    # zone_annotator TỰ VẼ số lượng hiện tại (current_count) vào giữa vùng
    annotated = zone_annotator.annotate(annotated)
    ...

# Vòng lặp: giữ nguyên bộ khung Bài 4 -> cửa sổ hiện polygon đỏ mờ,
# con số giữa vùng tăng/giảm theo số xe đang ở trong, thấy trực tiếp từng frame.
```

### 7.2. Ứng dụng thực tế của PolygonZone

| Bài toán | Cách dùng |
|---|---|
| Đếm xe theo **làn** | Mỗi làn 1 polygon, so sánh mật độ |
| Phát hiện **đỗ sai** | Xe có mặt trong vùng > N giây → cảnh báo |
| **Vùng an toàn** nhà máy | Người xuất hiện trong polygon máy đang chạy → còi |
| Đo **thời gian dừng** (dwell time) | Ghi frame vào/ra của từng tracker_id trong zone |

Ví dụ phát hiện dừng lâu — **cảnh báo đỏ hiện thẳng lên cửa sổ**, không chỉ in console:

```python
import cv2
from collections import defaultdict

frames_in_zone = defaultdict(int)

def check_dwell(frame, detections, zone, fps):
    in_zone = zone.trigger(detections)
    warnings = []
    for tid in detections.tracker_id[in_zone]:
        frames_in_zone[tid] += 1
        if frames_in_zone[tid] > fps * 10:          # dừng > 10 giây
            warnings.append(int(tid))
    if warnings:
        # 🖥️ Banner cảnh báo đỏ to đùng trên cửa sổ — không thể không thấy
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (0, 0, 255), -1)
        cv2.putText(frame, f"!!! XE DUNG QUA LAU TRONG VUNG: {warnings} !!!",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        print(f"⚠️ Xe {warnings} dừng quá lâu trong vùng!")   # log kèm theo
    return frame
```

**✅ Checkpoint Bài 7:** Cửa sổ live hiện 2 polygon (2 làn đường), mỗi vùng hiện số xe hiện tại ngay giữa vùng; khi có xe dừng > 10 giây, banner đỏ cảnh báo hiện trên đầu khung hình.

---

<a name="bài-8"></a>
## BÀI 8 — ĐO TỐC ĐỘ VỚI VIEWTRANSFORMER

**Mục tiêu:** Chuyển tọa độ pixel → tọa độ mét thật để tính km/h — **tốc độ hiện ngay trên nhãn từng xe trong cửa sổ live**.

### 8.1. Vấn đề phối cảnh

Trên ảnh camera, 100 pixel ở **gần** camera ≠ 100 pixel ở **xa** camera (tính theo mét thật). Giải pháp: **perspective transform** — ánh xạ 4 điểm trên ảnh (hình thang mặt đường) về hình chữ nhật kích thước thật.

```
   Ảnh camera (pixel)                Tọa độ thật (mét)
      A─────B                          A──────B
     ╱       ╲          ──────▶        │      │  ví dụ: rộng 25m
    ╱         ╲                        │      │        dài 250m
   D───────────C                       D──────C
```

### 8.2. Code

```python
import numpy as np
import supervision as sv
from collections import defaultdict, deque

# 4 điểm hình thang trên ảnh — PHẢI đo/ước lượng theo video thực tế của em
# (dùng tool click chuột ở Bài 6.4 để lấy tọa độ ngay trên cửa sổ)
SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

# Kích thước thật của vùng đó (mét). Mẹo: dựa vào vạch kẻ đường
# (vạch đứt chuẩn VN: 1–3m vạch, 3–6m khoảng trống tùy cấp đường)
TARGET_WIDTH, TARGET_HEIGHT = 25, 250
TARGET = np.array([
    [0, 0], [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1],
])

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        import cv2
        self.m = cv2.getPerspectiveTransform(
            source.astype(np.float32), target.astype(np.float32))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        import cv2
        if points.size == 0:
            return points
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(reshaped, self.m).reshape(-1, 2)

view_transformer = ViewTransformer(SOURCE, TARGET)

# Lưu lịch sử tọa độ y (mét) của từng track trong 1 giây gần nhất
coordinates = defaultdict(lambda: deque(maxlen=int(video_info.fps)))

def compute_speed_labels(detections):
    points = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    points = view_transformer.transform_points(points)

    labels = []
    for tracker_id, (_, y) in zip(detections.tracker_id, points):
        coordinates[tracker_id].append(y)
        if len(coordinates[tracker_id]) < video_info.fps / 2:
            labels.append(f"#{tracker_id}")
        else:
            # quãng đường (m) đi được trong khoảng thời gian (s)
            distance = abs(coordinates[tracker_id][-1] - coordinates[tracker_id][0])
            time_s = len(coordinates[tracker_id]) / video_info.fps
            speed_kmh = distance / time_s * 3.6
            labels.append(f"#{tracker_id} {int(speed_kmh)} km/h")
    return labels

# Trong process_frame: labels = compute_speed_labels(detections)
# rồi đưa vào label_annotator như thường.
# Vòng lặp: giữ nguyên bộ khung Bài 4 -> nhìn cửa sổ thấy tốc độ km/h
# hiện trên đầu từng xe, cập nhật mượt theo thời gian thực.
```

> 💡 **Mẹo debug bằng mắt:** vẽ luôn hình thang SOURCE lên khung hình để kiểm tra mình chọn điểm đúng chưa — sai là thấy liền trên cửa sổ:
> ```python
> annotated = sv.draw_polygon(annotated, polygon=SOURCE, color=sv.Color.RED, thickness=4)
> ```

> ⚠️ **Độ chính xác phụ thuộc:** (1) camera cố định, (2) 4 điểm SOURCE đo đúng, (3) kích thước TARGET đúng. Sai số ±5–10% là bình thường với phương pháp đơn camera.

**✅ Checkpoint Bài 8:** Trên cửa sổ live, nhãn mỗi xe hiện tốc độ km/h, giá trị ổn định (không nhảy loạn). Em giải thích được vì sao phải dùng `deque` trung bình 1 giây thay vì tính giữa 2 frame liên tiếp (đáp án: khử nhiễu jitter của box).

---

<a name="bài-9"></a>
## BÀI 9 — SẢN PHẨM CUỐI: TRAFFIC MONITOR HOÀN CHỈNH

**Mục tiêu:** Ghép toàn bộ Bài 1–8 thành một sản phẩm chạy được, có cấu trúc code chuẩn, **màn hình giám sát live + bảng thống kê realtime trên cửa sổ**, xuất báo cáo.

### 9.1. Đặc tả sản phẩm

| Tính năng | Bài học tương ứng |
|---|---|
| Phát hiện + phân loại 4 loại xe | Bài 1, 3 |
| Tracking ID + vệt quỹ đạo | Bài 5 |
| Đếm xe 2 chiều qua vạch, chi tiết theo loại | Bài 6 |
| Đo tốc độ km/h, cảnh báo quá tốc độ | Bài 8 |
| 🖥️ **Cửa sổ giám sát live + bảng thống kê realtime, Q để dừng** | Bài 0, 4, 6 |
| Xuất video annotated + báo cáo JSON | Bài 4 |

### 9.2. Cấu trúc dự án

```
traffic_monitor/
├── config.py        # mọi tham số chỉnh ở đây
├── display.py       # show_frame + draw_stats (Bài 0, Bài 6)
├── transformer.py   # ViewTransformer (Bài 8)
├── monitor.py       # class TrafficMonitor — pipeline chính
└── main.py          # điểm chạy
```

### 9.3. `config.py`

```python
import numpy as np

SOURCE_VIDEO = "vehicles.mp4"
TARGET_VIDEO = "output_final.mp4"
REPORT_PATH = "report.json"

MODEL_NAME = "yolov8n.pt"       # đổi yolov8s/m nếu có GPU
CONF_THRESHOLD = 0.3
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
SPEED_LIMIT_KMH = 80

SHOW_PREVIEW = True             # 🖥️ False nếu chạy trên server không màn hình

# Hiệu chỉnh theo video của em (Bài 6, Bài 8)
LINE_Y_RATIO = 0.5              # vạch đếm ở 50% chiều cao khung hình
PERSPECTIVE_SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
ROAD_WIDTH_M, ROAD_LENGTH_M = 25, 250
```

### 9.4. `monitor.py` — trái tim sản phẩm

```python
import json
from collections import defaultdict, deque

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

import config
from display import show_frame, close_windows
from transformer import ViewTransformer


class TrafficMonitor:
    def __init__(self):
        self.model = YOLO(config.MODEL_NAME)
        self.video_info = sv.VideoInfo.from_video_path(config.SOURCE_VIDEO)
        fps = self.video_info.fps

        # --- Tracking (Bài 5) ---
        self.tracker = sv.ByteTrack(frame_rate=fps)
        self.smoother = sv.DetectionsSmoother(length=5)

        # --- Đếm qua vạch (Bài 6) ---
        y = int(self.video_info.height * config.LINE_Y_RATIO)
        self.line_zone = sv.LineZone(
            start=sv.Point(0, y),
            end=sv.Point(self.video_info.width, y),
            triggering_anchors=[sv.Position.BOTTOM_CENTER],
            minimum_crossing_threshold=2,
        )
        self.class_counts = defaultdict(int)

        # --- Đo tốc độ (Bài 8) ---
        target = np.array([
            [0, 0], [config.ROAD_WIDTH_M - 1, 0],
            [config.ROAD_WIDTH_M - 1, config.ROAD_LENGTH_M - 1],
            [0, config.ROAD_LENGTH_M - 1],
        ])
        self.view_transformer = ViewTransformer(config.PERSPECTIVE_SOURCE, target)
        self.coordinates = defaultdict(lambda: deque(maxlen=int(fps)))
        self.speed_violations = {}   # tracker_id -> max speed vi phạm

        # --- Annotators (Bài 2) ---
        thickness = sv.calculate_optimal_line_thickness(self.video_info.resolution_wh)
        text_scale = sv.calculate_optimal_text_scale(self.video_info.resolution_wh)
        self.box_annotator = sv.BoxAnnotator(
            thickness=thickness, color_lookup=sv.ColorLookup.TRACK)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=text_scale, color_lookup=sv.ColorLookup.TRACK)
        self.trace_annotator = sv.TraceAnnotator(
            trace_length=int(fps * 2), thickness=thickness,
            color_lookup=sv.ColorLookup.TRACK)
        self.line_annotator = sv.LineZoneAnnotator(
            thickness=thickness, text_scale=text_scale,
            custom_in_text="Vao", custom_out_text="Ra")

    # ---------- các bước pipeline ----------

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        mask = (detections.confidence > config.CONF_THRESHOLD) & \
               np.isin(detections.class_id, config.VEHICLE_CLASSES)
        return detections[mask]

    def track(self, detections):
        detections = self.tracker.update_with_detections(detections)
        return self.smoother.update_with_detections(detections)

    def count(self, detections):
        crossed_in, crossed_out = self.line_zone.trigger(detections)
        for name in detections.data["class_name"][crossed_in]:
            self.class_counts[f"{name}_in"] += 1
        for name in detections.data["class_name"][crossed_out]:
            self.class_counts[f"{name}_out"] += 1

    def speeds(self, detections):
        pts = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pts = self.view_transformer.transform_points(pts)
        fps = self.video_info.fps

        labels = []
        for tid, name, (_, y) in zip(
                detections.tracker_id, detections.data["class_name"], pts):
            self.coordinates[tid].append(y)
            if len(self.coordinates[tid]) < fps / 2:
                labels.append(f"#{tid} {name}")
                continue
            dist = abs(self.coordinates[tid][-1] - self.coordinates[tid][0])
            speed = dist / (len(self.coordinates[tid]) / fps) * 3.6
            tag = " !QUA TOC DO!" if speed > config.SPEED_LIMIT_KMH else ""
            if tag:
                self.speed_violations[int(tid)] = max(
                    self.speed_violations.get(int(tid), 0), int(speed))
            labels.append(f"#{tid} {name} {int(speed)}km/h{tag}")
        return labels

    def draw_dashboard(self, frame):
        """🖥️ Bảng thống kê realtime góc trên trái — nhìn phát biết ngay tình hình."""
        stats = {
            "Tong VAO": self.line_zone.in_count,
            "Tong RA": self.line_zone.out_count,
            "Vi pham toc do": len(self.speed_violations),
            **dict(self.class_counts),
        }
        x, y, line_h = 25, 70, 55
        h = line_h * len(stats) + 30
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (560, 10 + h), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        for i, (k, v) in enumerate(stats.items()):
            color = (0, 0, 255) if "Vi pham" in k and v > 0 else (255, 255, 255)
            cv2.putText(frame, f"{k}: {v}", (x, y + i * line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        return frame

    def annotate(self, frame, detections, labels):
        out = frame.copy()
        out = self.trace_annotator.annotate(out, detections)
        out = self.box_annotator.annotate(out, detections)
        out = self.label_annotator.annotate(out, detections, labels=labels)
        out = self.line_annotator.annotate(out, line_counter=self.line_zone)
        out = self.draw_dashboard(out)          # 🖥️ bảng thống kê realtime
        return out

    # ---------- chạy ----------

    def process_frame(self, frame):
        detections = self.detect(frame)
        detections = self.track(detections)
        self.count(detections)
        labels = self.speeds(detections)
        return self.annotate(frame, detections, labels)

    def run(self):
        # 🖥️ Vòng lặp bộ khung Bài 4: XEM LIVE + GHI FILE song song, Q để dừng
        with sv.VideoSink(target_path=config.TARGET_VIDEO,
                          video_info=self.video_info) as sink:
            for frame in sv.get_video_frames_generator(config.SOURCE_VIDEO):
                annotated = self.process_frame(frame)
                sink.write_frame(annotated)
                if config.SHOW_PREVIEW and not show_frame(
                        annotated, window_name="Traffic Monitor - Q de thoat"):
                    print("Da dung theo yeu cau nguoi dung.")
                    break
        close_windows()
        self.export_report()

    def export_report(self):
        report = {
            "video": config.SOURCE_VIDEO,
            "total_in": self.line_zone.in_count,
            "total_out": self.line_zone.out_count,
            "by_class": dict(self.class_counts),
            "speed_violations": self.speed_violations,
        }
        with open(config.REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2))
```

### 9.5. `main.py`

```python
from monitor import TrafficMonitor

if __name__ == "__main__":
    TrafficMonitor().run()
```

Chạy: `python main.py` → **cửa sổ giám sát hiện lên ngay**: box màu theo từng xe, vệt quỹ đạo, vạch đếm Vao/Ra, tốc độ km/h trên nhãn, bảng thống kê realtime góc trái (số vi phạm hiện màu đỏ). Bấm Q dừng lúc nào cũng được. Song song đó video annotated được ghi ra file + `report.json`:

```json
{
  "total_in": 47,
  "total_out": 52,
  "by_class": {"car_in": 40, "truck_in": 5, "bus_in": 2, "car_out": 48, ...},
  "speed_violations": {"7": 94, "23": 88}
}
```

### 9.6. Hướng mở rộng thành sản phẩm thương mại

1. **Realtime từ camera/RTSP:** thay generator bằng `cv2.VideoCapture("rtsp://...")`, thêm hàng đợi frame (threading) để không nghẽn — cửa sổ live sẵn có rồi, chỉ đổi nguồn.
2. **Dashboard web:** đẩy `report` qua WebSocket/FastAPI, vẽ biểu đồ realtime.
3. **Đọc biển số xe vi phạm:** crop box xe quá tốc độ → chạy OCR biển số.
4. **Nhiều camera:** mỗi camera một `TrafficMonitor` instance, mỗi instance một cửa sổ (`window_name` khác nhau), gộp báo cáo về DB.

**✅ Checkpoint Bài 9 (tốt nghiệp):** Sản phẩm chạy end-to-end trên video của chính em, cửa sổ giám sát đầy đủ thông tin realtime, số đếm sai lệch < 5% so với đếm tay.

---

<a name="bài-10"></a>
## BÀI 10 — NÂNG CAO (TỰ CHỌN)

### 10.1. InferenceSlicer — phát hiện vật thể nhỏ (SAHI)

Ảnh flycam/độ phân giải lớn, vật thể bé xíu → cắt ảnh thành ô, detect từng ô, gộp lại:

```python
def callback(image_slice: np.ndarray) -> sv.Detections:
    result = model(image_slice, verbose=False)[0]
    return sv.Detections.from_ultralytics(result)

slicer = sv.InferenceSlicer(
    callback=callback,
    slice_wh=(640, 640),
    overlap_wh=(128, 128),
)
detections = slicer(image)

# So sánh bằng mắt trên cửa sổ: detect thường vs qua slicer (ghép ngang như Bài 3.2)
```

### 10.2. DetectionDataset — làm việc với dataset

```python
ds = sv.DetectionDataset.from_yolo(
    images_directory_path="dataset/images",
    annotations_directory_path="dataset/labels",
    data_yaml_path="dataset/data.yaml",
)
# Chuyển đổi format: ds.as_coco(...), ds.as_pascal_voc(...)
# Chia tập: train_ds, test_ds = ds.split(split_ratio=0.8)

# Duyệt xem dataset bằng cửa sổ: bấm phím lật ảnh, Q thoát
box_annotator = sv.BoxAnnotator()
for _, image, gt in ds:
    if not show_frame(box_annotator.annotate(image.copy(), gt), wait=0):
        break
close_windows()
```

### 10.3. Metrics — đánh giá model

```python
from supervision.metrics import MeanAveragePrecision, F1Score

map_metric = MeanAveragePrecision()
for _, image, gt in test_ds:
    pred = sv.Detections.from_ultralytics(model(image, verbose=False)[0])
    map_metric.update(pred, gt)
print(map_metric.compute())   # mAP50, mAP50-95...
```

---

<a name="bài-tập"></a>
## BÀI TẬP & LỘ TRÌNH TỰ LUYỆN

### Lộ trình gợi ý (2 tuần)

| Ngày | Nội dung |
|---|---|
| 1–2 | Bài 0–2: môi trường, cửa sổ hiển thị, Detections, Annotators |
| 3–4 | Bài 3–4: lọc, pipeline video live preview |
| 5–6 | Bài 5: tracking + đọc paper ByteTrack |
| 7–8 | Bài 6–7: LineZone, PolygonZone |
| 9–10 | Bài 8: đo tốc độ, hiệu chỉnh phối cảnh trên video tự quay |
| 11–13 | Bài 9: ráp sản phẩm, refactor, đo độ chính xác |
| 14 | Bài 10 + demo, viết README |

### Đồ án tự chọn (nâng dần độ khó)

1. ⭐ **Đếm người ra vào cửa hàng** — LineZone dọc, class `person`, số đếm hiện live trên cửa sổ, xuất thống kê theo giờ.
2. ⭐⭐ **Giám sát bãi đỗ xe** — mỗi ô đỗ 1 PolygonZone, cửa sổ hiển thị ô trống (xanh) / ô đầy (đỏ) realtime.
3. ⭐⭐ **Phát hiện đi vào vùng cấm nhà xưởng** — PolygonZone + banner cảnh báo đỏ trên cửa sổ + âm thanh + lưu clip vi phạm.
4. ⭐⭐⭐ **Phân tích bóng đá** — EllipseAnnotator, tracking cầu thủ, heatmap vị trí, tốc độ chạy, tất cả hiện live.
5. ⭐⭐⭐ **Đếm sản phẩm băng chuyền** — train YOLO custom trên Roboflow + LineZone, xử lý và hiển thị realtime.

### Tài nguyên chính thống

- Docs: `supervision.roboflow.com` (mục **How-to Guides** và **Cookbooks** cực kỳ đáng đọc)
- GitHub: `github.com/roboflow/supervision` (thư mục `examples/` có sẵn speed estimation, traffic analysis)
- Công cụ lấy tọa độ polygon: `polygonzone.roboflow.com`
- Kênh YouTube Roboflow: các video của Piotr Skalski (tác giả supervision) làm đúng bài toán trong giáo trình này

---

*Giáo trình biên soạn cho mục đích giảng dạy — cập nhật theo supervision 0.29.x (2026). Khi API thay đổi, kiểm tra mục Deprecated trong docs chính thức.*
