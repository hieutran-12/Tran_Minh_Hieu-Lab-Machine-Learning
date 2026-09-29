# TT-26 — CNN sàng lọc viêm phổi trên ảnh X-quang ngực

> ## ⚖️ CẢNH BÁO Y TẾ
> Đây là công cụ **sắp thứ tự ưu tiên đọc phim**, **KHÔNG phải công cụ chẩn đoán** và **không được dùng thay bác sĩ**.
> - Dữ liệu huấn luyện từ **Quảng Châu (Trung Quốc)**, trên **bệnh nhi 1–5 tuổi** → **không tổng quát hoá** cho người lớn hay dân số khác.
> - Mọi ca đều phải có bác sĩ đọc lại. Model có thể bỏ sót ca viêm phổi (số ca bỏ sót trên tập test ghi ở mục Kết quả).

## 1. Bài toán

Bệnh viện tuyến huyện không có bác sĩ chẩn đoán hình ảnh trực đêm; phim chụp lúc 2h sáng phải chờ tới sáng mới có người đọc. Model gắn cờ ca nghi ngờ viêm phổi để bác sĩ trực xem trước.

**RECALL là tối thượng** (mục tiêu ≥ 0,97 trên validation, ≥ 0,96 trên test): bỏ sót viêm phổi ở trẻ em có thể dẫn tới tử vong, còn báo động giả chỉ tốn thêm một lần bác sĩ liếc mắt.

## 2. Cấu trúc thư mục

```
TT-26-CNN-HoTen/
├── README.md
├── notebooks/cnn_chest_xray.ipynb   # chạy từng bước 1–12 có giải thích
├── src/
│   ├── data_pipeline.py             # tải/đếm/tách validation/tf.data/augmentation
│   ├── model.py                     # CNN baseline, EfficientNetB0, fine-tuning, Grad-CAM
│   └── train.py                     # huấn luyện + đánh giá end-to-end (CLI)
├── models/best_model.keras          # sinh ra sau khi chạy
├── reports/                         # learning_curves, confusion_matrix, gradcam_examples, ca_du_doan_sai (.png) + results.md, metrics.json
└── requirements.txt
```

## 3. Cài đặt và chạy

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Dữ liệu** — [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (~1,2 GB). Chọn một trong các cách:

1. **Tự động (mặc định):** chạy `train.py`/notebook, code tự tải bằng `kagglehub` (có thể cần tài khoản Kaggle: đặt biến môi trường `KAGGLE_USERNAME`, `KAGGLE_KEY`).
2. **Thủ công:** tải file zip từ Kaggle, giải nén vào thư mục `data/` (chỉ cần có `train/`, `test/` chứa `NORMAL/` và `PNEUMONIA/`; thư mục lồng nhau `chest_xray/chest_xray/` được tự nhận diện).
3. **Chỉ định đường dẫn:** `--data_dir /duong/dan/chest_xray`.

**Chạy toàn bộ pipeline:**

```bash
python src/train.py                                  # ./data hoặc tự tải từ Kaggle
python src/train.py --data_dir /duong/dan/chest_xray
python src/train.py --help                           # batch size, số epoch, no_cache...
```

**Hoặc chạy notebook** `notebooks/cnn_chest_xray.ipynb` (khuyến nghị Colab/Kaggle GPU).

> ⏱ Cần **GPU** (khoảng 8–12 giờ theo đề bài gồm cả thử nghiệm). Mặc định ảnh 224×224 được cache trong RAM (~1 GB); nếu thiếu RAM thêm `--no_cache`, nếu thiếu VRAM giảm `--batch_size 16`.

## 4. Những điểm kỹ thuật chính

| Vấn đề của dữ liệu | Cách xử lý |
|---|---|
| Tập val gốc chỉ có **16 ảnh** | Tự tách lại 15% từ train (~1/7), **phân tầng theo nhãn** và **nhóm theo bệnh nhân** (mã suy ra từ tên file) để không rò rỉ ảnh cùng một người giữa train/val |
| Mất cân bằng (~74% PNEUMONIA) | `class_weight` cân bằng |
| Train ≠ test về phân phối | Không đánh giá bằng train/val; test chỉ dùng **một lần** ở cuối. Accuracy test thấp hơn recall là hiện tượng thật |

- **Augmentation hợp lý với y tế:** xoay ±10°, dịch ±10%, phóng to ±10%, đổi sáng/tương phản nhẹ; **không lật ngang** (tim nằm bên trái, lật là sai giải phẫu) và không lật dọc. Chỉ áp dụng cho train.
- **3 hướng so sánh:** CNN 3 khối Conv train từ đầu → EfficientNetB0 (ImageNet) đóng băng, lr=1e-3 → fine-tune 30 tầng cuối, **lr=1e-5** (BatchNorm giữ đóng băng). Model tốt nhất chọn theo **AUC trên validation**.
- **Ngưỡng quyết định:** chọn trên validation là ngưỡng cao nhất mà recall ≥ 0,97 (ít báo động giả nhất trong các ngưỡng đạt recall). Không dùng accuracy làm metric chính.
- **Grad-CAM** (tính trên logit) cho 6 ca đa dạng (TP/TN/FN/FP) để kiểm tra model nhìn vào vùng phổi hay chữ/nhãn ở góc phim.

## 5. Kết quả

Phần này **tự động cập nhật** mỗi khi chạy `python src/train.py` (cũng lưu ở `reports/results.md`; số liệu chi tiết ở `reports/metrics.json`).

<!-- RESULTS:START -->
_Chưa có kết quả — chạy `python src/train.py` để sinh._
<!-- RESULTS:END -->

### Nhận xét (điền sau khi xem các hình trong `reports/`)

- **So sánh CNN từ đầu vs Transfer Learning vs Fine-tuning:** …
- **Grad-CAM (`gradcam_examples.png`):** model có tập trung vào vùng phổi không? Có ca nào chú ý vào chữ/nhãn/góc phim?
- **Phân tích 10 ca sai (`ca_du_doan_sai.png`):** các ca bỏ sót có đặc điểm chung gì?
- **Giới hạn:** dữ liệu từ một bệnh viện nhi, bệnh nhân 1–5 tuổi; phân phối test khác train; chưa kiểm định trên dữ liệu bên ngoài.

## 6. Đối chiếu tiêu chí hoàn thành

| Tiêu chí | Ở đâu |
|---|---|
| Tự tách lại validation (nêu lý do val gốc chỉ 16 ảnh) | Mục 4; `split_train_val` trong `src/data_pipeline.py` |
| So sánh CNN từ đầu / Transfer Learning / Fine-tuning | Bảng so sánh ở mục 5, `reports/model_comparison.csv` |
| Augmentation không lật ngang | `build_augmentation` trong `src/data_pipeline.py` |
| Recall test ≥ 0,96 | Mục 5 (kết quả thực tế sau khi chạy) |
| Ma trận nhầm lẫn ghi rõ số ca bỏ sót | `reports/confusion_matrix.png`, mục 5 |
| Grad-CAM + nhận xét | `reports/gradcam_examples.png`, mục 5 (Nhận xét) |
| Phân tích 10 ca sai | `reports/ca_du_doan_sai.png`, mục 5 (Nhận xét) |
| Cảnh báo y tế trong README | Đầu file này |

## 7. Mở rộng (chưa thực hiện)

1. So sánh ResNet50 · EfficientNetB0 · DenseNet121.
2. Hiệu chuẩn xác suất (calibration).
3. Ước lượng độ bất định bằng MC Dropout — ca model "không chắc" thì chuyển bác sĩ ngay.
