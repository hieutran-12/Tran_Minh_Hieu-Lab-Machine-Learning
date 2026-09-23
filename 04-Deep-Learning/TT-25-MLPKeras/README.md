# TT-25 — MLP với Keras

## Chấm điểm khách hàng tiềm năng mua bảo hiểm ô tô

Bài này xây dựng pipeline end-to-end so sánh **MLP (Keras/TensorFlow)** với baseline
**LightGBM** để xếp hạng 380.000 khách hàng đang mua bảo hiểm sức khoẻ theo khả năng
quan tâm mua thêm bảo hiểm ô tô, phục vụ đội telesales gọi ~3.000 cuộc/ngày.

## 1. Cài đặt

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Tải dữ liệu

Dữ liệu **không được đính kèm** trong gói này (theo điều khoản của Kaggle). Hãy tải về:

1. Truy cập: https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction
2. Tải file `train.csv`
3. Đặt vào: `data/train.csv`

File cần có đúng các cột: `id, Gender, Age, Driving_License, Region_Code, Previously_Insured,
Vehicle_Age, Vehicle_Damage, Annual_Premium, Policy_Sales_Channel, Vintage, Response`.

## 3. Chạy pipeline

**Cách 1 — script (khuyến nghị, chạy toàn bộ 12 bước trong ~vài phút đến vài chục phút
tuỳ CPU/GPU):**

```bash
python -m src.train --data data/train.csv
```

**Cách 2 — notebook (chạy từng bước, xem biểu đồ trực tiếp):**

```bash
jupyter notebook notebooks/mlp_keras_insurance.ipynb
```

## 4. Kết quả sau khi chạy

```
models/
├── mlp_plain.keras                 # MLP không Dropout/BatchNorm
├── mlp_bn_dropout.keras            # MLP + Dropout/BatchNorm  ⭐ model chính
├── mlp_arch_*.keras                # 3 kiến trúc so sánh
├── mlp_class_weight.keras / mlp_no_class_weight.keras
├── mlp_onehot_highcard.keras / mlp_embedding.keras
reports/
├── eda_summary.json                # Tỉ lệ quan tâm theo Previously_Insured / Vehicle_Damage / tuổi
├── learning_curves_plain.png / learning_curves_bn_dropout.png
├── kien_truc_comparison.png
├── embedding_vs_onehot.png
├── pr_curve_lightgbm.png / pr_curve_mlp_bn_dropout.png
├── threshold.json                  # Ngưỡng xác suất ứng với Precision@3000
└── comparison_summary.md           # ⭐ Bảng kết luận MLP vs LightGBM — TỰ ĐỘNG SINH khi chạy pipeline
```

> **Lưu ý:** `reports/comparison_summary.md` được sinh lại mỗi lần chạy `src/train.py` trên
> dữ liệu thật (380.109 dòng) — đây là nguồn kết luận chính thức, không phải số liệu cố định
> viết sẵn trong README này. Pipeline đã được kiểm thử end-to-end (đọc dữ liệu → tiền xử lý →
> baseline → 5 nhóm thử nghiệm MLP → ngưỡng Precision@3000 → bảng kết luận) và chạy không lỗi;
> mức tham chiếu PR-AUC ~0,30–0,35 nêu trong đề bài là mức kỳ vọng khi chạy trên bộ dữ liệu
> Kaggle thật.

## 5. Cấu trúc mã nguồn

| File                                  | Nội dung                                                                                       |
| ------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `src/data.py`                         | Đọc dữ liệu, tiền xử lý (log1p, one-hot, scaling, remap ID cho embedding), chia train/val/test |
| `src/model.py`                        | Kiến trúc MLP (có/không Dropout+BatchNorm), MLP+Embedding, 3 callback bắt buộc                 |
| `src/evaluate.py`                     | PR-AUC, Precision@K, learning curves, PR curve, biểu đồ so sánh                                |
| `src/train.py`                        | Orchestrator — chạy toàn bộ pipeline, sinh `reports/comparison_summary.md`                     |
| `notebooks/mlp_keras_insurance.ipynb` | Notebook chạy từng bước, dùng chung logic với `src/`                                           |

## 6. Metric sử dụng

⚠️ **Không dùng accuracy** — vì chỉ ~12,3% khách quan tâm, đoán toàn bộ "không quan tâm"
đã đạt ~87,7% accuracy mà vô dụng. Toàn bộ pipeline đánh giá bằng:

- **PR-AUC** (Precision-Recall AUC) — phù hợp với dữ liệu lệch mạnh
- **Precision@3000** — precision khi chỉ chọn top 3.000 khách theo xác suất dự đoán,
  đúng năng lực gọi điện thực tế của đội telesales

## 7. Mở rộng (chưa triển khai trong gói này)

1. **TabNet** hoặc **FT-Transformer** — kiến trúc DL thiết kế riêng cho dữ liệu bảng
2. Xếp chồng (stacking): dùng đầu ra MLP + LightGBM làm đầu vào cho meta-model
3. Chuyển sang PyTorch và so sánh trải nghiệm phát triển

**Tham khảo:** [Buổi 7 — Neural Network & Keras](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-07-Neural-Network/Tai-Lieu/ly_thuyet_chi_tiet_buoi_07.md)
