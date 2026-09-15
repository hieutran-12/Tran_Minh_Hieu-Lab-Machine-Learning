# TT-19 — XGBoost Regressor
## Dự báo nhu cầu thuê xe đạp công cộng theo giờ để điều phối xe

---

## 0. CÀI ĐẶT & CHẠY

```bash
# 1. Cài thư viện
pip install -r requirements.txt

# 2. Tải dataset gốc và đặt vào data/hour.csv
#    https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
#    (giải nén Bike-Sharing-Dataset.zip, copy hour.csv vào data/)

# 3. Chạy pipeline đầy đủ (train + đánh giá + lưu report/model)
python src/train.py --data data/hour.csv

# Tuỳ chọn: giảm số vòng RandomizedSearchCV để chạy nhanh hơn khi thử nghiệm
python src/train.py --data data/hour.csv --search-iter 15
```

Sau khi chạy `train.py`, kết quả được ghi vào:
- `models/xgb_bike.json` — model đã train
- `reports/*.png`, `reports/model_comparison.csv`, `reports/summary.json` — biểu đồ và số liệu

Hoặc mở `notebooks/xgboost_bike_demand.ipynb` để chạy từng bước tương tác (Jupyter).

---

## 1. CHỨNG MINH RÒ RỈ

Dataset gốc có tính chất: **`cnt = casual + registered`** — cột nhãn đúng bằng tổng của
hai cột đặc trưng khác trong bảng dữ liệu.

`src/features.py::prove_leakage()` train một hồi quy tuyến tính đơn giản **chỉ dùng
`casual` và `registered`** làm đặc trưng đầu vào để minh hoạ điều này:

```python
from features import load_data, prove_leakage

df = load_data("data/hour.csv")
r2 = prove_leakage(df)
print(r2)   # ≈ 1.000000
```

Vì mô hình chỉ cần cộng hai số lại là ra đáp án chính xác tuyệt đối, R² xấp xỉ **1.0**
không phản ánh khả năng dự báo thật — đây là **rò rỉ dữ liệu (data leakage)**: trong
thực tế vận hành, `casual` và `registered` của giờ *sắp tới* là thứ ta **chưa biết**
(chính là một phần của cái cần dự báo), nên không thể dùng làm đầu vào.

**Bước xử lý:** `src/features.py::build_feature_matrix(df, drop_leak=True)` loại bỏ
hoàn toàn `casual` và `registered` (cùng với `instant`, `dteday` dạng thô) trước khi
đưa vào model chính thức. Toàn bộ các bước huấn luyện trong `train.py` và notebook
đều dùng `drop_leak=True`; hàm `prove_leakage()` chỉ được gọi riêng để chứng minh,
không tham gia vào pipeline dự báo thật.

---

## 2. SO SÁNH VỚI BASELINE NAIVE

Baseline: dự báo `cnt` của một (ngày, giờ) bằng `cnt` của **cùng giờ, tuần trước**
(lệch 168 giờ). Với dữ liệu có tính chu kỳ tuần mạnh (giờ cao điểm đi làm, ngày
cuối tuần), baseline này thường **rất khó thắng**.

Sau khi chạy `train.py` trên dữ liệu thật, bảng so sánh được ghi tự động vào
`reports/model_comparison.csv` (bao gồm cả RMSE/R²/MAE trên tập test), với cấu trúc:

| model | rmse_test | r2_test | mae_test |
|---|---|---|---|
| Baseline naive | ... | ... | ... |
| Random Forest (TT-17) | ... | ... | ... |
| Gradient Boosting (TT-18) | ... | ... | ... |
| XGBoost (TT-19) | ... | ... | ... |

Tiêu chí đạt: **RMSE của XGBoost phải thấp hơn RMSE của baseline naive** (log ở cuối
`train.py`: `Xgboost thắng baseline naive: True/False`). Mức tham chiếu theo đề bài:
RMSE ~40–55 lượt/giờ, R² ~0.93–0.95 trên tập test chia theo thời gian.

*(Điền số liệu thật vào bảng trên sau khi chạy `train.py` với `data/hour.csv` tải từ
UCI — bảng mẫu ở trên sẽ được sinh tự động tại `reports/model_comparison.csv`.)*

---

## 3. HAI BẪY ĐÃ XỬ LÝ

| Bẫy | Xử lý trong code |
|---|---|
| Rò rỉ trực tiếp (`cnt = casual + registered`) | `build_feature_matrix(drop_leak=True)` bỏ cả hai cột; `prove_leakage()` minh hoạ riêng |
| Chia dữ liệu ngẫu nhiên trên chuỗi thời gian | `time_split()` chia theo `yr`/`mnth`, không shuffle: năm 1 → train, 9 tháng đầu năm 2 → validation, 3 tháng cuối → test |

## 4. ĐẶC TRƯNG CHU KỲ

`hr` (0–23), `mnth` (1–12), `weekday` (0–6) được mã hoá thêm dạng sin/cos
(`add_cyclical_features()` trong `src/features.py`) để mô hình học được tính liên
tục vòng tròn (giờ 23 và giờ 0 liền kề nhau). Các cột số nguyên gốc vẫn được giữ
song song vì XGBoost (dựa trên cây quyết định) có thể khai thác thêm thông tin thứ
tự từ đó.

## 5. NHÃN LỆCH PHẢI (log1p)

`train.py` tự động train cả hai phương án (có/không `log1p(cnt)`) và **chọn phương
án cho RMSE validation thấp hơn** để dùng cho tập test — không giả định trước
phương án nào tốt hơn.

## 6. XỬ LÝ DỰ ĐOÁN ÂM

Mọi output dự đoán đều được `np.clip(pred, 0, None)` vì số lượt thuê không thể âm.

---

## 7. CẤU TRÚC THƯ MỤC

```
TT-19-XGBoostRegressor-Thuy/
├── README.md
├── requirements.txt
├── notebooks/xgboost_bike_demand.ipynb   # chạy từng bước, có biểu đồ inline
├── src/
│   ├── features.py    # load_data, prove_leakage, cyclical encoding, time_split, baseline
│   └── train.py        # pipeline end-to-end: EDA → train → tuning → SHAP → so sánh model
├── data/                # đặt hour.csv (tự tải) vào đây — KHÔNG kèm sẵn trong gói này
├── models/              # xgb_bike.json được ghi ra sau khi chạy train.py
└── reports/             # các biểu đồ + summary.json + model_comparison.csv sau khi chạy train.py
```

## 8. MỞ RỘNG (chưa triển khai trong pipeline chính)

1. Hàm loss bất đối xứng (phạt dự báo *thiếu* nặng gấp đôi dự báo *thừa*) — cần
   custom objective trong XGBoost (`obj=...`), phù hợp vì thiếu xe tệ hơn thừa xe
   về mặt vận hành.
2. Dự báo nhiều bước (1h / 6h / 24h tới) — đánh giá độ chính xác giảm dần theo horizon.
3. Dùng dữ liệu thời tiết *dự báo* thay vì thời tiết *thực tế đã xảy ra* để sát với
   điều kiện vận hành thật (tại thời điểm dự báo, ta chưa biết thời tiết chính xác).

**Tham khảo:** [Buổi 9 — Time Series](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-09-TimeSeries) · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao)
