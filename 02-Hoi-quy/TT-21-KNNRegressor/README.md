# TT-21 — KNN REGRESSOR
## Định giá nhà theo "các căn tương tự trong khu vực" — đúng cách thẩm định viên làm

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · Buổi 13 — Math Regression Nâng cao |
| 🧠 **Nhóm** | Hồi quy dựa trên khoảng cách |
| 🔧 **Thuật toán** | `KNeighborsRegressor` (scikit-learn) |
| 📦 **Dữ liệu** | California Housing — 20.640 dòng × 8 đặc trưng (tải tự động từ sklearn) |
| 🎯 **Nhãn** | `MedHouseVal` — giá nhà trung vị, đơn vị 100.000 USD |
| 🏭 **Lĩnh vực** | Bất động sản · Thẩm định giá |

---

## 1. Chạy như thế nào

```bash
pip install -r requirements.txt

python src/train.py          # chạy đầy đủ 11 bước (~2–4 phút), xuất model + 4 biểu đồ
python src/train.py --nhanh  # bản rút gọn (~1 phút) để kiểm tra nhanh

jupyter lab notebooks/knn_regressor_housing.ipynb   # bản phân tích có diễn giải
```

> Lần chạy đầu tiên sklearn sẽ **tự tải** California Housing (~1,5 MB) về
> `~/scikit_learn_data/` — cần kết nối mạng đúng một lần, các lần sau dùng cache.

Sau khi chạy, thư mục `reports/` và `models/` sẽ được sinh ra đầy đủ; bảng kết quả ở
mục 5 của README này **được `src/train.py` tự động ghi đè** bằng số liệu thật trên máy bạn.

---

## 2. Cấu trúc dự án

```
TT-21-KNNRegressor-<HoTen>/
├── README.md                              ← bảng so sánh Linear vs KNN vs RF (mục 5)
├── requirements.txt
├── notebooks/knn_regressor_housing.ipynb  ← phân tích có diễn giải, chạy end-to-end
├── src/
│   ├── data.py                            ← nạp dữ liệu, lọc outlier, dựng pipeline, tra "căn tương tự"
│   └── train.py                           ← chạy đủ 11 bước, xuất biểu đồ + model + bảng kết quả
├── models/
│   ├── knn_pipeline.joblib                ← pipeline đã fit (scaler + trọng số vị trí + KNN)
│   └── cau_hinh.json                       ← siêu tham số chốt + chỉ số + ví dụ căn tương tự
└── reports/
    ├── rmse_theo_K.png · trong_so_vi_tri.png
    ├── can_tuong_tu_vi_du.png · thoi_gian_predict.png
    ├── rmse_theo_K.csv · ket_qua.md
```

> `models/` và `reports/` trống khi bàn giao (chỉ có `.gitkeep`) — toàn bộ nội dung được
> sinh ra khi chạy `src/train.py` trên dữ liệu thật.

**Nạp lại model đã lưu** — cần thư mục gốc dự án nằm trong `sys.path` vì pipeline dùng lớp
`TrongSoViTri` định nghĩa trong `src/data.py`:

```python
import sys, joblib, pandas as pd
sys.path.append("/duong/dan/toi/TT-21-KNNRegressor-<HoTen>")

knn = joblib.load("models/knn_pipeline.joblib")
can = pd.DataFrame([[3.2, 25, 5.4, 1.1, 1200, 3.0, 34.05, -118.24]],
                   columns=["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                            "Population", "AveOccup", "Latitude", "Longitude"])
print(knn.predict(can))   # đơn vị 100.000 USD
```

---

## 3. Hướng tiếp cận

```python
Pipeline([
    ("scale", StandardScaler()),        # BẮT BUỘC — Population hàng nghìn sẽ áp đảo mọi đặc trưng
    ("vitri", TrongSoViTri(w=2)),       # nhân Lat/Lon sau khi scale → ưu tiên hàng xóm ĐỊA LÝ
    ("knn",   KNeighborsRegressor(n_neighbors=K, weights="distance", n_jobs=-1)),
])
```

**Nguyên tắc đánh giá:** K, `weights`, `metric` và hệ số vị trí đều được chọn bằng **tập
validation tách từ train** (rồi kiểm chứng lại bằng 5-fold CV). Tập test chỉ dùng để
báo cáo — không dùng để chọn tham số.

| Bước | Nội dung |
|---|---|
| 1 | Nạp dữ liệu, lọc outlier `AveRooms > 15`, `AveBedrms > 4`, `AveOccup > 10` |
| 2 | Baseline: `DummyRegressor` + `LinearRegression` (đối chiếu TT-11) |
| 3 | KNN **không** chuẩn hoá — ghi lại để thấy hậu quả |
| 4 | KNN **có** chuẩn hoá, K = 5 |
| 5 | Đường cong RMSE train/val/test theo K = 1…50 cho cả `uniform` và `distance` |
| 6–7 | So sánh `uniform` vs `distance`, `euclidean` vs `manhattan` |
| 8 | ⭐ Thí nghiệm trọng số vị trí: nhân Lat/Lon × {1, 2, 3, 5} |
| 9 | ⭐ In ra 5 "căn tương tự" cho 3 căn nhà + vẽ lên bản đồ |
| 10 | ⭐ Bảng so sánh Linear vs KNN vs Random Forest kèm **thời gian dự đoán** |
| 11 | ⚠️ Đo thời gian dự đoán khi tập train tăng 1× / 5× / 10× |

---

## 4. Bốn điểm cần rút ra

**a) Chuẩn hoá không phải tuỳ chọn.** `Population` có đơn vị hàng nghìn còn `AveBedrms`
quanh 1, nên khoảng cách Euclid khi chưa scale gần như *chỉ còn là chênh lệch Population*.
KNN không chuẩn hoá thua cả Linear Regression; thêm đúng một dòng `StandardScaler` là đảo ngược kết quả.

**b) Không được chọn K bằng RMSE train.** Với `K = 1`, hàng xóm gần nhất của một điểm train
chính là bản thân nó → **RMSE train = 0**, mô hình thuộc lòng dữ liệu chứ không tổng quát hoá.
Với `weights='distance'` thì RMSE train = 0 với **mọi K** (khoảng cách 0 ⇒ trọng số vô hạn).
Đường cong trong `reports/rmse_theo_K.png` vẽ cả hai trường hợp để thấy rõ điều này.

**c) Vị trí đáng giá hơn số phòng.** Sau chuẩn hoá mọi đặc trưng có tiếng nói ngang nhau —
nhưng với bất động sản thì không nên như vậy. Nhân `Latitude`/`Longitude` với hệ số 2–3 ép KNN
tìm hàng xóm **địa lý** trước, và RMSE giảm thêm: một bằng chứng định lượng cho câu
*location, location, location* (`reports/trong_so_vi_tri.png`).

**d) Ưu thế riêng: KNN chỉ ra được căn cứ.** Với mỗi căn cần định giá, mô hình liệt kê
đúng 5 căn đã dùng để ra con số — hệt một chứng thư thẩm định:
*"Căn này ước tính 4,2 tỷ, dựa trên 5 căn tương tự đã bán gần đây: …"*.
Random Forest cho RMSE tốt hơn nhưng **không làm được điều này** (`reports/can_tuong_tu_vi_du.png`).

---

## 5. Kết quả

<!-- KET-QUA:BEGIN --> <!-- Tự sinh bởi src/train.py — đừng sửa tay -->

> Bảng dưới đây là **mức tham chiếu kỳ vọng**. Chạy `python src/train.py` để thay bằng
> số liệu thật đo trên máy bạn (script tự ghi đè đúng khối này).

| Mô hình | RMSE test | R² test | Dự đoán 1 căn | Giải thích được? |
|---|---|---|---|---|
| Dummy (giá trung bình) | ~1,15 | ~0,00 | tức thì | — |
| Linear Regression (TT-11) | ~0,72 | ~0,60 | tức thì | Có — hệ số từng đặc trưng |
| KNN K=5 — KHÔNG chuẩn hoá | ~1,05 | ~0,15 | chậm | Có |
| **KNN K≈10, distance (TT-21)** | **~0,61** | **~0,70–0,75** | chậm | **Có — chỉ ra ĐÚNG các căn tương tự** |
| Random Forest | ~0,50 | ~0,81 | trung bình | Khó — chỉ có feature importance |

<!-- KET-QUA:END -->

**Tiêu chí quan trọng nhất của đề bài:** KNN (có chuẩn hoá) **phải thắng** Linear Regression
về RMSE. Nếu không, gần như chắc chắn là quên `StandardScaler`.

---

## 6. Hạn chế của KNN

| Hạn chế | Biểu hiện trong dự án này |
|---|---|
| Chậm khi dự đoán | Bước 11: thời gian dự đoán tăng gần tuyến tính theo số dòng train |
| Không ngoại suy | Dự đoán luôn nằm trong khoảng giá của tập train → căn đắt bất thường bị kẹp trần (bộ này nhãn còn bị chặn sẵn tại 5.0) |
| Phải lưu toàn bộ dữ liệu | File `.joblib` mang theo cả tập train, không có "công thức" gọn như Linear Regression |
| Nhạy với đặc trưng nhiễu | Mọi đặc trưng vô dụng đều làm méo khoảng cách |
| Nhạy với thang đo | Quên chuẩn hoá là hỏng toàn bộ |

---

## 7. Hướng mở rộng

1. `algorithm='kd_tree'` / `BallTree` — tăng tốc tìm hàng xóm trên dữ liệu lớn.
2. `RadiusNeighborsRegressor` — lấy **mọi** căn trong bán kính X km thay vì K căn cố định,
   sát với quy trình thẩm định thực tế hơn.
3. Dùng KNN sinh đặc trưng *"giá trung bình 10 căn lân cận"* rồi đưa vào XGBoost —
   kết hợp trực giác địa lý của KNN với sức mạnh của boosting.
