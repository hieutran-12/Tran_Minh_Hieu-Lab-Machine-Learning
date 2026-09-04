# TT-11 — Linear Regression: Định giá nhà ở (California Housing)

## 1. Cách chạy

```bash
pip install -r requirements.txt

# Cách 1: chạy script (huấn luyện + lưu model + lưu toàn bộ ảnh vào reports/)
python src/train.py

# Cách 2: chạy notebook (khuyến nghị — có giải thích từng bước)
jupyter notebook notebooks/linear_regression_housing.ipynb
```

Dữ liệu được tải tự động qua `sklearn.datasets.fetch_california_housing`
(cần kết nối mạng ở lần chạy đầu để scikit-learn tải và cache dữ liệu).

## 2. Dữ liệu

- Bộ dữ liệu: **California Housing** (20.640 dòng × 8 đặc trưng), lấy đúng theo
  yêu cầu đề bài qua `fetch_california_housing`. **Không** dùng bộ Boston Housing.
- Nhãn: `MedHouseVal` (giá trung vị, đơn vị 100.000 USD).

## 3. Kiểm tra 4 giả định hồi quy tuyến tính

| Giả định | Cách kiểm tra trong notebook | Kết quả |
|---|---|---|
| ① Tuyến tính | Scatter `MedInc` vs giá, residual plot | Xem `reports/scatter_medinc.png`, `reports/residual_plot.png` |
| ② Độc lập | Mỗi dòng là một khu vực dân cư khác nhau | Hợp lý theo thiết kế dữ liệu |
| ③ Phương sai đều | Residual vs fitted plot | Xem `reports/residual_plot.png` — so sánh thêm với `residual_plot_log.png` sau log-transform |
| ④ Sai số chuẩn | Q-Q plot | Xem `reports/qq_plot.png` |

Nhận xét chi tiết (dạng phễu, độ lệch phân phối...) được ghi trực tiếp trong notebook
ngay dưới từng biểu đồ — điền/cập nhật sau khi chạy trên dữ liệu thật.

## 4. Các bước đã thực hiện (đối chiếu mục 5 của đề bài)

- [x] Nạp dữ liệu, `describe()` để phát hiện outlier `AveOccup`, `AveRooms`
- [x] Phát hiện nhãn bị cắt ngọn tại 5.0 (đếm số dòng bị ảnh hưởng)
- [x] EDA: scatter `MedInc` vs giá · heatmap tương quan · bản đồ giá theo toạ độ
- [x] Baseline: `DummyRegressor(strategy='mean')`
- [x] Linear Regression cơ bản (RMSE, MAE, R²)
- [x] Residual plot (phần dư vs giá dự đoán)
- [x] Q-Q plot kiểm tra phân phối phần dư
- [x] Thử dự đoán `log(giá)` và so sánh residual plot
- [x] Kiểm tra đa cộng tuyến bằng VIF
- [x] Feature engineering: `rooms_per_household`, khoảng cách tới San Francisco / Los Angeles
- [x] Bảng hệ số đã chuẩn hoá + diễn giải 3 yếu tố ảnh hưởng mạnh nhất
- [x] So sánh với Ridge và Random Forest Regressor

## 5. Bẫy dữ liệu đã xử lý

1. **Nhãn bị cắt ngọn ở 5.0** — được đếm và nêu rõ trong hạn chế; model không dự
   đoán được nhà đắt hơn ngưỡng này.
2. **Outlier cực đoan** ở `AveRooms`, `AveOccup` (có dòng `AveOccup > 1000`) —
   được clip theo phân vị 99 trước khi huấn luyện.
3. **Toạ độ (Latitude/Longitude) phi tuyến với giá** — bổ sung đặc trưng khoảng
   cách tới trung tâm (`dist_to_nearest_city`) để giảm bớt hạn chế của mô hình tuyến tính.

## 6. Kết quả tham chiếu

Chạy `python src/train.py` hoặc notebook để tự tạo bảng so sánh model tại
`reports/model_comparison.csv`. Mức tham chiếu theo đề bài: R² ≈ 0,58–0,61,
RMSE ≈ 0,72–0,75 (đơn vị 100k USD) cho Linear Regression; Random Forest
thường đạt R² ≈ 0,80 — chênh lệch này là cái giá của tính giải thích được.

## 7. Hạn chế

- Nhãn bị cắt ngọn tại 5.0 khiến model đánh giá thấp phân khúc nhà cao cấp.
- Quan hệ toạ độ–giá là phi tuyến, mô hình tuyến tính bắt kém dù đã thêm đặc trưng khoảng cách.
- Hệ số hồi quy phản ánh tương quan, không phải quan hệ nhân quả.

## 8. Cấu trúc thư mục

```
TT-11-LinearRegression/
├── README.md
├── requirements.txt
├── notebooks/linear_regression_housing.ipynb
├── src/train.py
├── models/              ← sinh ra sau khi chạy (lr_pipeline.joblib)
└── reports/             ← sinh ra sau khi chạy (ảnh + bảng CSV)
```

## 9. Mở rộng (tự làm thêm nếu muốn nâng điểm)

1. Dùng `statsmodels.OLS` để lấy p-value và khoảng tin cậy cho từng hệ số.
2. Thêm đặc trưng tương tác (`MedInc × HouseAge`) và so sánh R².
3. Hồi quy có trọng số (WLS) để xử lý phương sai không đều.
