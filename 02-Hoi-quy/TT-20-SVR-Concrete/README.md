# TT-20 — SVR: Dự đoán cường độ chịu nén bê tông trước khi đổ móng

Hồi quy phi tuyến bằng **Support Vector Regression** trên bộ *Concrete Compressive Strength* (UCI).
Mục tiêu: ước lượng cường độ (MPa) ngay từ tỉ lệ phối trộn, thay vì chờ đủ 28 ngày mới nén mẫu.

---

## 1. Chạy thử trong 3 lệnh

```bash
pip install -r requirements.txt
python src/train.py          # tự tải dữ liệu UCI, chạy toàn bộ 13 bước thí nghiệm
python src/predict.py --cement 380 --water 170 --age 28
```

Nếu máy không ra được Internet: tải thủ công tại
<https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength>
rồi đặt `Concrete_Data.xls` vào thư mục `data/` (hoặc dùng `--data <đường_dẫn>`).

Chạy từng phần:

```bash
python src/train.py --steps 1 3            # EDA + baseline
python src/train.py --steps 4 5 6          # chuẩn hoá + so sánh kernel
python src/train.py --steps 7 8 9          # GridSearchCV + model cuối
python src/train.py --steps 10 11 12 13    # ablation, so sánh model, scalability, an toàn
```

Notebook: `notebooks/svr_concrete.ipynb` (chạy lại đúng các bước trên, có diễn giải).

---

## 2. Cấu trúc

```
TT-20-SVR-Concrete/
├── README.md
├── requirements.txt
├── data/                       ← dữ liệu UCI (tự tải về, không commit)
├── src/
│   ├── data.py                 ← tải dữ liệu + đặc trưng từ kiến thức miền
│   ├── train.py                ← toàn bộ 13 bước thí nghiệm, xuất hình + metrics.json
│   └── predict.py              ← suy luận từ model đã lưu
├── notebooks/svr_concrete.ipynb
├── models/svr_pipeline.joblib  ← sinh ra sau khi train
└── reports/                    ← hình + metrics.json sinh ra sau khi train
    ├── eda_scatter.png, correlation_heatmap.png, age_distribution.png
    ├── scale_vs_noscale.png, kernel_comparison.png, C_gamma_heatmap.png
    ├── residual_analysis.png, domain_feature_gain.png
    ├── model_comparison.png, thoi_gian_train.png, safety_quantile.png
    └── metrics.json
```

---

## 3. ⭐ ĐẶC TRƯNG TỪ KIẾN THỨC MIỀN

Đây là phần đóng góp lớn nhất vào độ chính xác — lớn hơn cả việc tinh chỉnh siêu tham số.
Model không thể tự nghĩ ra những tỉ lệ này từ 8 cột gốc.

| Đặc trưng | Công thức | Vì sao |
|---|---|---|
| `water_cement_ratio` | `Water / Cement` | **Định luật Abrams** — yếu tố số 1 quyết định cường độ. w/c càng thấp, cường độ càng cao theo hàm mũ. |
| `water_binder_ratio` | `Water / (Cement + Slag + FlyAsh)` | Xỉ lò cao và tro bay cũng là chất kết dính; tỉ lệ nước/chất-kết-dính phản ánh đúng hơn w/c thuần. |
| `total_binder` | `Cement + Slag + FlyAsh` | Tổng lượng chất kết dính trong 1 m³. |
| `log_age` | `log1p(Age)` | Cường độ tăng gần **tuyến tính theo log(tuổi)**, không theo tuổi. Cột `Age` lệch phải rất mạnh (1–365 ngày, đa số là 28). |
| `aggregate_binder_ratio` | `(Coarse + Fine) / binder` | Tỉ lệ cốt liệu trên chất kết dính — phản ánh độ "giàu" của hỗn hợp. |
| `sp_binder_ratio` | `Superplasticizer / binder` | Liều phụ gia siêu dẻo chỉ có ý nghĩa khi so với lượng chất kết dính. |

Bước 10 của `train.py` đo trực tiếp mức cải thiện: huấn luyện cùng một cấu hình SVR trên
(a) 8 đặc trưng gốc, (b) đầy đủ nhưng bỏ `water_cement_ratio`, (c) đầy đủ — rồi in bảng RMSE.

Ngoài ra `load_raw()` **bỏ các dòng trùng lặp hoàn toàn** (bộ UCI có ~25 dòng trùng).
Nếu giữ lại, cùng một mẻ trộn có thể rơi vào cả train lẫn test → rò rỉ, R² bị thổi phồng.

---

## 4. Vì sao SVR bắt buộc chuẩn hoá CẢ X LẪN y

`epsilon` được hiểu trên **thang đo của y**. Với y tính bằng MPa (2–83, độ lệch chuẩn ≈ 16),
`epsilon=0.1` mặc định gần như bằng 0:

```
   Không scale y  →  ε = 0.1 MPa  ≈ 0.6% độ lệch chuẩn
                  →  gần như MỌI điểm nằm ngoài ống
                  →  mất sạch lợi ích ε-insensitive, số support vector ≈ 100% dữ liệu

   Scale y (std=1) →  ε = 0.1  ≈ ±1.6 MPa  → hợp lý cho bài toán bê tông
                  →  số support vector giảm mạnh, model gọn và tổng quát hoá tốt hơn
```

```python
svr = Pipeline([("scale", StandardScaler()),
                ("svr", SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1))])
model = TransformedTargetRegressor(regressor=svr, transformer=StandardScaler())
```

`TransformedTargetRegressor` scale y khi fit và **tự nghịch đảo khi predict**, nên đầu ra vẫn là MPa.
Bước 4+5 in ra bảng so sánh 3 cấu hình (không scale / chỉ scale X / scale cả hai) kèm số support vector.

---

## 5. Ba siêu tham số

| | Ý nghĩa | Tăng lên thì |
|---|---|---|
| `C` | Mức phạt điểm nằm ngoài ống | Bám sát dữ liệu hơn, dễ overfit |
| `epsilon` | Bề rộng ống (trên thang y đã scale) | Ít support vector hơn, model đơn giản hơn, bias cao hơn |
| `gamma` | Tầm ảnh hưởng của 1 điểm (RBF) | Ranh giới ngoằn ngoèo, overfit nặng |

GridSearchCV quét `C ∈ {1,10,100,1000} × gamma ∈ {scale,0.01,0.1,1} × epsilon ∈ {0.01,0.1,0.5}`
(48 tổ hợp × 5-fold), chấm điểm bằng RMSE. Heatmap `C_gamma_heatmap.png` lấy epsilon tốt nhất
cho từng ô (C, gamma).

---

## 6. Các bước thí nghiệm (khớp mục 5 của đề)

| Bước | Nội dung | Đầu ra |
|---|---|---|
| 1 | EDA: scatter từng đặc trưng vs cường độ, heatmap tương quan, phân bố `Age` | `eda_scatter.png`, `correlation_heatmap.png`, `age_distribution.png` |
| 2 | Tạo đặc trưng miền | `src/data.py::add_domain_features` |
| 3 | Baseline: DummyRegressor + Linear Regression | bảng in ra |
| 4 | SVR **không** scale → kết quả rất tệ | `scale_vs_noscale.png` |
| 5 | SVR scale cả X và y → so sánh, chứng minh | `scale_vs_noscale.png` |
| 6 | So sánh kernel linear / poly(2) / poly(3) / rbf | `kernel_comparison.png` |
| 7 | GridSearchCV 48 tổ hợp | bảng in ra |
| 8 | Heatmap RMSE theo (C, gamma) | `C_gamma_heatmap.png` |
| 9 | Đếm support vector (số lượng + %), phân tích phần dư | `residual_analysis.png` |
| 10 | Đo hiệu quả đặc trưng miền: RMSE trước vs sau | `domain_feature_gain.png` |
| 11 | SVR vs XGBoost vs Random Forest vs LinearSVR (RMSE + thời gian) | `model_comparison.png` |
| 12 | Thời gian train khi nhân dữ liệu 1×→10×, ước lượng bậc O(nᵏ) | `thoi_gian_train.png` |
| 13 | Mở rộng: phân tích an toàn kết cấu + dự báo phân vị 10% | `safety_quantile.png` |

Tất cả số liệu được ghi vào `reports/metrics.json` để đối chiếu lại.

---

## 7. ⚠️ An toàn kết cấu — điều quan trọng nhất của bài toán này

SVR (như mọi mô hình hồi quy tối thiểu hoá sai số đối xứng) ước lượng **trung bình có điều kiện**,
nên khoảng 50% số mẻ sẽ bị dự báo **cao hơn** thực tế. Trong xây dựng, đây là sai lầm nguy hiểm:
mẻ bê tông yếu hơn dự báo → kết cấu không đạt → đục phá sau khi đã đổ móng.

Bước 13 đo trực tiếp rủi ro này (đếm số mẫu bị dự báo vượt thực tế > 5 MPa) và so sánh với
`GradientBoostingRegressor(loss="quantile", alpha=0.1)` — cho **cận dưới an toàn** thay vì trung bình.
Khuyến nghị vận hành: dùng SVR để ước lượng và **phân vị thấp để ra quyết định đổ móng**.

---

## 8. Mức tham chiếu & tiêu chí hoàn thành

Trên bộ UCI gốc, mức kỳ vọng: **R² test ≈ 0.88–0.92 · RMSE ≈ 4.5–5.5 MPa**
(XGBoost thường nhỉnh hơn SVR một chút).

- [x] Bảng so sánh CÓ/KHÔNG chuẩn hoá (cả X và y) + số support vector từng cấu hình
- [x] Đặc trưng miền `water_cement_ratio` + đo mức cải thiện (bước 10)
- [x] So sánh kernel (linear / poly bậc 2, 3 / rbf)
- [x] Heatmap C × gamma
- [x] Báo cáo số support vector và tỉ lệ %
- [x] Thí nghiệm thời gian train theo kích thước dữ liệu
- [x] `R² test > 0.88` — in cảnh báo ĐẠT/CHƯA ĐẠT ở cuối `train.py`
- [x] Nêu hạn chế (mục 9)

---

## 9. Cạm bẫy & hạn chế

| Cạm bẫy | Hậu quả | Cách xử lý trong repo |
|---|---|---|
| Chỉ scale X, quên scale y | ε=0.1 vô nghĩa với y đơn vị MPa | `TransformedTargetRegressor` |
| Dùng SVR cho dữ liệu lớn | Train hàng giờ, có khi không hội tụ | Bước 12 đo và chứng minh ~O(n²) |
| `gamma` quá lớn | Overfit nặng | GridSearchCV + heatmap |
| Bỏ qua kiến thức miền | Mất đặc trưng mạnh nhất | Mục 3 + bước 10 |
| Dự báo cao hơn thực tế | Rủi ro an toàn kết cấu | Bước 13, dự báo phân vị |
| Giữ dòng trùng lặp | Rò rỉ train/test | `drop_duplicates()` trong `load_raw` |

**Hạn chế cốt lõi của SVR:**
1. **Không giải thích được từng dự đoán.** Không có hệ số như hồi quy tuyến tính, cũng không
   có `feature_importances_` như cây. Muốn giải thích phải gắn thêm SHAP (KernelExplainer, rất chậm) hoặc PDP.
2. **Không mở rộng được.** Độ phức tạp ~O(n²)–O(n³); trên 50.000 dòng là bất khả thi.
   `LinearSVR` nhanh hơn nhiều bậc nhưng mất khả năng phi tuyến (bước 11 đo chính xác mức đánh đổi).
3. **Nhạy với siêu tham số** — bắt buộc phải grid search, không dùng được mặc định.

---

**Tham khảo:** [Buổi 4 — SVM](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-04-LogReg-SVM-Metrics/Tai-Lieu) ·
[Buổi 13 — Math Regression nâng cao](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao/Tai-Lieu)
