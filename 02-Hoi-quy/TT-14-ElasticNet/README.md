# TT-14 — ElasticNet: Dự báo tiêu thụ năng lượng toà nhà

Dự báo **tải sưởi (Y1)** và **tải làm mát (Y2)** của toà nhà từ 8 biến thiết kế,
trong bối cảnh các biến thiết kế **đa cộng tuyến nặng** (diện tích tường/mái/sàn/chiều
cao đều bị ràng buộc hình học lẫn nhau).

## Dữ liệu

- Nguồn: [UCI Energy Efficiency Dataset](https://archive.ics.uci.edu/dataset/242/energy+efficiency), 768 dòng × 8 đặc trưng, 2 nhãn.
- File đang dùng: `data/ENB2012_data.csv` (đã tải sẵn, đầy đủ 768 dòng, không thiếu giá trị).
- Nếu bạn muốn dùng file bạn tự tải từ UCI (`.xlsx` hoặc `.csv`), đặt đè vào `data/`
  rồi chạy:
  ```bash
  python src/train.py --data data/<tên_file_của_bạn>.xlsx
  ```
  `src/data_loader.py` tự nhận diện `.csv`/`.xlsx` và kiểm tra đủ cột X1–X8, Y1, Y2.

## Cách chạy

```bash
pip install -r requirements.txt
python src/train.py                       # dùng data/ENB2012_data.csv mặc định
# hoặc mở notebooks/elasticnet_energy.ipynb để chạy từng bước có giải thích
```

Kết quả (bảng, hình, model) được ghi vào `reports/` và `models/`.

## 1. VIF — bằng chứng đa cộng tuyến

**Bảng đầy đủ (`reports/vif_table.csv`)** — VIF của X2, X3, X4 ra ~10¹⁵:

| feature | VIF |
|---|---|
| X2 (diện tích bề mặt) | ~1.0×10¹⁵ |
| X3 (diện tích tường) | ~1.0×10¹⁵ |
| X4 (diện tích mái) | ~1.0×10¹⁵ |
| X1 (độ gọn tương đối) | 105.5 |
| X5 (chiều cao tổng) | 31.2 |
| X7 (diện tích kính) | 1.0 |

Con số ~10¹⁵ này **kỹ thuật đúng nhưng vô nghĩa để định lượng** — nó chỉ nói lên
"VIF không hội tụ", không nói được mức độ cộng tuyến bao nhiêu. Lý do: trong bộ
Energy Efficiency, `X2 ≈ 2×(X3+X4)/X1` gần như là **hệ thức hình học chính xác**
(một đặc điểm đã biết của bộ dữ liệu mô phỏng này), tức X2 gần như phụ thuộc tuyến
tính tuyệt đối vào X3, X4, X1 chứ không chỉ tương quan cao — nên phần dư khi hồi quy
X2 theo các biến còn lại gần bằng 0 và VIF (= 1/(1−R²)) tràn số.

**Bảng sau khi bỏ X2 (`reports/vif_table_no_X2.csv`)** — có giá trị hữu hạn, đọc được:

| feature | VIF |
|---|---|
| X4 | 211.9 |
| X1 | 105.5 |
| X5 | 31.2 |
| X3 | 27.7 |
| X7 | 1.0 |

Ngay cả sau khi bỏ X2, VIF của X4, X1, X5, X3 vẫn rất cao (>10, có cái >200) — xác
nhận đa cộng tuyến nặng lan ra cả nhóm chứ không chỉ giữa 2 biến, đúng lý do bài chọn
ElasticNet thay vì hồi quy tuyến tính thường hay Lasso.

## 2. Bảng so sánh 3 model (train/test split 80/20, sau CV chọn siêu tham số)

| target | model | số biến giữ | RMSE | R² |
|---|---|---|---|---|
| Y1 (sưởi) | LinearRegression | 14 | 2.872 | 0.9208 |
| Y1 (sưởi) | Ridge | 14 | 2.886 | 0.9201 |
| Y1 (sưởi) | Lasso | **12** | 2.899 | 0.9194 |
| Y1 (sưởi) | **ElasticNetCV** | 14 | **2.876** | **0.9207** |
| Y2 (mát) | LinearRegression | 14 | 3.120 | 0.8950 |
| Y2 (mát) | Ridge | 14 | 3.128 | 0.8944 |
| Y2 (mát) | Lasso | **13** | 3.135 | 0.8939 |
| Y2 (mát) | **ElasticNetCV** | 14 | **3.121** | **0.8949** |

Đề bài tham chiếu **R² ~0,90–0,92 cho Y1** — kết quả ElasticNetCV đạt **R² = 0,9207**,
nằm ngay trong khoảng tham chiếu.

> ⚠️ **Về cột "số biến giữ"**: cột này đếm số hệ số > 1e-6, chỉ có ý nghĩa "chọn biến"
> thật sự với Lasso/ElasticNet — 2 model *có thể* đẩy hệ số về đúng 0. Ridge và
> LinearRegression gần như không bao giờ zero-out một hệ số, nên con số 14 ở 2 dòng
> đó không nên hiểu là "Ridge chọn 14 biến" — đơn giản là Ridge không loại biến nào,
> khác bản chất với việc Lasso/ElasticNet chủ động zero-out.

**Về khoảng cách RMSE giữa 3 model**: chênh lệch giữa Ridge/Lasso/ElasticNet rất nhỏ
(2.876 vs 2.886 vs 2.899 cho Y1) — không nên coi đây là "ElasticNet thắng Ridge/Lasso".
Điểm đáng giá của bài tập này **không nằm ở RMSE ai thấp hơn**, mà ở câu chuyện phía
sau: ElasticNet xử lý nhóm biến tương quan khác hẳn Lasso về mặt hệ số (mục 3) và ổn
định hơn khi lấy mẫu lại (mục 6) — đó mới là lý do chọn ElasticNet trong bài toán có
đa cộng tuyến, chứ không phải vì nó cho RMSE tốt nhất.

`ElasticNetCV` chọn `l1_ratio = 0.1` cho cả 2 nhãn — nghiêng về phía Ridge (ít biến bị
zero-out), phù hợp với thực tế: các biến hình học ở đây đều mang thông tin liên quan
đến nhau, không có biến nào "hoàn toàn vô dụng" cần loại hẳn — máy chọn giữ gần hết,
chỉ phạt độ lớn hệ số nhẹ để chống overfit do cộng tuyến.

## 3. HIỆU ỨNG GOM NHÓM (Lasso vs ElasticNet trên nhóm {X1, X2, X4, X5})

| feature | Lasso_coef | ElasticNet_coef | (Y1 — tải sưởi) |
|---|---|---|---|
| X1 | -4.59 | -6.22 | |
| X2 | **-0.05** | **-3.48** | Lasso gần như loại bỏ X2, ElasticNet giữ lại |
| X4 | -4.92 | -3.64 | |
| X5 | 7.91 | 7.32 | |

Với Y2 (tải làm mát) hiện tượng lặp lại tương tự nhưng nhẹ hơn: Lasso vẫn ép hệ số X2
xuống thấp hơn hẳn 3 biến còn lại trong nhóm (dù không về gần 0 tuyệt đối như ở Y1),
trong khi ElasticNet phân bổ hệ số đều hơn giữa cả 4 biến tương quan. Đây chính là
**grouping effect** — điểm khác biệt cốt lõi giữa Lasso và ElasticNet khi các biến đầu
vào tương quan chặt: Lasso có xu hướng chọn đại diện một biến trong nhóm và bỏ gần hết
phần còn lại (nhạy với dữ liệu, có thể đổi biến "đại diện" khi dữ liệu thay đổi nhẹ —
xem mục 6), còn ElasticNet giữ cả nhóm lại với hệ số san sẻ hợp lý và ổn định hơn.

## 4. Heatmap alpha × l1_ratio

`reports/heatmap_alpha_l1ratio_Y1_heating.png` và `..._Y2_cooling.png` — đã zoom lưới
alpha vào dải 1e-4 đến 1 (thay vì 1e-3 đến 10) để tránh vùng alpha nhỏ bị "bão hoà" một
màu, giúp thấy rõ hơn RMSE tăng dần khi alpha lớn lên; nhãn trục alpha dùng ký hiệu khoa
học (1.0e-04, 1.9e-04, ...) để không bị làm tròn về "0.000". Vùng RMSE thấp nhất tập
trung ở `l1_ratio` nhỏ (0.1–0.3) và alpha rất nhỏ (~1e-3–1e-4), khớp với lựa chọn của
`ElasticNetCV`. Trục và thang màu đã kiểm tra lại, đọc được rõ ràng.

## 5. Y1 vs Y2 — biến nào quan trọng cho sưởi, biến nào cho làm mát

- X5 (chiều cao tổng): hệ số dương lớn nhất ở **cả hai** nhãn — nhà cao hơn tốn năng
  lượng hơn cho cả sưởi lẫn làm mát.
- X7 (diện tích kính) và các mức của X6 (hướng nhà): ảnh hưởng đến **Y2 (làm mát)**
  rõ hơn Y1 — hợp lý vì kính hấp thụ bức xạ mặt trời, ảnh hưởng nhiệt ban ngày (làm
  mát) nhiều hơn nhu cầu sưởi.

## 6. Bootstrap ổn định hệ số (100 lần lấy mẫu lại)

Nhóm biến tương quan {X1, X2, X4, X5} có `coef_std` (độ lệch chuẩn hệ số) cao nhất
trong toàn bộ đặc trưng (xem `reports/bootstrap_stability_*.csv`) — phản ánh đúng bản
chất: khi các biến tương quan chặt, việc "chia" trọng số giữa chúng có thể dao động
giữa các lần lấy mẫu, dù tổng ảnh hưởng của cả nhóm vẫn ổn định. Đây là bằng chứng thứ
hai (cùng với mục 3) cho thấy giá trị thật của ElasticNet trong bài này nằm ở tính ổn
định và cách xử lý nhóm biến, không phải ở việc thắng RMSE.

## 7. Đề xuất thiết kế giảm tải năng lượng

1. **Giảm chiều cao tổng thể (X5)** khi thiết kế cho phép — hệ số dương lớn nhất cho
   cả tải sưởi lẫn tải làm mát.
2. **Tăng độ gọn tương đối (X1)** — khối nhà gọn hơn (tỉ lệ thể tích/diện tích bao che
   cao hơn) giảm diện tích tiếp xúc môi trường ngoài, giảm cả hai loại tải.
3. **Kiểm soát diện tích kính (X7) theo hướng nhà (X6)** — ưu tiên kính nhỏ hơn hoặc
   cách nhiệt tốt hơn ở các mặt hướng nắng nhiều, vì nhóm biến này tác động đến tải
   làm mát rõ hơn tải sưởi.

## Cấu trúc project

```
TT-14-ElasticNet-Hieu/
├── README.md
├── requirements.txt
├── data/ENB2012_data.csv
├── notebooks/elasticnet_energy.ipynb
├── src/{data_loader.py, train.py}
├── models/{elasticnet_Y1.joblib, elasticnet_Y2.joblib}
└── reports/{vif_table.csv, vif_table_no_X2.csv, grouping_effect_*.csv,
             so_sanh_3_model.{csv,png}, heatmap_alpha_l1ratio_*.png,
             bootstrap_stability_*.csv}
```
