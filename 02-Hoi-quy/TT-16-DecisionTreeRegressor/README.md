# TT-16-DecisionTreeRegressor-Hieu — DECISION TREE REGRESSOR
## Định giá cước chuyến xe kiểu NYC Taxi — bảng giá dạng LUẬT cho tổng đài

## ⚠️ CẬP NHẬT SAU PHẢN HỒI CHẤM ĐIỂM (6.5/10 — lỗi dùng dữ liệu giả)

Bài gốc dùng dữ liệu **mô phỏng** vì môi trường soạn bài (sandbox của trợ lý AI)
bị giới hạn mạng, không gọi được tới `nyc.gov` / CDN `cloudfront.net` nơi TLC
lưu file `yellow_tripdata_*.parquet`. Theo đúng góp ý, đã sửa như sau:

1. **Thêm `src/download_data.py`** — tải **dữ liệu THẬT** trực tiếp từ TLC
   (`https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_<năm-tháng>.parquet`),
   lấy mẫu ngẫu nhiên 200.000 dòng, join thêm `pickup_borough` từ bảng tra vùng
   chính thức (`data/taxi_zone_lookup.csv`, đã kèm sẵn trong repo), rồi ghi ra
   **đúng file/đúng schema** mà `src/train.py` đọc — nghĩa là **không phải sửa
   gì thêm ở `train.py`**, toàn bộ pipeline làm sạch/feature/cây/bảng tra cước
   phía sau giữ nguyên 100%.
2. **`src/generate_data.py` (dữ liệu mô phỏng) chuyển thành phương án dự phòng
   cuối cùng**, không còn là mặc định — chỉ dùng khi hoàn toàn không có internet
   kể cả để tải thủ công.
3. Môi trường sandbox dùng để sửa bài **vẫn không có quyền truy cập mạng tới
   TLC** (đã thử: `web_fetch` gọi được tới CDN nhưng bị chặn ở giới hạn dung
   lượng phản hồi ~30MB, trong khi 1 tháng dữ liệu ~45-50MB; `bash` không được
   phép gọi domain đó), nên **chưa tự chạy lại được `train.py` trên dữ liệu
   thật ở đây** — các con số MAE/MAPE/bảng tra cước ở mục 1 dưới đây **vẫn là
   số liệu trên dữ liệu mô phỏng cũ**.

### → Việc bạn cần làm để có kết quả thật (2 lệnh, chạy trên máy có mạng bình thường)

```bash
pip install -r requirements.txt
python src/download_data.py --year-month 2024-01   # tải + chuẩn hoá dữ liệu THẬT
python src/train.py                                  # chạy lại toàn bộ pipeline, ghi đè report/outputs/model
```

Nếu mạng của bạn cũng chặn `nyc.gov`, tải file `.parquet` bằng trình duyệt rồi:
```bash
python src/download_data.py --parquet-path duong/dan/toi/yellow_tripdata_2024-01.parquet
python src/train.py
```

Sau khi chạy xong, **thay toàn bộ số liệu ở mục 1, 5, 6, 8, 10 bên dưới** bằng
số liệu in ra từ lần chạy thật (script tự ghi lại `outputs/*.csv` và
`reports/*.png` mới), và xoá dòng cảnh báo này khỏi bản nộp cuối.

---

## 1. Tóm tắt kết quả

| | |
|---|---|
| **Baseline DummyRegressor (mean)** | MAE test = **$10.43** |
| **Baseline công thức thủ công (3.0 + 2.8×km)** | MAE test = **$4.24** |
| **Cây không giới hạn độ sâu** | depth=50, 146.058 lá — MAE train=$0.12 vs MAE test=$2.26 (**overfit rõ**) |
| **Cây chọn cho tổng đài** | `max_depth=5`, `min_samples_leaf=500` → **29 lá / 29 mức giá** |
| **MAE / MAPE / RMSE (test, cây depth=5)** | **$1.95** / **15.8%** / **$2.59** |
| **% chuyến đạt sai số ±15%** | **64.9%** — chưa đạt 100%, xem mục 8 |
| **So sánh** | Random Forest chính xác hơn (MAE $1.59) nhưng không tra tay được; Linear Regression thua cây (MAE $1.87) |

---

## 2. Bài toán & vì sao chọn cây nông

Tổng đài cần **tra giá nhanh bằng tay**, giải thích được với khách, sai số chấp
nhận ±15%. Cây sâu/Random Forest chính xác hơn nhưng in ra hàng trăm/hàng nghìn
dòng thì vô dụng với tổng đài viên. → Bài chọn `max_depth=5` (29 lá) làm model
bàn giao, **không phải** cây có MAE test nhỏ nhất tuyệt đối.

## 3. Bốn bước làm sạch (log đầy đủ ở `outputs/log_lam_sach.csv`)

| Bước | Số dòng còn lại | Bị loại |
|---|---|---|
| Ban đầu | 200.000 | — |
| Loại `fare_amount <= 0` | 198.481 | 1.519 |
| Loại `trip_distance <=0` hoặc `>100` dặm | 195.482 | 2.999 |
| Loại `passenger_count == 0` | 193.983 | 1.499 |

`tip_amount`, `tolls_amount`, `total_amount` **không** được đưa vào `FEATURES`
— các cột này chỉ biết được SAU khi chuyến đi kết thúc (rò rỉ), được assert
kiểm tra tự động trong `clean_data()`.

## 4. EDA

`reports/scatter_distance_fare.png`: quan hệ quãng đường–cước gần tuyến tính
nhưng có các "dải" phụ phí rời rạc chồng lên (giờ cao điểm, congestion
Manhattan). `reports/gia_trung_binh_theo_gio.png`: cước trung bình theo giờ có
hai đỉnh rõ, khớp hai khung giờ cao điểm đi làm/tan tầm.

## 5. Chứng minh overfit — cây không giới hạn độ sâu

Cây không giới hạn phát triển tới **depth=50, 146.058 lá** (gần bằng số dòng
train) — MAE train chỉ **$0.12** (gần như học thuộc lòng từng điểm) trong khi
MAE test **$2.26**, cao gấp ~19 lần. Xem `reports/overfit_train_vs_test.png`.

## 6. Quét max_depth 1→20 (`reports/mae_theo_depth.png`)

| max_depth | MAE train | MAE test | số lá |
|---|---|---|---|
| 1 | 5,37 | 5,43 | 2 |
| 3 | 2,56 | 2,58 | 8 |
| 5 | 1,94 | 1,95 | 29 |
| 7 | 1,65 | 1,66 | 85 |
| 11 | 1,618 | **1,631 (nhỏ nhất)** | 226 |
| 20 | 1,618 | 1,631 | 227 |

**Phát hiện quan trọng:** với `min_samples_leaf=500`, MAE test gần như đi ngang
từ depth≈8 trở đi — ràng buộc `min_samples_leaf` đã tự chặn overfit khá hiệu
quả trong khoảng khảo sát, khác hẳn hình dạng "chữ U tăng trở lại" kinh điển.
MAE test nhỏ nhất tuyệt đối rơi vào depth=11 (226 lá) nhưng chỉ tốt hơn depth=5
đúng **$0.02** — không đáng đánh đổi lấy việc không tra tay được nữa. → **Chọn
depth=5 theo yêu cầu nghiệp vụ, không theo MAE nhỏ nhất.**

## 7. Hàm bậc thang (`reports/ham_bac_thang.png`)

Quét `trip_distance` từ 0.1→20 dặm, giữ các biến khác cố định: đường dự đoán là
**14 bậc thang rõ rệt**, không mượt — đúng bản chất cây hồi quy: số mức giá
quan sát được **bằng đúng số lá cây chạm tới** khi quét biến này.

## 8. Cây cuối + bảng tra cước (`outputs/bang_tra_cuoc.csv`)

`max_depth=5`, `min_samples_leaf=500` → 29 lá → 29 mức giá, mỗi dòng là một
luật dạng: `trip_distance <= 9.28 VÀ trip_distance <= 3.70 VÀ ... → $7.15`. Xem
đầy đủ ở `reports/cay_luat_text.txt` và hình cây ở `reports/cay_quyet_dinh.png`.

**Kiểm tra sai số ±15%:** chỉ **64,9%** số chuyến đạt yêu cầu — đây là đánh đổi
thật giữa "tra tay được" (29 lá) và "chính xác ±15%": mục 6 cho thấy tăng độ
sâu lên tới 11 chỉ cải thiện MAE test 0,02$, không đủ để kéo tỉ lệ ±15% lên
đáng kể — nguyên nhân sâu xa là nhiễu ngẫu nhiên trong dữ liệu (mục cước có
thành phần nhiễu độc lập với các đặc trưng đầu vào) mà không cây nào "học" được
thêm nếu không có đặc trưng khác (ví dụ giao thông thời gian thực).

## 9. Vì sao cây KHÔNG ngoại suy được

Cây chỉ trả về **giá trị trung bình của một lá đã học từ dữ liệu train**. Với
chuyến 200 km (vượt xa dải train ~0.1–30 dặm), mẫu vẫn rơi đúng vào lá xa nhất
(`trip_distance > 19.71 dặm...`) và nhận **đúng giá của lá đó** (~$66–70), dù
quãng đường gấp cả chục lần — vì lá là **hằng số**, không phải hàm liên tục
theo x, nên không có cơ chế nào để giá trị dự đoán tăng thêm ngoài dải đã thấy.

## 10. So sánh với Random Forest (TT-17) và Linear Regression

| Model | MAE | MAPE | RMSE |
|---|---|---|---|
| **DecisionTree depth=5 (tra bảng được)** | 1,947 | 15,8% | 2,591 |
| Random Forest (200 cây, depth=10) | **1,591** | **12,6%** | **2,172** |
| Linear Regression | 1,874 | 13,8% | 2,862 |

Random Forest chính xác hơn nhưng **không in được thành bảng tra tay** (200 cây
× hàng chục lá mỗi cây). Linear Regression thua cả cây depth=5 vì các phụ phí
(giờ cao điểm, congestion, sân bay) là **bước nhảy cộng thêm rời rạc**, không
phải quan hệ tuyến tính liên tục với các đặc trưng đầu vào — đúng loại cấu trúc
mà cây quyết định nắm bắt tự nhiên hơn hồi quy tuyến tính.

---

## 11. Cách chạy lại

```bash
pip install -r requirements.txt

# Cách 1 — DỮ LIỆU THẬT (khuyến nghị, xem mục cập nhật ở đầu README):
python src/download_data.py --year-month 2024-01
python src/train.py

# Cách 2 — dữ liệu mô phỏng (chỉ dùng khi không có internet):
python src/generate_data.py
python src/train.py
```

`train.py` không phân biệt hai nguồn trên — cả hai đều ghi ra cùng file
`data/yellow_tripdata_sample_raw.csv` với cùng schema, nên phần pipeline
(làm sạch → feature → cây → bảng tra cước) giữ nguyên bất kể nguồn dữ liệu.

## 12. Cấu trúc thư mục

```
TT-16-DecisionTreeRegressor-Hieu/
├── README.md
├── requirements.txt
├── data/
│   ├── taxi_zone_lookup.csv               ← bảng tra LocationID -> Borough (TLC chính thức)
│   └── yellow_tripdata_sample_raw.csv     ← hiện đang là dữ liệu mô phỏng; chạy download_data.py để thay bằng dữ liệu THẬT (xem đầu README)
├── notebooks/tree_regressor_taxi.ipynb   ← đã chạy sẵn, đủ 12 bước (trên dữ liệu mô phỏng — chạy lại sau khi có dữ liệu thật)
├── src/
│   ├── download_data.py      ← ⭐ tải + chuẩn hoá DỮ LIỆU THẬT từ TLC (mặc định nên dùng)
│   ├── generate_data.py      ← dữ liệu mô phỏng — chỉ dùng dự phòng khi không có mạng
│   ├── train.py               ← script tái tạo toàn bộ pipeline + báo cáo (không đổi theo nguồn dữ liệu)
│   └── build_notebook.py      ← sinh notebook từ code
├── models/tree_reg.joblib    ← cây cuối (depth=5)
├── outputs/
│   ├── bang_tra_cuoc.csv     ← ⭐ sản phẩm bàn giao cho tổng đài (29 mức giá)
│   ├── so_sanh_model.csv
│   └── log_lam_sach.csv
└── reports/
    ├── scatter_distance_fare.png
    ├── gia_trung_binh_theo_gio.png
    ├── overfit_train_vs_test.png
    ├── mae_theo_depth.png
    ├── ham_bac_thang.png
    ├── cay_quyet_dinh.png
    └── cay_luat_text.txt
```

## 13. Mở rộng (chưa làm, gợi ý cho lần sau)

1. Thử `criterion='absolute_error'` — ít nhạy outlier hơn nhưng chậm hơn nhiều.
2. Chứng minh tính KHÔNG ỔN ĐỊNH: train 10 cây với 10 mẫu con khác nhau → so
   sánh bảng tra có giống nhau không (cây quyết định nổi tiếng nhạy với thay
   đổi nhỏ trong dữ liệu train).
3. Dự đoán khoảng giá thay vì 1 con số: `GradientBoostingRegressor(loss='quantile')`
   cho phân vị 10%/90%.
4. ~~Nếu có internet: thay dữ liệu mô phỏng bằng dữ liệu thật~~ → đã làm, xem
   `src/download_data.py` và mục cập nhật ở đầu README. Việc còn lại: chạy nó.
5. Thử gộp 2-3 tháng dữ liệu thật khác nhau (mùa hè vs mùa đông, trước/sau tăng
   giá) để xem bảng tra cước có ổn định theo thời gian không.

## 14. Nguồn dữ liệu

NYC Taxi & Limousine Commission (TLC) Trip Record Data:
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page — tải bằng
`src/download_data.py`. Bảng tra vùng `data/taxi_zone_lookup.csv` cũng lấy từ
nguồn TLC chính thức (Taxi Zone Lookup Table, LocationID ↔ Borough).
