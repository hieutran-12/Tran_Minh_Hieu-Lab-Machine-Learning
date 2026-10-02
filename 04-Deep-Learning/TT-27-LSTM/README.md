# TT-27 — RNN / LSTM / GRU
## Dự báo lưu lượng giao thông theo giờ để điều khiển đèn tín hiệu

Dữ liệu: **Metro Interstate Traffic Volume (UCI)** — 48.204 dòng theo giờ, 2012–2018.
Mục tiêu: dự báo `traffic_volume` (xe/giờ) 1 / 3 / 6 giờ tới, và chứng minh LSTM **có đáng dùng hay không**
bằng cách so với 2 baseline naive + XGBoost.

---

## 1. Cách chạy (end-to-end)

```bash
pip install -r requirements.txt

# Cách 1: chạy toàn bộ bằng 1 lệnh (tự tải dữ liệu từ UCI, huấn luyện, vẽ biểu đồ, điền kết quả vào README)
python -m src.train

# Cách 2: chạy từng bước có giải thích bằng notebook
jupyter notebook notebooks/lstm_traffic.ipynb
```

- Dữ liệu được **tự tải** về `data/raw/` ở lần chạy đầu. Nếu mạng chặn UCI: tải thủ công tại
  https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume, giải nén vào `data/raw/`
  (hoặc `python -m src.train --csv <đường_dẫn_file>`).
- Tuỳ chọn: `--epochs 40` (tối đa, có early stopping), `--seed 42`.
- Thời gian chạy tham khảo trên CPU: khoảng 20–60 phút (8 lần huấn luyện mạng nơ-ron + XGBoost).
- Kết quả sinh ra: `models/lstm_traffic.keras`, `models/scaler.joblib`, các biểu đồ trong `reports/`,
  `reports/ket_qua.md`, và **mục 5 của README này được tự điền** sau khi chạy xong.

## 2. Cấu trúc

```
TT-27-LSTM-<HoTen>/
├── README.md
├── notebooks/lstm_traffic.ipynb     # chạy từng bước, có giải thích
├── src/
│   ├── data.py        # tải dữ liệu, xử lý 4 vấn đề dữ liệu, tạo đặc trưng
│   ├── sequences.py   # chia theo thời gian, scaler, cửa sổ trượt (tao_chuoi)
│   └── train.py       # baseline, XGBoost, RNN/LSTM/GRU, khảo sát, báo cáo
├── models/            # lstm_traffic.keras, scaler.joblib (sinh khi chạy)
├── reports/           # eda_theo_gio.png, rnn_lstm_gru.png, du_bao_vs_thuc_te.png,
│                      # do_dai_cua_so.png, du_bao_nhieu_buoc.png, phan_tich_loi.png, ket_qua.md
└── requirements.txt
```

## 3. Cách xử lý (đối chiếu với đề bài)

**Bốn vấn đề dữ liệu** (`src/data.py::clean`)

| # | Vấn đề | Cách xử lý |
|---|--------|-----------|
| 1 | Lỗ hổng thời gian | Reindex theo giờ đầy đủ. Lỗ hổng ≤ 3 giờ: nội suy theo thời gian + cột cờ `is_filled`. Lỗ hổng > 3 giờ: **cắt thành các đoạn liên tục riêng**, cửa sổ không bao giờ vắt qua lỗ hổng |
| 2 | Ngoại lai phi lý | `temp` < 200 K (vd 0 K) và `rain_1h` > 100 mm → coi là thiếu, nội suy nếu ngắn |
| 3 | Trùng `date_time` | Gộp về 1 dòng / giờ (số: trung bình, `weather_main`: dòng đầu) |
| 4 | `holiday` chỉ ở dòng đầu | Lấy các ngày có nhãn lễ rồi lan ra **toàn bộ 24 giờ** của ngày đó |

**Ba nguyên tắc chống rò rỉ** (`src/sequences.py`)

1. Chia **theo thời gian** 70 / 15 / 15, không shuffle khi chia (chỉ shuffle các cửa sổ lúc `fit`). Mẫu thuộc tập nào theo thời điểm đích.
2. `StandardScaler` **chỉ fit trên train**, sau đó transform val/test.
3. Cửa sổ `tao_chuoi` chỉ chứa quá khứ (`X[i:i+do_dai]` → nhãn `y[i+do_dai+buoc_du_bao-1]`), tạo riêng trong từng đoạn liên tục.

**Đặc trưng**: lưu lượng, `temp`, `log1p(rain_1h)`, `log1p(snow_1h)`, `clouds_all`, sin/cos của giờ và thứ,
cờ ngày lễ, cờ nội suy, one-hot `weather_main`.

**Mô hình**: `LSTM(64, return_sequences=True) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1)` (không activation),
Adam 1e-3, MSE; SimpleRNN và GRU dùng đúng cùng cấu trúc để so sánh công bằng.

**Baseline** (đều được tính trên đúng tập giờ test chung với mô hình): Naive "cùng giờ hôm qua",
Seasonal naive "cùng giờ tuần trước", XGBoost + lag features (lag 0–23 giờ, cùng giờ hôm qua / tuần trước, lịch ngày lễ, giờ, thứ).

**Dự báo nhiều bước**: huấn luyện riêng một mô hình cho từng mức 1h / 3h / 6h (`buoc_du_bao`).

## 4. Phần mở rộng (gợi ý trả lời)

1. **Bidirectional LSTM** — *không hợp lý* khi dự báo tương lai: chiều ngược cần nhìn dữ liệu sau thời điểm hiện tại.
2. **Seq2Seq 24 giờ** và 3. **Transformer (TT-28)**: chưa thực hiện, nằm ngoài phạm vi bắt buộc.

## 5. Kết quả

<!-- KET_QUA_START -->
### Xử lý dữ liệu
- Khoảng thời gian: 2012-10-02 → 2018-09-30
- Số dòng gốc: 48204
- (3) Dòng trùng date_time đã khử: 7629
- (2) temp phi lý (<200K) đã lọc: 10
- (2) rain_1h phi lý (>100mm) đã lọc: 1
- (1) Giờ thiếu trong khoảng thời gian: 11976
- (1)   - nội suy (lỗ hổng <= 3h, có cờ is_filled): 2771
- (1)   - giữ trống, cắt đoạn (lỗ hổng > 3h): 9205
- (1) Số đoạn liên tục: 137
- Số giờ dùng được: 43346
- (4) Dòng holiday gốc → số giờ is_holiday=1 sau khi lan ra cả ngày: 61 → 1272

### So sánh với baseline (dự báo 1 giờ tới, MAE/RMSE đơn vị xe/giờ)
| Mô hình | MAE | RMSE | Thời gian train (s) | Số tham số | Số epoch/cây |
|---|---|---|---|---|---|
| Naive (hôm qua) | 560.3 | 1,022.9 | 0.0 | - | 0 |
| Seasonal naive (tuần trước) | 336.8 | 646.2 | 0.0 | - | 0 |
| XGBoost + lag | 144.0 | 223.0 | 1.9 | - | 371 |
| SimpleRNN | 170.3 | 255.5 | 107.0 | 8,705 | 38 |
| LSTM | 167.2 | 251.4 | 359.8 | 34,721 | 40 |
| GRU | 164.0 | 243.3 | 327.0 | 26,337 | 38 |

![so sánh](reports/rnn_lstm_gru.png)

### Khảo sát độ dài cửa sổ (LSTM)
| Cửa sổ (giờ) | MAE | RMSE | Thời gian train (s) |
|---|---|---|---|
| 6 | 154.4 | 231.6 | 118.4 |
| 12 | 156.7 | 230.4 | 186.8 |
| 24 | 167.2 | 250.9 | 359.8 |
| 48 | 162.4 | 241.2 | 642.1 |

![cửa sổ](reports/do_dai_cua_so.png)

### Dự báo nhiều bước
| Dự báo trước (giờ) | LSTM | XGBoost + lag | Seasonal naive (tuần trước) | Naive (hôm qua) | LSTM tăng so với 1h (%) |
|---|---|---|---|---|---|
| 1 | 167.3 | 144.0 | 337.0 | 559.6 | 0.0 |
| 3 | 221.5 | 185.2 | 337.0 | 559.6 | 32.4 |
| 6 | 229.3 | 205.5 | 337.0 | 559.6 | 37.1 |

![nhiều bước](reports/du_bao_nhieu_buoc.png)

### Dự báo vs thực tế (1 tuần cuối tập test)
![tuần cuối](reports/du_bao_vs_thuc_te.png)

### Phân tích lỗi (LSTM, 1h tới)
- Giờ sai nhiều nhất: 22h (298), 16h (280), 21h (234)
- Thứ sai nhiều nhất: T7 (183), T2 (177)
- Ngày sai nhiều nhất: 2018-04-14 (585), 2018-03-05 (479), 2018-01-22 (467), 2018-05-21 (329), 2018-04-15 (298)

![lỗi](reports/phan_tich_loi.png)

### Kết luận
- LSTM **thắng** seasonal naive: MAE 167 so với 337 xe/giờ (giảm 50.3%).
- XGBoost + lag features cho MAE 144 (LSTM: 167) và train nhanh hơn ~192 lần → nên ưu tiên XGBoost khi triển khai.
- Trong 3 kiến trúc hồi quy, **GRU** có MAE thấp nhất (164); SimpleRNN 170 · LSTM 167 · GRU 164.
- Cửa sổ tốt nhất: **6 giờ** (MAE 154).
- LSTM dự báo 6h tới có MAE tăng 37% so với 1h tới.
- Mức tham chiếu của đề bài cho 1 giờ tới: MAE ~250–400 xe/giờ.
<!-- KET_QUA_END -->
