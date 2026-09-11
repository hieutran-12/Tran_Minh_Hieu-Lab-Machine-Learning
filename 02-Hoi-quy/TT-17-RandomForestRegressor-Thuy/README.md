# TT-17 — RANDOM FOREST REGRESSOR (v2 — đã tối ưu, end-to-end)
## Dự đoán giá vé máy bay để tư vấn khách "nên mua bây giờ hay chờ"

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · Buổi 13 |
| 🧠 **Nhóm** | Hồi quy · Ensemble (Bagging) |
| 🔧 **Thuật toán** | Random Forest Regressor (có tinh chỉnh siêu tham số) |
| 🏭 **Lĩnh vực** | Du lịch · Hàng không · OTA |
| 📦 **Bộ dữ liệu** | Flight Price Prediction (Kaggle) — `Clean_Dataset.csv` |

---

## 0. Bản v2 khác gì bản v1?

Bản đầu (v1) đã bao phủ đủ 12 mục trong checklist nhưng còn 6 lỗ hổng thực chất.
Bản v2 (trong repo này) vá toàn bộ:

| # | Lỗ hổng ở bản v1 | Đã sửa trong v2 |
|---|---|---|
| 1 | Random Forest dùng **tham số mặc định**, chưa từng tinh chỉnh | Thêm `tune_random_forest()` — `RandomizedSearchCV` (k-fold CV) tìm `n_estimators`, `max_depth`, `min_samples_leaf/split`, `max_features` |
| 2 | Chỉ đánh giá bằng **1 lần** train/test split — dễ "may rủi" | Thêm `run_cross_validation()` — 5-fold CV cho model cuối, báo cáo mean ± std |
| 3 | Ngoại suy chỉ thử trên **1 dòng dữ liệu đầu tiên** (`X_train.iloc[[0]]`) — không đại diện | `run_extrapolation_experiment()` giờ chạy trên **30 hồ sơ khách ngẫu nhiên**, vẽ dải ± std |
| 4 | Claim "Gini importance thiên vị biến nhiều mức" chỉ **nói suông**, không có số liệu chứng minh | Thêm biểu đồ `so_sanh_permutation_vs_gini.png` — so sánh trực tiếp 2 phương pháp trên cùng dữ liệu |
| 5 | Không có gì đảm bảo dữ liệu đầu vào sạch (null, trùng lặp, giá âm) trước khi train | Thêm `_validate_data_quality()` — chạy tự động trong `load_data()`, raise lỗi nếu có giá trị vô lý |
| 6 | Không có **bài test nào** — README chỉ nói "đã chạy thử" bằng lời | Thêm `tests/` với **21 test pytest** (đơn vị + end-to-end qua CLI), chạy trên dữ liệu giả lập, không cần `Clean_Dataset.csv` thật |
| 7 (bonus) | Split train/test không phân tầng, "class" (biến mạnh nhất) có thể lệch tỉ lệ giữa 2 tập | `train_test_split(..., stratify=X["class"])` |
| 8 (bonus) | Không kiểm tra rò rỉ dữ liệu một cách định lượng | Thêm **shuffled-target sanity check**: train lại với target xáo trộn ngẫu nhiên, R² phải rơi về ~0 |
| 9 (bonus) | Không lưu metadata để tái lập kết quả | `summary.json` giờ có mục `metadata` (phiên bản thư viện, tham số CLI, thời gian chạy) |

---

## 1. Cách chạy dự án

```bash
# 1. Cài thư viện (đã thêm pytest cho phần kiểm thử)
pip install -r requirements.txt

# 2. Đặt file dữ liệu gốc vào đúng vị trí
cp /duong/dan/toi/Clean_Dataset.csv data/Clean_Dataset.csv

# 3a. Chạy pipeline ĐẦY ĐỦ (khuyến nghị) — có tinh chỉnh siêu tham số + CV
python src/train.py --data data/Clean_Dataset.csv

# 3b. Chạy NHANH để debug (ít cây hơn, ít cấu hình tinh chỉnh hơn, ~vài phút)
python src/train.py --data data/Clean_Dataset.csv --fast

# 3c. Bỏ qua bước tinh chỉnh siêu tham số (dùng RF mặc định, nhanh nhất)
python src/train.py --data data/Clean_Dataset.csv --no-tune

# 3d. HOẶC mở notebook để xem từng bước có giải thích
jupyter notebook notebooks/rf_regressor_flight.ipynb

# 4. Chạy bộ kiểm thử (không cần Clean_Dataset.csv — dùng dữ liệu giả lập)
pytest tests/ -v
```

Các cờ CLI hữu ích khác: `--tune-iter` (số cấu hình thử, mặc định 20),
`--tune-sample-frac` (tỉ lệ dữ liệu dùng để tìm siêu tham số, mặc định 0.35),
`--cv-folds` (số fold cross-validation, mặc định 5).

Script và notebook dùng **chung một module** (`src/train.py`) nên luôn cho kết quả
đồng nhất. Sau khi chạy xong, `reports/` sẽ có đủ **9 hình** + `summary.json`
(RMSE, R², oob_score_, kết quả tinh chỉnh siêu tham số, cross-validation,
permutation importance, so sánh Gini vs permutation, kết luận PDP, độ phủ
khoảng dự báo, kết quả ngoại suy, sanity check rò rỉ dữ liệu, metadata).

---

## 2. Những quyết định kỹ thuật quan trọng (và lý do)

| Quyết định | Lý do |
|---|---|
| **Bỏ cột `flight`** | Mã chuyến bay có hàng nghìn giá trị duy nhất (VD `SG-8709`). One-hot encode sẽ tạo hàng nghìn cột và cây sẽ "học thuộc lòng" từng mã chuyến thay vì học quy luật giá → overfit nặng. |
| **Kiểm tra chất lượng dữ liệu trước khi train** | Không "tin mù" file CSV đầu vào: kiểm tra null, dòng trùng lặp, giá vé/duration ≤ 0, days_left < 0 — raise lỗi cứng nếu có giá trị vô lý thay vì âm thầm train trên dữ liệu lỗi. |
| **Split có phân tầng (`stratify=class`)** | `class` là biến chi phối mạnh nhất tới giá — phân tầng đảm bảo tỉ lệ Economy/Business giống nhau ở train và test, tránh test set bị lệch ngẫu nhiên. |
| **`max_features=1.0`** (không phải `'sqrt'`) | Điểm khác biệt quan trọng giữa RF hồi quy và RF phân loại trong scikit-learn: bản hồi quy mặc định dùng **hết** đặc trưng tại mỗi lần chia, còn phân loại mặc định `'sqrt'`. (Tuy nhiên bước tinh chỉnh siêu tham số vẫn thử cả `'sqrt'`, `0.5`, `0.7` — để dữ liệu tự quyết định, không áp đặt.) |
| **Không scale biến số** | Cây quyết định / rừng chia dựa trên ngưỡng, không dựa trên khoảng cách → scale không ảnh hưởng kết quả. |
| **`RandomizedSearchCV` trên tập con** | Tìm kiếm đầy đủ (`GridSearchCV`) trên toàn bộ ~300k dòng quá tốn thời gian cho một bài tập. Search trên tập con (mặc định 35%) rồi refit model cuối trên **toàn bộ** train — cân bằng giữa tốc độ và chất lượng. |
| **Cross-validation 5-fold cho model cuối** | 1 train/test split có thể "may rủi". CV cho biết RMSE dao động bao nhiêu giữa các lần chia khác nhau (mean ± std), đáng tin cậy hơn để báo cáo. |
| **Permutation importance + so sánh với Gini importance** | `feature_importances_` mặc định (Gini/MSE) thiên vị biến phân loại có nhiều mức giá trị. Biểu đồ so sánh trực tiếp chứng minh điều này bằng số liệu thay vì chỉ nói suông. |
| **Khoảng dự báo 10–90% từ dự đoán từng cây** | Nghiệp vụ OTA cần hiển thị "giá dự kiến 3,2–3,9 triệu", không phải một con số cứng. |
| **Ngoại suy trên 30 hồ sơ đại diện** | Thử trên 1 dòng dữ liệu ngẫu nhiên dễ cho kết luận sai nếu dòng đó không đại diện. Lấy trung bình trên nhiều hồ sơ cho kết luận đáng tin cậy hơn về mặt thống kê. |
| **Shuffled-target sanity check** | Cách kiểm tra rò rỉ dữ liệu định lượng: train lại trên target đã xáo trộn ngẫu nhiên — nếu R² vẫn cao, chắc chắn có cột nào đó "nhìn thấy" target thật. |

---

## 3. Cấu trúc dự án

```
TT-17-RandomForestRegressor-Thuy/
├── README.md                              ← file này
├── requirements.txt
├── notebooks/
│   └── rf_regressor_flight.ipynb          ← từng bước có giải thích, tái sử dụng src/train.py
├── src/
│   └── train.py                           ← script sản xuất, chạy độc lập bằng CLI
├── tests/
│   ├── conftest.py                        ← sinh dữ liệu giả lập cùng schema Clean_Dataset.csv
│   └── test_train.py                      ← 21 test: đơn vị + end-to-end qua CLI
├── data/
│   └── (đặt Clean_Dataset.csv vào đây — không commit dữ liệu gốc)
├── models/
│   └── rf_reg.joblib                      ← sinh ra sau khi chạy train.py
└── reports/
    ├── gia_theo_days_left.png
    ├── boxplot_gia_theo_class.png
    ├── boxplot_gia_theo_airline.png
    ├── rmse_theo_so_cay.png
    ├── permutation_importance.png
    ├── so_sanh_permutation_vs_gini.png    ← MỚI (v2)
    ├── pdp_days_left.png
    ├── khoang_du_bao.png
    ├── ngoai_suy_days_left.png
    └── summary.json                       ← toàn bộ số liệu, bao gồm tinh chỉnh + CV + sanity check
```

---

## 4. Đã xác thực pipeline chạy đúng — bằng test, không chỉ bằng lời

Thay vì chỉ khẳng định "đã chạy thử end-to-end" trong README (như bản v1),
bản v2 có **21 test tự động** trong `tests/test_train.py`, chạy trên dữ liệu
giả lập đúng schema Kaggle (sinh ra trong `tests/conftest.py`, có gắn sẵn hiệu
ứng phi tuyến theo `days_left` và chênh lệch giá theo `class` để test được cả
tính đúng về mặt *nội dung*, không chỉ *chạy không lỗi*). Ví dụ:

- `test_permutation_importance_class_is_top_feature` — vì dữ liệu giả lập được
  thiết kế để `class` ảnh hưởng mạnh nhất, permutation importance PHẢI xếp
  `class` đầu bảng, nếu không nghĩa là logic tính importance có lỗi.
- `test_extrapolation_experiment_flags_capping` — xác nhận Random Forest thực
  sự bị "kẹp trần" khi ngoại suy, đúng lý thuyết.
- `test_shuffled_target_sanity_check_detects_no_leakage` — xác nhận cơ chế
  phát hiện rò rỉ hoạt động đúng trên dữ liệu sạch.
- `test_full_cli_pipeline_runs_without_error` — chạy toàn bộ `python src/train.py`
  qua `subprocess` (chế độ `--fast`), kiểm tra `models/rf_reg.joblib` và
  `reports/summary.json` được tạo ra với đầy đủ tất cả các mục.

Chạy `pytest tests/ -v` để tự kiểm chứng — không cần `Clean_Dataset.csv` thật.
Bộ test này **đã được chạy và pass 21/21** trước khi bàn giao (xem log chạy
thử với dữ liệu giả lập 6.000 dòng: RF đạt Test R²≈0.98, đánh bại toàn bộ
baseline, sanity check xác nhận không rò rỉ).

---

## 5. Tiêu chí hoàn thành — đối chiếu với checklist gốc

| # | Tiêu chí trong checklist | Trạng thái |
|---|---|---|
| 1 | Nạp dữ liệu, bỏ cột `flight` + index thừa | ✅ + kiểm tra chất lượng dữ liệu |
| 2 | EDA: giá theo `days_left` (đường) | ✅ `reports/gia_theo_days_left.png` |
| 3 | EDA: boxplot theo `class`, `airline` | ✅ 2 file `boxplot_*.png` |
| 4 | Pipeline OneHotEncoder, không scale | ✅ `build_preprocessor()` |
| 5 | Baseline: Dummy + Linear + 1 cây đơn | ✅ `run_baselines()` |
| 6 | Random Forest + `oob_score_` | ✅ + tinh chỉnh siêu tham số (RandomizedSearchCV) |
| 7 | RMSE theo n_estimators = 10..500 | ✅ `reports/rmse_theo_so_cay.png` + tự tìm điểm bão hoà |
| 8 | Permutation importance, không dùng `feature_importances_` mặc định | ✅ + so sánh trực tiếp 2 phương pháp bằng biểu đồ |
| 9 | ⭐ PDP cho `days_left` + định lượng tiền tiết kiệm | ✅ `reports/pdp_days_left.png` |
| 10 | ⭐ Khoảng dự báo 10–90% + đo độ phủ | ✅ `reports/khoang_du_bao.png` |
| 11 | ⚠️ Ngoại suy `days_left=100` | ✅ trên 30 hồ sơ đại diện, không phải 1 dòng |
| 12 | So sánh với XGBoost | ✅ có early stopping |
| — | *(bổ sung)* Tinh chỉnh siêu tham số | ✅ `RandomizedSearchCV` |
| — | *(bổ sung)* Cross-validation cho model cuối | ✅ 5-fold |
| — | *(bổ sung)* Kiểm tra rò rỉ dữ liệu định lượng | ✅ shuffled-target test |
| — | *(bổ sung)* Bộ kiểm thử tự động | ✅ 21 test pytest |

---

## 6. Hạn chế cần lưu ý khi bàn giao cho khách hàng

```
1. KHÔNG NGOẠI SUY ĐƯỢC — giá vé nằm ngoài khoảng days_left đã thấy trong tập
   train sẽ bị "kẹp trần" (đã kiểm chứng bằng test, không chỉ lý thuyết).

2. Dữ liệu là thị trường Ấn Độ (2022) — KHÔNG áp dụng trực tiếp cho thị trường
   Việt Nam. Cần thu thập lại dữ liệu nội địa trước khi đưa vào sản xuất.

3. `class` là biến chi phối mạnh nhất — nên cân nhắc huấn luyện 2 model riêng
   cho 2 hạng vé nếu muốn độ chính xác cao hơn (xem mục "Mở rộng").

4. Tinh chỉnh siêu tham số hiện chạy trên tập CON của train (mặc định 35%) để
   tiết kiệm thời gian — nếu có nhiều thời gian/tài nguyên hơn, tăng
   `--tune-sample-frac` lên 1.0 và `--tune-iter` lên 50+ để tìm kỹ hơn.
```

---

## 7. Hướng mở rộng (nếu có thời gian)

1. **Tách 2 model riêng** cho Economy và Business — so sánh tổng RMSE với 1 model chung.
2. **Log-transform giá vé** trước khi train (giá vé thường lệch phải) — có thể cải thiện RMSE ở đuôi phân phối, cần so sánh cẩn thận vì đổi thang đo ảnh hưởng cách diễn giải RMSE.
3. **`ExtraTreesRegressor`** — chia ngẫu nhiên hoàn toàn thay vì tìm ngưỡng tối ưu, thường nhanh hơn.
4. **Quy tắc tư vấn thực tế**: "nếu giá dự báo tuần sau cao hơn hiện tại > 10% → khuyên mua ngay", sau đó backtest trên dữ liệu lịch sử.
5. **`GridSearchCV` đầy đủ** thay vì `RandomizedSearchCV` nếu có đủ tài nguyên tính toán.
