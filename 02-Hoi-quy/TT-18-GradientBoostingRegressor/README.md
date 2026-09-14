# TT-18 — Gradient Boosting Regressor: Thẩm định giá nhà tự động (AVM)

Bài nộp cho **TT-18** (Buổi 13 — Regression Nâng Cao). Mô hình `GradientBoostingRegressor`
huấn luyện trên bộ **Ames Housing** để dự đoán `SalePrice`, kèm khoảng dự báo 10–90%
(hồi quy phân vị) và cơ chế human-in-the-loop cho hệ thống AVM.

## 1. Cách chạy

```bash
pip install -r requirements.txt
```

Tải bộ dữ liệu **Ames Housing / House Prices - Advanced Regression Techniques** từ Kaggle:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
→ giải nén, đặt file `train.csv` vào thư mục `data/` (tạo thư mục này nếu chưa có).

Sau đó chọn 1 trong 2 cách chạy:

**a) Chạy toàn bộ pipeline bằng script (nhanh nhất):**
```bash
cd src
python train.py --data ../data/train.csv --out_dir ..
```
Lệnh trên tạo các biểu đồ trong `reports/`, model `models/gbr_pipeline.joblib` và in ra
Median APE, coverage, bảng human-in-the-loop trên terminal.

**b) Chạy từng bước bằng notebook (để xem giải thích + biểu đồ inline):**
```bash
jupyter notebook notebooks/01_data_cleaning.ipynb   # bước 1–6: làm sạch, encode, kiểm tra skew
jupyter notebook notebooks/02_gbr_model.ipynb       # bước 7–13: baseline, GBR, khoảng giá, APE, HITL
```

## 2. Cấu trúc thư mục

```
TT-18-GradientBoostingRegressor/
├── README.md
├── requirements.txt
├── data/                       ← đặt train.csv của bạn ở đây (không kèm theo trong bản nộp)
├── notebooks/
│   ├── 01_data_cleaning.ipynb  ← missing-value analysis, ordinal encoding, log1p(SalePrice)
│   └── 02_gbr_model.ipynb      ← baseline, GBR + early stopping, quantile regression, APE, HITL
├── src/
│   ├── features.py             ← xử lý missing values, feature engineering, ColumnTransformer
│   └── train.py                ← pipeline huấn luyện end-to-end (dùng được độc lập với notebook)
├── models/
│   ├── gbr_pipeline.joblib      ← preprocessor + GradientBoostingRegressor (điểm dự đoán p50)
│   └── gbr_quantile_models.joblib ← 3 model phân vị (p10 / p50 / p90) cho khoảng giá
└── reports/
    ├── missing_analysis.png
    ├── loss_theo_so_cay.png
    ├── khoang_gia.png
    └── ape_distribution.png
```

## 3. Xử lý dữ liệu

- **Giá trị thiếu có ý nghĩa**: `PoolQC`, `Alley`, `FireplaceQu`, `GarageQual/Cond/Type/Finish`,
  `BsmtQual/Cond/Exposure/FinType1/2`, `MasVnrType`, `Fence`, `MiscFeature` → NaN = *không có
  tiện ích đó*, được điền chuỗi `'None'` (categorical) hoặc `0` (numeric tương ứng như
  `GarageArea`, `TotalBsmtSF`). Không cột nào bị xoá.
- **Giá trị thiếu thật**: `LotFrontage` (và các cột lẻ tẻ còn sót) → điền median/mode.
- **Biến thứ tự**: 12 cột chất lượng (`ExterQual`, `KitchenQual`, `BsmtQual`, `HeatingQC`,
  `FireplaceQu`, `GarageQual`, `GarageCond`, `PoolQC`, `ExterCond`, `BsmtCond`, `BsmtExposure`,
  `GarageFinish`) được mã hoá bằng `OrdinalEncoder` theo đúng thứ tự `Po < Fa < TA < Gd < Ex`
  (không one-hot, để giữ quan hệ thứ bậc).
- **Nhãn lệch phải**: huấn luyện trên `log1p(SalePrice)`, báo cáo giá bằng `expm1()`.
- **Feature engineering**: `TotalSF = 1stFlrSF + 2ndFlrSF + TotalBsmtSF`,
  `TuoiNha = YrSold − YearBuilt`, `DaSuaChua = (YearRemodAdd != YearBuilt)`.

## 4. Mô hình

- `GradientBoostingRegressor(n_estimators=1000, learning_rate=0.03, max_depth=3, subsample=0.8,
  max_features='sqrt', validation_fraction=0.1, n_iter_no_change=50)` — dừng sớm khi loss
  validation ngừng cải thiện 50 vòng liên tiếp.
- Khoảng giá 10–90% từ 3 `GradientBoostingRegressor(loss='quantile', alpha=q)` với
  q ∈ {0.1, 0.5, 0.9}.
- **Human-in-the-loop**: nếu `(p90 − p10) / p50 > 25%` → hồ sơ tự động chuyển cho thẩm định
  viên thay vì duyệt tự động.

## 5. Kết quả (chạy thử để xác minh pipeline)

> ⚠️ Các số dưới đây đến từ một lần chạy thử nghiệm nội bộ để xác minh toàn bộ pipeline chạy
> **end-to-end không lỗi** trước khi bàn giao — **không phải kết quả trên bộ Ames Housing thật**
> (repo không có quyền truy cập mạng tới Kaggle để tải sẵn dữ liệu thật). Sau khi bạn đặt
> `train.csv` thật vào `data/` và chạy lại `python src/train.py`, các số này sẽ được ghi đè
> bằng số liệu thật — cấu trúc bảng/biểu đồ giữ nguyên.

| Chỉ số | Giá trị (chạy thử) | Ngưỡng yêu cầu |
|---|---|---|
| Median APE | 8.4% | < 10–12% |
| RMSE (thang log) | 0.124 | ~0.12–0.13 |
| Coverage khoảng 10–90% | 63.7% | ~80% (mục tiêu) |
| Số cây dùng (early stopping) | 661 / 1000 | — |
| Cải thiện nhờ feature engineering | +0.8% RMSE | — |

**Human-in-the-loop (chạy thử):**

| Nhóm | Tỷ lệ | Số hồ sơ | Median APE |
|---|---|---|---|
| Tự động duyệt (khoảng ≤ 25%) | 42.5% | 124 | 8.4% |
| Chuyển thẩm định viên (khoảng > 25%) | 57.5% | 168 | 8.4% |

**GradientBoostingRegressor vs HistGradientBoostingRegressor** (300 cây/iter, chạy thử):
GBR ≈ 1.1s, HistGBR ≈ 0.2s (HistGBR nhanh hơn đáng kể nhờ binning histogram, phù hợp
production khi cần train lại thường xuyên).

## 6. Mở rộng (chưa triển khai trong bản nộp này)

1. So sánh với XGBoost / LightGBM.
2. Giải thích dự đoán từng căn nhà bằng SHAP.
3. Stacking: GBR + Ridge + RandomForest.
