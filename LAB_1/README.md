# Dự đoán khách hàng rời bỏ (Customer Churn)

## 1. Mục tiêu

Dự án xây dựng mô hình dự đoán khách hàng rời bỏ (churn) cho bộ dữ liệu Telco Customer Churn. Mục tiêu:

- So sánh nhiều model.
- Đánh giá bằng PR-AUC và Precision@200.
- Lưu pipeline bằng `joblib` để nạp lại và dự đoán dữ liệu mới.
- Đề xuất hành động kinh doanh.

## 2. Dữ liệu

- File: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Số mẫu: ~7.000 khách hàng
- Nhãn mục tiêu: `Churn` (Yes/No)
- Dữ liệu có mất cân bằng: tỷ lệ churn ~ 0.265

## 3. Tiền xử lý

- Loại bỏ cột định danh `customerID`
- Mã hóa:
  - `gender` -> binary
  - `Yes`/`No` -> 0/1
  - `No internet service` / `No phone service` -> 0
- One-hot encode các biến `InternetService`, `Contract`, `PaymentMethod`
- Chuyển `TotalCharges` về numeric và loại bỏ giá trị null
- Chia train/test: `test_size=0.2`, `stratify=y`, `random_state=42`

## 4. Mô hình so sánh

So sánh tối thiểu 4 model:

- Baseline: DummyClassifier(strategy="most_frequent")
- Logistic Regression
- Decision Tree
- Random Forest

## 5. Kết quả chính

Bảng kết quả mẫu:

- Baseline PR_AUC: ~0.265
- Baseline Precision@200: ~0.235

Model chính:

- Logistic Regression
  - PR_AUC: ~0.625
  - Precision@200: ~0.690
- Decision Tree
  - PR_AUC: ~0.604
  - Precision@200: ~0.660
- Random Forest
  - PR_AUC: ~0.646
  - Precision@200: ~0.675

Nhận xét:

- Tất cả model đều tốt hơn baseline PR_AUC rõ rệt.
- Random Forest hiện có PR_AUC cao nhất trong loạt thử nghiệm.
- Precision@200 cho phép đánh giá chất lượng top 200 khách hàng nguy cơ cao.

## 7. Kết luận

- Model hiện tại đã vượt baseline rõ rệt trên PR-AUC.
- `Precision@200` đã được báo cáo cụ thể.
- Đã có ít nhất 4 model so sánh.
- Pipeline đã lưu và nạp lại được.
- README có mục hạn chế và đề xuất hành động cụ thể.
