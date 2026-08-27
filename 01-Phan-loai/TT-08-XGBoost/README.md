
## VÌ SAO ROC-AUC ĐÁNH LỪA

Với dữ liệu lệch cực đoan (gian lận chỉ ~0,17%), số lượng True Negative áp
đảo khiến **False Positive Rate luôn nhỏ** cho dù model sai khá nhiều trên
lớp thiểu số (gian lận) — vì FPR = FP / (FP + TN), và TN quá lớn nên FPR khó
tăng dù FP tăng đáng kể. Kết quả là **ROC-AUC luôn ở mức rất cao (~0,99+)**
gần như bất kể model tốt hay tệ, không phản ánh đúng khả năng phát hiện gian
lận thực tế.

**PR-AUC (Average Precision)** thì chỉ nhìn vào Precision và Recall trên lớp
dương (gian lận), không bị pha loãng bởi số lượng True Negative khổng lồ, nên
phản ánh đúng năng lực thật của model trên đúng bài toán ta quan tâm: trong
số các giao dịch bị gắn cờ, bao nhiêu % thực sự là gian lận (Precision), và
trong số gian lận thật, bắt được bao nhiêu % (Recall).

→ **Kết luận: dùng PR-AUC làm chỉ số chính để đánh giá và so sánh model,
không dùng Accuracy hay ROC-AUC một mình** trên bài toán lệch cực đoan này.
Xem chi tiết + biểu đồ minh hoạ ở mục 8 của notebook (`reports/pr_vs_roc.png`).

## Các lỗi đã phát hiện và sửa so với bản nháp đầu

1. **Rò rỉ dữ liệu khi scale `Amount`** — `StandardScaler` trước đây được
   `fit` trên toàn bộ dữ liệu (gồm cả val/test) trước khi chia tập, khiến
   mean/std bị ảnh hưởng bởi thông tin tương lai. Đã sửa: `fit` chỉ trên
   `train_df`, rồi `transform` cho val/test.
2. **LightGBM early-stop sai chỗ** — dùng `average_precision` để early-stop
   trên tập validation chỉ có 22 giao dịch gian lận khiến chỉ số này dao động
   rất mạnh: tình cờ đạt ~0,17 ngay ở vòng lặp thứ 2 rồi mất hàng trăm vòng
   mới vượt lại được mức đó, khiến early stopping (patience=50) dừng gần như
   ngay từ đầu và cho ra model gần như chưa học gì (ROC-AUC dưới hoặc gần
   mức ngẫu nhiên). Đã sửa: early-stop theo `binary_logloss` (đơn điệu, ổn
   định hơn nhiều trên tập nhỏ), tăng patience và số vòng lặp tối đa.
3. **Lệch đơn vị tiền tệ ở ngưỡng tối ưu chi phí** — cột `Amount` của bộ dữ
   liệu Kaggle gốc là **Euro**, nhưng bị so sánh thẳng với chi phí chặn nhầm
   200.000 **VND** mà không quy đổi, khiến thiệt hại do bỏ lọt gian lận gần
   như bằng 0 so với chi phí chặn nhầm → kết luận "ngưỡng tối ưu" vô nghĩa.
   Đã sửa: quy đổi Euro → VND theo tỷ giá giả định 1 EUR = 27.000 VND (giả
   định được nêu rõ trong code, có thể thay bằng tỷ giá thực tế).
4. **Đường dẫn dữ liệu không portable** — `load_data()` trước đây dùng
   đường dẫn tuyệt đối kiểu Windows, khác với mô tả trong markdown. Đã sửa
   thành đường dẫn tương đối `../data/creditcard.csv` (notebook) /
   `data/creditcard.csv` (script), khớp với mô tả và chạy được trên mọi máy.
5. **Thiếu sản phẩm nộp bài** — bổ sung `README.md` này, `src/train.py`,
   `requirements.txt`.

## Hạn chế

- `V1`–`V28` đã qua PCA nên không giải thích được ý nghĩa nghiệp vụ của từng
  feature quan trọng trong biểu đồ feature importance.
- Tỷ giá EUR→VND dùng để tính ngưỡng tối ưu chi phí là giả định cố định, cần
  thay bằng tỷ giá thực tế hoặc mô hình chi phí chi tiết hơn khi triển khai.
- Dữ liệu mô phỏng khi không có `data/creditcard.csv` chỉ mô phỏng cấu trúc
  và tỉ lệ lệch, không phải hành vi gian lận thật.

