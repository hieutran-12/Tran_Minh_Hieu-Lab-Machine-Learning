# TT-07 — Gradient Boosting: Dự đoán thu nhập (Adult Income Dataset)

## 1. Mục tiêu

Xây dựng và đánh giá mô hình **Gradient Boosting** để dự đoán một cá nhân có
thu nhập `>50K` hay `<=50K`/năm, dựa trên bộ dữ liệu Adult Census Income (UCI).
So sánh với Decision Tree, Random Forest, AdaBoost, HistGradientBoosting; đồng
thời kiểm tra thiên lệch (bias) của mô hình theo `sex` và `race`.

## 2. Cấu trúc thư mục

```
TT-07-GradientBoosting-<HoTen>/
├── README.md
├── notebooks/gradient_boosting_income.ipynb   # khám phá dữ liệu, thử nghiệm, giải thích
├── src/train.py                               # script tái tạo pipeline, xuất model + report
├── models/gb_pipeline.joblib                  # pipeline (preprocessor + model) đã huấn luyện
├── reports/
│   ├── loss_theo_so_cay.png                   # train/val loss theo số cây
│   ├── lr_vs_nestimators.png                  # heatmap F1-CV theo learning_rate x n_estimators
│   └── bias_by_group.png                      # tỉ lệ dự đoán >50K theo sex/race
└── requirements.txt
```

## 3. Cách chạy

```bash
pip install -r requirements.txt
python src/train.py --data-path ./adult.data
```

Kết quả: `models/gb_pipeline.joblib` và 3 file `.png` trong `reports/` được tạo/ghi đè.

## 4. BAGGING vs BOOSTING

**Vì sao Boosting dùng cây nông còn Random Forest (Bagging) dùng cây sâu?**

- **Random Forest (Bagging):** mỗi cây được huấn luyện độc lập trên một mẫu
  bootstrap, sau đó lấy trung bình/biểu quyết để **giảm phương sai
  (variance)**. Muốn giảm phương sai hiệu quả, từng cây cần có độ lệch (bias)
  thấp → cây phải **sâu** (gần như fit hết dữ liệu của nó), vì việc lấy trung
  bình nhiều cây sâu, tương quan thấp sẽ tự triệt tiêu phần overfit riêng lẻ.
- **Gradient Boosting:** các cây được huấn luyện **tuần tự**, mỗi cây sau chỉ
  học phần sai số (residual/gradient) còn lại của các cây trước → mục tiêu là
  **giảm độ lệch (bias) từ từ, có kiểm soát**. Nếu mỗi cây đã sâu (độ lệch
  thấp), tổng nhiều cây sâu cộng dồn sẽ overfit rất nhanh. Vì vậy Boosting
  dùng cây **nông** (thường `max_depth` 2-4, thậm chí stump), kết hợp với
  `learning_rate` nhỏ để mô hình học chậm và ổn định.

So sánh thực nghiệm (Random Forest vs Gradient Boosting vs AdaBoost vs
HistGradientBoosting) — bao gồm PR-AUC và thời gian huấn luyện — nằm ở mục 7
và 8 của notebook. `HistGradientBoostingClassifier` gom giá trị đặc trưng vào
bin (histogram-based) nên tìm điểm chia nhanh hơn nhiều so với
`GradientBoostingClassifier`, cho thời gian train ngắn hơn đáng kể với PR-AUC
tương đương.

## 5. THIÊN LỆCH (Bias)

Đo **tỉ lệ dự đoán thu nhập >50K**, **Recall** và **Precision** theo từng
nhóm `sex` và `race` trên tập test (xem `reports/bias_by_group.png` và mục 9
của notebook).

Kết quả chính:

| sex    | Số lượng | Tỉ lệ dự đoán >50K | Recall | Precision |
| ------ | -------- | ------------------ | ------ | --------- |
| Male   | 4065     | 0.246              | 0.589  | 0.745     |
| Female | 1968     | 0.080              | 0.477  | 0.713     |

| race               | Số lượng | Tỉ lệ dự đoán >50K | Recall | Precision |
| ------------------ | -------- | ------------------ | ------ | --------- |
| Asian-Pac-Islander | 186      | 0.226              | 0.636  | 0.667     |
| White              | 5186     | 0.205              | 0.579  | 0.746     |
| Other              | 38       | 0.105              | 0.667  | 0.500     |
| Amer-Indian-Eskimo | 64       | 0.094              | 0.333  | 0.333     |
| Black              | 559      | 0.081              | 0.425  | 0.756     |

**Nhận định:** Chênh lệch lớn về "tỉ lệ dự đoán >50K" giữa các nhóm không tự
động nghĩa là mô hình "bất công" — dữ liệu census 1994 vốn phản ánh chênh
lệch thu nhập thực tế theo giới tính/chủng tộc trong xã hội thời điểm đó. Tuy
nhiên, nếu **Recall** (khả năng phát hiện đúng người thu nhập cao) chênh lệch
mạnh giữa các nhóm — như giữa `Amer-Indian-Eskimo` (0.333) và `Asian-Pac-Islander`
(0.636) — đó là dấu hiệu mô hình học kém hơn với một số nhóm nhất định và cần
lưu ý khi triển khai thực tế.

## 6. Dữ liệu

Adult Census Income Dataset (UCI Machine Learning Repository), 32.561 dòng,
15 cột gốc (14 đặc trưng + nhãn `income`).
