# TT-29 — TF-IDF: Tự động phân loại ticket hỗ trợ khách hàng về đúng bộ phận

Pipeline **TF-IDF + LinearSVC (calibrated)** phân loại văn bản, kèm cơ chế **ngưỡng tin cậy**:
nếu `max(proba) < 0,6` thì chuyển cho người xử lý. Bộ dữ liệu luyện: 20 Newsgroups (18.846 văn bản × 20 chủ đề),
tự tải qua `sklearn.datasets.fetch_20newsgroups` (lần chạy đầu cần internet, sklearn cache lại ở `~/scikit_learn_data`).

## Cách chạy

```bash
pip install -r requirements.txt

python src/train.py                 # chạy đủ 11 bước (~5–10 phút, chậm nhất là bước 5)
python src/train.py --extensions    # thêm mở rộng: LSA + SGDClassifier.partial_fit
jupyter notebook notebooks/tfidf_ticket_classification.ipynb   # bản có diễn giải từng bước

python src/predict.py "My graphics card driver keeps crashing in games"   # demo định tuyến 1 ticket
```

Sau khi chạy, mục **Kết quả** bên dưới được tự điền; toàn bộ bảng/biểu đồ nằm trong `reports/`.

## Cấu trúc

```
├── README.md
├── notebooks/tfidf_ticket_classification.ipynb
├── src/{preprocess.py, train.py, predict.py}
├── models/tfidf_pipeline.joblib      (+ labels.json)
├── reports/{leakage_comparison.png, top_tu_moi_lop.png, confusion_matrix.png, nguong_tin_cay.png, *.csv, bao_cao.md}
└── requirements.txt
```

## Đối chiếu yêu cầu đề bài

| Bước / tiêu chí | Nơi thực hiện | Đầu ra |
|---|---|---|
| 1. Bảng chứng minh rò rỉ (có / không `remove`) | `b1_ro_ri` | `leakage_comparison.png/.csv` |
| 2. EDA | `b2_eda` | `eda.png` |
| 3. Baseline Dummy + Naive Bayes | `b3_baseline` | `baseline.csv` |
| 4. TF-IDF + LinearSVC (calibrated) | `b4_fit_final` | `tfidf_pipeline.joblib` |
| 5. Khảo sát vectorizer (ngram × min_df × sublinear_tf) | `b5_khao_sat_vectorizer` | `khao_sat_vectorizer.csv` |
| 6. So sánh NB · LogReg · LinearSVC | `b6_so_sanh_bo_phan_loai` | `so_sanh_bo_phan_loai.csv` |
| 7. Top 15 từ mỗi lớp + kiểm tra rò rỉ | `b7_top_tu` | `top_tu_moi_lop.png/.csv` |
| 8. Ma trận nhầm lẫn + cặp hay nhầm | `b8_confusion` | `confusion_matrix.png` |
| 9. Bảng ngưỡng tin cậy | `b9_nguong_tin_cay` | `nguong_tin_cay.png/.csv` |
| 10. Thời gian train / dự đoán 1 ticket (< 5 ms) | `b10_thoi_gian` | `thoi_gian.csv` |
| 11. Phân tích 10 ca sai | `b11_ca_sai` | `phan_tich_10_ca_sai.csv` |
| Hạn chế: TF-IDF không hiểu nghĩa | `han_che` | mục cuối của Kết quả |

## Lưu ý thiết kế

- `fetch_20newsgroups(remove=('headers','footers','quotes'))` được dùng cho mọi bước trừ lần so sánh rò rỉ ở bước 1.
- Vectorizer luôn được `fit` **chỉ trên tập train** (nằm trong `Pipeline`) nên không rò rỉ từ vựng/IDF.
- `LinearSVC` không có `predict_proba` → bọc `CalibratedClassifierCV(cv=3, ensemble=False)`; chọn `ensemble=False`
  để dự đoán 1 ticket nhanh hơn (1 bộ SVM thay vì 3).
- Cột "Nguyên nhân" ở bước 11 là **gợi ý tự động** theo luật đơn giản; nên đọc lại trích đoạn để kết luận.
- Dữ liệu tiếng Việt thật: ẩn danh tên/SĐT/email/số tài khoản, rồi dùng `tach_tu_vn` trong `src/preprocess.py`.

## Kết quả

<!-- KET-QUA:BAT-DAU -->
*(Chạy `python src/train.py` để tự điền mục này.)*
<!-- KET-QUA:KET-THUC -->
