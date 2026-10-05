# TT-28 — Transformer: tự động phân loại chủ đề tin tức cho toà soạn

Phân loại tin vào 4 chuyên mục **World · Sports · Business · Sci/Tech** trên bộ **AG News**
(120.000 train + 7.600 test), so sánh **4 cách**:

| # | Cách | Framework |
|---|------|-----------|
| 1 | TF-IDF + LinearSVC (baseline bắt buộc) | scikit-learn |
| 2 | BiLSTM (kiến trúc tuần tự, mốc TT-27) | Keras |
| 3 | **Transformer Encoder tự xây** (positional encoding + multi-head attention + FFN/residual/LayerNorm) | Keras |
| 4 | **Fine-tune DistilBERT** | PyTorch + Hugging Face |

Mọi thứ chạy end-to-end bằng một lệnh; dữ liệu AG News **tự tải** ở lần chạy đầu (cần Internet).

## 1. Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Nhánh Transformer tự xây và DistilBERT **cần GPU** (Google Colab T4 miễn phí là đủ). TF-IDF + SVC chạy CPU trong vài chục giây.

## 2. Chạy

**Cách 1 — dòng lệnh (khuyến nghị, một lệnh):**

```bash
python src/train.py all                    # toàn bộ: baseline → LSTM → Transformer → ablation → số head → DistilBERT → attention → báo cáo
python src/train.py all --max-train 5000   # chạy thử nhanh (5.000 mẫu train) để kiểm tra môi trường
```

Chạy từng bước: `baseline` · `lstm` · `scratch` · `ablation` · `heads` · `bert` · `attention` · `compare` · `lr-demo` (xem `python src/train.py -h`).

**Cách 2 — notebook** (chạy lần lượt, mỗi notebook có giải thích từng bước):

| Notebook | Nội dung | Bước README gốc |
|---|---|---|
| `notebooks/01_baseline_tfidf.ipynb` | khám phá dữ liệu, TF-IDF + LinearSVC | 1, 2 |
| `notebooks/02_transformer_tu_xay.ipynb` | BiLSTM, Transformer tự xây, bỏ positional encoding, khảo sát số head, heatmap attention | 3, 4, 5, 6, 10 |
| `notebooks/03_finetune_bert.ipynb` | fine-tune DistilBERT, thí nghiệm lr, attention DistilBERT | 7, 10 |
| `notebooks/04_tong_hop_so_sanh.ipynb` | bảng so sánh 4 cách, ma trận nhầm lẫn, human-in-the-loop, kết luận | 8, 9, 11 |

Hyper-parameter nằm trong `src/config.py`. Đổi sang model khác (ví dụ tiếng Việt) bằng biến môi trường: `BERT_NAME=vinai/phobert-base`.

## 3. Cấu trúc thư mục

```
TT-28-Transformer-HoTen/
├── README.md                 ← file này (mục "Kết quả" tự cập nhật sau khi chạy)
├── requirements.txt
├── notebooks/                ← 01_baseline_tfidf · 02_transformer_tu_xay · 03_finetune_bert · 04_tong_hop_so_sanh
├── src/
│   ├── config.py             ← siêu tham số, đường dẫn
│   ├── data.py               ← nạp AG News, chia train/val/test, khám phá
│   ├── baseline.py           ← TF-IDF + LinearSVC
│   ├── transformer_block.py  ← PositionalEncoding, KhoiTransformer, TransformerClassifier
│   ├── keras_pipeline.py     ← vectorize, BiLSTM, huấn luyện Transformer, ablation, khảo sát head
│   ├── bert_finetune.py      ← fine-tune DistilBERT
│   ├── attention_viz.py      ← heatmap attention
│   ├── so_sanh.py            ← bảng so sánh, confusion matrix, human-in-the-loop, kết luận
│   ├── utils.py
│   └── train.py              ← CLI chạy toàn bộ pipeline
├── models/                   ← model đã huấn luyện (tạo khi chạy)
└── reports/                  ← ảnh + bảng kết quả (tạo khi chạy)
```

Ảnh chính trong `reports/`: `so_sanh_4_cach.png`, `attention_heatmap.png`, `confusion_4x4.png`
(cùng `attention_cac_head.png`, `attention_heatmap_distilbert.png`, `confusion_all_models.png`, `hitl_curve.png`, `khao_sat_so_head.png`, đường học của từng model).
Bảng: `bang_so_sanh.csv`, `bang_human_in_the_loop.csv`, `cap_nham_lan.csv`, `thi_nghiem_positional_encoding.csv`, `khao_sat_so_head.csv`, `bao_cao_ket_qua.md`.

## 4. Thiết kế & quyết định kỹ thuật

- **Chia dữ liệu:** 120.000 train → 90% train / 10% validation (phân tầng, seed 42). Cả 4 cách huấn luyện trên cùng tập train; validation dùng để chọn checkpoint/early stopping và hiệu chỉnh xác suất; **test chỉ dùng để báo cáo**.
- **Làm sạch:** giải mã HTML entity (`&amp;`, `&lt;`…) và bỏ ký tự `\` thừa của AG News; áp dụng giống nhau cho cả 4 cách.
- **Transformer tự xây:** `Embedding → (+ Positional Encoding sin/cos) → 2 × KhoiTransformer → masked GlobalAvgPool → Dense`. Padding được che trong attention và trong pooling. `key_dim = d_model // n_heads` (đúng bài báo gốc) nên khi khảo sát số head, tổng tham số **không đổi** → so sánh công bằng. *(Snippet gợi ý trong đề bài dùng `key_dim = d_model`; cách đó làm số tham số tăng theo số head nên không phù hợp để khảo sát.)*
- **Thí nghiệm bỏ positional encoding:** hai model giống hệt, khác đúng một chi tiết; thêm phép thử *đảo ngẫu nhiên thứ tự từ* ở test để chứng minh model không có positional encoding chỉ là bag-of-words.
- **Fine-tune DistilBERT:** `lr = 3e-5`, 3 epoch, warmup 6% + giảm tuyến tính, `max_length = 96` với padding động, AdamW (weight decay 0.01), mixed precision khi có GPU, giữ checkpoint tốt nhất theo val accuracy. `lr-demo` minh hoạ lr = 1e-3 phá huỷ trọng số.
- **Đo thời gian:** *train* = tổng thời gian huấn luyện (gồm vectorize/fit TF-IDF với SVC); *dự đoán 1 tin* = trung bình 100 lần gọi pipeline đầy đủ (tokenize + model) cho từng tin, batch = 1. Số tham số của SVC = số hệ số của mô hình tuyến tính.
- **Human-in-the-loop:** xác suất được hiệu chỉnh bằng *temperature scaling* trên validation (cùng cách cho cả 4 model) vì softmax thô thường quá tự tin còn LinearSVC không có xác suất. Tin có độ tin cậy ≥ 95% được tự động gán chuyên mục, còn lại chuyển biên tập viên; giả định 5.000 tin/ngày, mỗi biên tập viên xử lý 400 tin/ngày (chỉnh ở `config.py`).

## 5. Kết quả

<!-- KET_QUA_START -->
_Chưa có kết quả. Chạy `python src/train.py all` (hoặc notebook 01 → 04); mục này sẽ được tự động điền bảng so sánh 4 cách, cặp chuyên mục hay nhầm, bảng human-in-the-loop và kết luận chi phí/lợi ích từ số đo thật của lần chạy._
<!-- KET_QUA_END -->

**Mức tham chiếu (từ đề bài, không phải kết quả đo):** TF-IDF+SVC ~91–92% · Transformer tự xây ~88–90% · DistilBERT ~94–95%.
Transformer tự xây thua TF-IDF với 120k mẫu là chuyện bình thường: kiến trúc hiện đại cần rất nhiều dữ liệu mới phát huy.

## 6. Cạm bẫy đã tránh

| Cạm bẫy | Cách xử lý trong code |
|---|---|
| Quên positional encoding | Có `PositionalEncoding`, kèm thí nghiệm bỏ nó đi |
| Fine-tune với lr = 1e-3 | `lr = 3e-5`; có `lr-demo` để thấy hậu quả |
| Không có baseline TF-IDF | Baseline là bước đầu tiên, là mốc so sánh |
| `max_length = 512` | `max_length = 96`, padding động |
| Fine-tune 10 epoch | 3 epoch + chọn checkpoint tốt nhất theo validation |
| Tiếng Việt không tách từ | Chỉ áp dụng khi mở rộng sang tiếng Việt (xem dưới) |

## 7. Mở rộng (chưa làm trong bản này)

1. Tiếng Việt với PhoBERT (`BERT_NAME=vinai/phobert-base`) — cần tách từ bằng `underthesea` trước khi tokenize.
2. Chưng cất tri thức: dùng DistilBERT dạy lại model nhỏ hơn.
3. Phân loại đa nhãn: một tin thuộc nhiều chuyên mục.