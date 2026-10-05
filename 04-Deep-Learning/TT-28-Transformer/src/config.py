"""Cấu hình chung cho toàn bộ dự án TT-28 (đổi siêu tham số ở đây)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

SEED = 42
CLASSES = ["World", "Sports", "Business", "Sci/Tech"]   # nhãn 0..3 của AG News
VAL_SIZE = 0.1                                          # tách 10% train làm validation

# ---- Keras (LSTM + Transformer tự xây) ----
VOCAB_SIZE = 30000
MAX_LEN = 80            # AG News: tiêu đề + mô tả ~40 từ, 80 là đủ cho >98% mẫu
D_MODEL, N_HEADS, D_FF, N_BLOCKS, DROPOUT = 128, 4, 256, 2, 0.1
LSTM_UNITS = 128
BATCH, EPOCHS, LR = 128, 10, 1e-3
EPOCHS_SWEEP = 5        # số epoch tối đa cho thí nghiệm khảo sát số head
HEAD_LIST = [1, 2, 4, 8]

# ---- Fine-tune DistilBERT (PyTorch) ----
# Đổi sang model khác bằng biến môi trường BERT_NAME (vd. "vinai/phobert-base" cho tiếng Việt)
BERT_NAME = os.getenv("BERT_NAME", "distilbert-base-uncased")
BERT_MAX_LEN = 96       # KHÔNG dùng 512: chậm gấp nhiều lần mà không cải thiện
BERT_LR = 3e-5          # fine-tune: 2e-5 – 5e-5 (RẤT NHỎ)
BERT_EPOCHS = 3
BERT_BATCH = 32

# ---- Human-in-the-loop ----
THRESHOLD = 0.95
NEWS_PER_DAY = 5000
NEWS_PER_EDITOR = 400
