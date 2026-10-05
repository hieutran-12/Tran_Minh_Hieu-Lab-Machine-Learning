"""Chạy toàn bộ pipeline TT-28 từ dòng lệnh.

    python src/train.py all                  # chạy hết, từ baseline đến báo cáo so sánh
    python src/train.py baseline             # TF-IDF + LinearSVC
    python src/train.py lstm                 # BiLSTM
    python src/train.py scratch              # Transformer tự xây
    python src/train.py ablation             # bỏ positional encoding
    python src/train.py heads                # khảo sát 1/2/4/8 head
    python src/train.py bert                 # fine-tune DistilBERT (cần GPU)
    python src/train.py attention            # heatmap attention
    python src/train.py compare              # bảng so sánh, ma trận nhầm lẫn, human-in-the-loop
    python src/train.py all --max-train 5000 # chạy thử nhanh với 5.000 mẫu train
"""
import argparse
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")                      # chạy dòng lệnh: chỉ lưu ảnh, không mở cửa sổ

import config as C                         # noqa: E402
from data import explore, load_ag_news     # noqa: E402

ALL = ["baseline", "lstm", "scratch", "ablation", "heads", "bert", "attention", "compare"]


def chay(stage, max_train, max_test):
    if stage == "compare":                 # chỉ đọc kết quả đã lưu, không cần nạp dữ liệu
        import so_sanh
        so_sanh.run()
        return
    data = load_ag_news(max_train, max_test)
    if stage == "baseline":
        explore(data)
        import baseline
        baseline.run(data)
    elif stage in ("lstm", "scratch", "ablation", "heads"):
        import keras_pipeline as kp
        {"lstm": kp.run_lstm, "scratch": kp.run_scratch,
         "ablation": kp.run_ablation_pos, "heads": kp.run_heads}[stage](data)
    elif stage == "bert":
        import bert_finetune
        bert_finetune.fine_tune(data)
    elif stage == "lr-demo":
        import bert_finetune
        bert_finetune.thi_nghiem_lr(data)
    elif stage == "attention":
        import attention_viz as av
        av.ve_attention_scratch(data)
        av.ve_cac_head(data)
        # TensorFlow và PyTorch trong cùng một tiến trình có thể gây segfault -> DistilBERT chạy tiến trình riêng
        if (C.MODELS / "distilbert_agnews").exists():
            extra = [x for k, v in (("--max-train", max_train), ("--max-test", max_test))
                     if v for x in (k, str(v))]
            subprocess.run([sys.executable, __file__, "attention-bert", *extra], check=True)
        else:
            print("[bỏ qua] chưa có DistilBERT đã fine-tune (chạy bước 'bert' trước).")
    elif stage == "attention-bert":
        import attention_viz as av
        av.ve_attention_bert(data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=ALL + ["lr-demo", "attention-bert", "all"])
    ap.add_argument("--max-train", type=int, default=None, help="giới hạn số mẫu train (chạy thử)")
    ap.add_argument("--max-test", type=int, default=None, help="giới hạn số mẫu test (chạy thử)")
    a = ap.parse_args()
    if a.stage == "all":                   # mỗi bước một tiến trình riêng: giải phóng RAM/VRAM giữa TF và PyTorch
        extra = [x for k, v in (("--max-train", a.max_train), ("--max-test", a.max_test))
                 if v for x in (k, str(v))]
        for s in ALL:
            print(f"\n{'=' * 20} {s.upper()} {'=' * 20}", flush=True)
            subprocess.run([sys.executable, __file__, s, *extra], check=True)
    else:
        chay(a.stage, a.max_train, a.max_test)
