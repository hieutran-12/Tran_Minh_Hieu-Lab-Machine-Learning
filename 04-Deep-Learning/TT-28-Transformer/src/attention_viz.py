"""Bước 10: trực quan hoá attention — từ nào chú ý tới từ nào (heatmap)."""
import matplotlib.pyplot as plt
import numpy as np

import config as C


def chon_mau(data, classes=(0, 2, 3), min_words=8):
    """Chọn mỗi lớp 1 tin NGẮN (dễ đọc heatmap) từ tập test — cố định, không ngẫu nhiên."""
    x, y = data["test"]
    out = []
    for c in classes:
        cand = [t for t, yy in zip(x, y) if yy == c and len(t.split()) >= min_words]
        out.append((min(cand, key=lambda t: (len(t.split()), t)), int(c)))
    return out


def _heatmap(ax, mat, toks, title):
    ax.imshow(mat, cmap="viridis", vmin=0)
    ax.set_xticks(range(len(toks)))
    ax.set_xticklabels(toks, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(range(len(toks)))
    ax.set_yticklabels(toks, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("từ được chú ý (key)", fontsize=8)
    ax.set_ylabel("từ đang hỏi (query)", fontsize=8)


# ------------------------------------------------------------- Transformer tự xây (Keras)
def ve_attention_scratch(data, path=C.REPORTS / "attention_heatmap.png"):
    from keras_pipeline import load_scratch, vectorize
    vec, model = load_scratch()
    vocab = vec.get_vocabulary()
    mau = chon_mau(data)
    fig, axes = plt.subplots(len(mau), C.N_BLOCKS, figsize=(5.2 * C.N_BLOCKS, 5.3 * len(mau)),
                             squeeze=False)
    for r, (text, true) in enumerate(mau):
        ids = vectorize(vec, [text])
        logits, scores = model(ids, return_attention=True)
        pred = int(np.argmax(np.array(logits)[0]))
        n = int((ids[0] != 0).sum())
        toks = [str(vocab[i]) for i in ids[0][:n]]
        for b in range(C.N_BLOCKS):
            att = np.array(scores[b])[0].mean(0)[:n, :n]     # trung bình các head
            _heatmap(axes[r, b], att, toks,
                     f"Khối {b + 1} (TB các head) — thật: {C.CLASSES[true]} · dự đoán: {C.CLASSES[pred]}")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.show()
    return mau


def ve_cac_head(data, block=-1, path=C.REPORTS / "attention_cac_head.png"):
    """Mỗi head là một 'góc nhìn' riêng: vẽ từng head của 1 khối cho tin đầu tiên."""
    from keras_pipeline import load_scratch, vectorize
    vec, model = load_scratch()
    vocab = vec.get_vocabulary()
    text, _ = chon_mau(data)[0]
    ids = vectorize(vec, [text])
    _, scores = model(ids, return_attention=True)
    att = np.array(scores[block])[0]                            # (heads, T, S)
    n = int((ids[0] != 0).sum())
    toks = [str(vocab[i]) for i in ids[0][:n]]
    fig, axes = plt.subplots(1, len(att), figsize=(4.6 * len(att), 4.8), squeeze=False)
    for h, ax in enumerate(axes[0]):
        _heatmap(ax, att[h][:n, :n], toks, f"Head {h + 1}")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.show()


# ------------------------------------------------------------- DistilBERT (PyTorch)
def ve_attention_bert(data, path=C.REPORTS / "attention_heatmap_distilbert.png"):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from bert_finetune import BERT_DIR, DEV
    tok = AutoTokenizer.from_pretrained(BERT_DIR)
    # attn_implementation="eager" để model trả về ma trận attention
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_DIR, attn_implementation="eager").to(DEV).eval()
    mau = chon_mau(data)
    layers_show = [0, model.config.n_layers - 1] if hasattr(model.config, "n_layers") else [0, -1]
    fig, axes = plt.subplots(len(mau), len(layers_show), figsize=(5.4 * len(layers_show), 5.6 * len(mau)),
                             squeeze=False)
    for r, (text, true) in enumerate(mau):
        enc = tok(text, truncation=True, max_length=C.BERT_MAX_LEN, return_tensors="pt").to(DEV)
        with torch.no_grad():
            out = model(**enc, output_attentions=True)
        pred = int(out.logits.argmax(-1))
        toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
        for c, li in enumerate(layers_show):
            att = out.attentions[li][0].mean(0).cpu().numpy()    # trung bình các head
            name = "đầu" if c == 0 else "cuối"
            _heatmap(axes[r, c], att, toks,
                     f"DistilBERT lớp {name} — thật: {C.CLASSES[true]} · dự đoán: {C.CLASSES[pred]}")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.show()
    return mau
