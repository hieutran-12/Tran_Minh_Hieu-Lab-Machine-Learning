"""Pipeline Keras: vectorize văn bản, huấn luyện BiLSTM + Transformer tự xây, thí nghiệm ablation."""
import json
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers

import config as C
from transformer_block import TransformerClassifier
from utils import bao_cao_phan_loai, do_ms_mot_tin, luu_ket_qua, set_seed, tinh_metrics

for _g in tf.config.list_physical_devices("GPU"):       # không chiếm hết VRAM (còn chỗ cho PyTorch)
    tf.config.experimental.set_memory_growth(_g, True)

TF_DIR = C.MODELS / "transformer_scratch"


def device():
    return "gpu" if tf.config.list_physical_devices("GPU") else "cpu"


# ---------------------------------------------------------------- vectorize
def _make_vec(vocabulary=None):
    return layers.TextVectorization(max_tokens=C.VOCAB_SIZE, output_sequence_length=C.MAX_LEN,
                                    standardize="lower_and_strip_punctuation", split="whitespace",
                                    vocabulary=vocabulary)


def make_vectorizer(train_texts):
    vec = _make_vec()
    vec.adapt(tf.data.Dataset.from_tensor_slices(train_texts).batch(1024))
    return vec


def vectorize(vec, texts, chunk=20000):
    """list văn bản -> mảng id (N, MAX_LEN), padding = 0 ở cuối câu."""
    return np.concatenate([vec(tf.constant(texts[i:i + chunk])).numpy()
                           for i in range(0, len(texts), chunk)])


def luu_vocab(vec, path):
    path.write_text(json.dumps([str(w) for w in vec.get_vocabulary()], ensure_ascii=False))


def nap_vocab(path):
    return _make_vec(json.loads(path.read_text()))


# ---------------------------------------------------------------- model
def build_transformer(n_heads=C.N_HEADS, use_pos=True, vocab_size=C.VOCAB_SIZE):
    m = TransformerClassifier(vocab_size, C.MAX_LEN, C.D_MODEL, n_heads, C.D_FF, C.N_BLOCKS,
                              C.DROPOUT, len(C.CLASSES), use_pos)
    m(np.ones((1, C.MAX_LEN), dtype="int32"))   # build trọng số
    return m


def build_lstm(vocab_size=C.VOCAB_SIZE):
    return keras.Sequential([
        layers.Input(shape=(C.MAX_LEN,), dtype="int32"),
        layers.Embedding(vocab_size, C.D_MODEL, mask_zero=True),
        layers.Bidirectional(layers.LSTM(C.LSTM_UNITS)),
        layers.Dropout(0.3),
        layers.Dense(len(C.CLASSES)),
    ])


# ---------------------------------------------------------------- train
def train(model, ids, epochs=C.EPOCHS):
    """Huấn luyện + early stopping theo val_accuracy. Trả về (history, thời gian train giây)."""
    xtr, ytr, xval, yval = ids
    model.compile(keras.optimizers.Adam(C.LR),
                  keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    cbs = [keras.callbacks.EarlyStopping("val_accuracy", patience=2, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=1, min_lr=1e-5)]
    t0 = time.perf_counter()
    h = model.fit(xtr, ytr, validation_data=(xval, yval), epochs=epochs, batch_size=C.BATCH,
                  callbacks=cbs, verbose=2)
    return h.history, time.perf_counter() - t0


def _predict_one(model, vec):
    def f(text):
        return model(vec(tf.constant([text])), training=False).numpy()
    return f


def danh_gia_va_luu(name, ten, model, vec, data, ids, hist, train_time, save=True):
    """Dự đoán val/test, đo thời gian 1 tin, lưu metrics + logits."""
    xtr, ytr, xval, yval, xte, yte = ids
    val_logits = model.predict(xval, batch_size=512, verbose=0)
    test_logits = model.predict(xte, batch_size=512, verbose=0)
    ms = do_ms_mot_tin(_predict_one(model, vec), data["test"][0])
    info = {"ten": ten, "train_time_s": train_time, "params": int(model.count_params()),
            "predict_ms": ms, "device": device(), "epochs": len(hist["loss"]),
            "history": {k: [float(v) for v in vs] for k, vs in hist.items()}}
    if save:
        return luu_ket_qua(name, info, val_logits, yval, test_logits, yte), test_logits
    return dict(info, **tinh_metrics(yte, test_logits)), test_logits


def ve_hoc_tap(hist, ten, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    ax[0].plot(hist["loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title(f"{ten} — loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist["accuracy"], label="train"); ax[1].plot(hist["val_accuracy"], label="val")
    ax[1].set_title(f"{ten} — accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.show()


def chuan_bi(data, vec=None):
    """Tạo vectorizer (fit trên train) + mảng id cho train/val/test."""
    set_seed()
    vec = vec or make_vectorizer(data["train"][0])
    ids = (vectorize(vec, data["train"][0]), data["train"][1],
           vectorize(vec, data["val"][0]), data["val"][1],
           vectorize(vec, data["test"][0]), data["test"][1])
    return vec, ids


# ---------------------------------------------------------------- các bước chính
def run_lstm(data):
    vec, ids = chuan_bi(data)
    model = build_lstm()
    hist, t = train(model, ids[:4])
    kq, logits = danh_gia_va_luu("lstm", "BiLSTM", model, vec, data, ids, hist, t)
    ve_hoc_tap(hist, "BiLSTM", C.REPORTS / "hoc_tap_lstm.png")
    print(f"Test acc {kq['accuracy']:.4f} | F1 {kq['f1_macro']:.4f} | train {t:.0f}s | "
          f"{kq['params']:,} tham số | {kq['predict_ms']:.2f} ms/tin")
    bao_cao_phan_loai(ids[5], logits)
    return kq


def run_scratch(data):
    vec, ids = chuan_bi(data)
    model = build_transformer()
    model.summary()
    hist, t = train(model, ids[:4])
    kq, logits = danh_gia_va_luu("transformer_scratch", "Transformer tự xây", model, vec, data,
                                 ids, hist, t)
    TF_DIR.mkdir(exist_ok=True)
    model.save_weights(TF_DIR / "model.weights.h5")
    (TF_DIR / "config.json").write_text(json.dumps(model.cfg))
    luu_vocab(vec, TF_DIR / "vocab.json")
    ve_hoc_tap(hist, "Transformer tự xây", C.REPORTS / "hoc_tap_transformer.png")
    print(f"Test acc {kq['accuracy']:.4f} | F1 {kq['f1_macro']:.4f} | train {t:.0f}s | "
          f"{kq['params']:,} tham số | {kq['predict_ms']:.2f} ms/tin")
    bao_cao_phan_loai(ids[5], logits)
    return kq


def load_scratch():
    """Nạp Transformer tự xây đã huấn luyện -> (vectorizer, model)."""
    cfg = json.loads((TF_DIR / "config.json").read_text())
    model = TransformerClassifier(**cfg)
    model(np.ones((1, cfg["max_len"]), dtype="int32"))
    model.load_weights(TF_DIR / "model.weights.h5")
    return nap_vocab(TF_DIR / "vocab.json"), model


def _xao_tron_thu_tu(ids, seed=C.SEED):
    """Đảo ngẫu nhiên thứ tự các từ thật trong mỗi câu (giữ nguyên padding)."""
    rng = np.random.default_rng(seed)
    out = ids.copy()
    for r in out:
        n = int((r != 0).sum())
        r[:n] = r[:n][rng.permutation(n)]
    return out


def run_ablation_pos(data):
    """Bước 5: bỏ positional encoding -> accuracy tụt bao nhiêu? + kiểm tra độ nhạy với thứ tự từ."""
    vec, ids = chuan_bi(data)
    rows = []
    for ten, use_pos in [("Có positional encoding", True), ("BỎ positional encoding", False)]:
        set_seed()
        m = build_transformer(use_pos=use_pos)
        hist, t = train(m, ids[:4])
        kq, _ = danh_gia_va_luu("tmp", ten, m, vec, data, ids, hist, t, save=False)
        pred_shuf = m.predict(_xao_tron_thu_tu(ids[4]), batch_size=512, verbose=0).argmax(1)
        rows.append({"cấu hình": ten, "accuracy": kq["accuracy"], "f1_macro": kq["f1_macro"],
                     "accuracy khi ĐẢO thứ tự từ": float((pred_shuf == ids[5]).mean()),
                     "số epoch": kq["epochs"], "train (s)": round(t, 1)})
    df = pd.DataFrame(rows).set_index("cấu hình")
    df.to_csv(C.REPORTS / "thi_nghiem_positional_encoding.csv")
    d = (df["accuracy"].iloc[0] - df["accuracy"].iloc[1]) * 100
    shuf = df["accuracy khi ĐẢO thứ tự từ"]
    print(df.round(4).to_string())
    print(f"\n=> Bỏ positional encoding: accuracy thay đổi {-d:+.2f} điểm % (có PE: {df['accuracy'].iloc[0]:.2%}, "
          f"không PE: {df['accuracy'].iloc[1]:.2%}).")
    print(f"=> Khi ĐẢO thứ tự từ ở tập test: model có PE còn {shuf.iloc[0]:.2%}, model không PE còn {shuf.iloc[1]:.2%} "
          f"(không PE gần như không bị ảnh hưởng nếu hai số này gần bằng accuracy gốc -> đúng bản chất bag-of-words).")
    print("Lưu ý: bài toán chủ đề phụ thuộc từ khoá nhiều hơn thứ tự từ nên chênh lệch thường nhỏ; "
          "chênh lệch nhỏ/âm trong một lần chạy có thể chỉ là nhiễu ngẫu nhiên (seed).")
    return df


def run_heads(data):
    """Bước 6: khảo sát số head 1/2/4/8 (d_model cố định -> số tham số gần như không đổi)."""
    vec, ids = chuan_bi(data)
    rows = []
    for h in C.HEAD_LIST:
        set_seed()
        m = build_transformer(n_heads=h)
        hist, t = train(m, ids[:4], epochs=C.EPOCHS_SWEEP)
        kq, _ = danh_gia_va_luu("tmp", f"{h} head", m, vec, data, ids, hist, t, save=False)
        rows.append({"số head": h, "accuracy": kq["accuracy"], "f1_macro": kq["f1_macro"],
                     "train (s)": round(t, 1), "thời gian/epoch (s)": round(t / kq["epochs"], 1),
                     "tham số": kq["params"], "dự đoán 1 tin (ms)": round(kq["predict_ms"], 2)})
    df = pd.DataFrame(rows).set_index("số head")
    df.to_csv(C.REPORTS / "khao_sat_so_head.csv")
    print(df.to_string())
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    ax[0].plot(df.index, df["accuracy"], "o-"); ax[0].set_title("Accuracy theo số head")
    ax[1].plot(df.index, df["thời gian/epoch (s)"], "o-", c="C1"); ax[1].set_title("Thời gian / epoch (s)")
    for a in ax:
        a.set_xticks(df.index); a.set_xlabel("số head")
    plt.tight_layout(); plt.savefig(C.REPORTS / "khao_sat_so_head.png", dpi=130); plt.show()
    return df
