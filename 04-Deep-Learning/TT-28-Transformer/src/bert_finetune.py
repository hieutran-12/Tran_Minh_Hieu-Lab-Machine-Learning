"""Nhánh B: fine-tune DistilBERT (Hugging Face + PyTorch) — cách làm thực tế."""
import copy
import time

import numpy as np
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

import config as C
from utils import bao_cao_phan_loai, do_ms_mot_tin, luu_ket_qua, set_seed, tinh_metrics

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_DIR = C.MODELS / "distilbert_agnews"


def _batches(tok, x, y, batch, shuffle, max_len=C.BERT_MAX_LEN):
    """Sinh batch đã tokenize (padding động theo batch -> nhanh hơn pad cố định)."""
    idx = np.random.permutation(len(x)) if shuffle else np.arange(len(x))
    for i in range(0, len(idx), batch):
        b = idx[i:i + batch]
        enc = tok([x[j] for j in b], truncation=True, max_length=max_len, padding=True,
                  return_tensors="pt")
        enc["labels"] = torch.as_tensor(y[b])
        yield {k: v.to(DEV) for k, v in enc.items()}


@torch.no_grad()
def predict_logits(model, tok, x, batch=128):
    model.eval()
    order = np.argsort([len(t) for t in x])        # gom câu cùng độ dài -> ít padding
    out = np.zeros((len(x), model.config.num_labels), dtype="float32")
    xs = [x[i] for i in order]
    pos = 0
    for b in _batches(tok, xs, np.zeros(len(xs), dtype=int), batch, shuffle=False):
        b.pop("labels")
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
            lg = model(**b).logits.float().cpu().numpy()
        out[order[pos:pos + len(lg)]] = lg
        pos += len(lg)
    return out


def fine_tune(data, lr=C.BERT_LR, epochs=C.BERT_EPOCHS, batch=C.BERT_BATCH, name="distilbert",
              save=True, bert_name=C.BERT_NAME):
    """Fine-tune toàn bộ model, giữ checkpoint có val accuracy cao nhất."""
    set_seed()
    (x_tr, y_tr), (x_val, y_val), (x_te, y_te) = data["train"], data["val"], data["test"]
    tok = AutoTokenizer.from_pretrained(bert_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        bert_name, num_labels=len(C.CLASSES)).to(DEV)

    steps = epochs * int(np.ceil(len(x_tr) / batch))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.type == "cuda")

    hist, best_acc, best_state = {"loss": [], "val_accuracy": []}, -1.0, None
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        losses = []
        for b in _batches(tok, x_tr, y_tr, batch, shuffle=True):
            with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
                loss = model(**b).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        val_acc = tinh_metrics(y_val, predict_logits(model, tok, x_val))["accuracy"]
        hist["loss"].append(float(np.mean(losses)))
        hist["val_accuracy"].append(val_acc)
        print(f"epoch {ep + 1}/{epochs} — loss {hist['loss'][-1]:.4f} — val_acc {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    train_time = time.perf_counter() - t0
    model.load_state_dict(best_state)

    val_logits = predict_logits(model, tok, x_val)
    test_logits = predict_logits(model, tok, x_te)

    @torch.no_grad()
    def one(text):
        model.eval()
        enc = tok(text, truncation=True, max_length=C.BERT_MAX_LEN, return_tensors="pt").to(DEV)
        out = model(**enc).logits.cpu().numpy()      # .cpu() đồng bộ GPU -> thời gian đo chính xác
        return out
    ms = do_ms_mot_tin(one, x_te)

    info = {"ten": "DistilBERT fine-tune", "train_time_s": train_time,
            "params": int(sum(p.numel() for p in model.parameters())), "predict_ms": ms,
            "device": DEV.type, "epochs": epochs, "lr": lr, "history": hist}
    if save:
        kq = luu_ket_qua(name, info, val_logits, y_val, test_logits, y_te)
        model.save_pretrained(BERT_DIR)
        tok.save_pretrained(BERT_DIR)
    else:
        kq = dict(info, **tinh_metrics(y_te, test_logits))
    print(f"Test acc {kq['accuracy']:.4f} | F1 {kq['f1_macro']:.4f} | train {train_time:.0f}s | "
          f"{kq['params']:,} tham số | {ms:.2f} ms/tin | lr={lr}")
    if save:
        bao_cao_phan_loai(y_te, test_logits)
    return kq


def thi_nghiem_lr(data, n=8000, lrs=(C.BERT_LR, 1e-3)):
    """Minh hoạ cạm bẫy: fine-tune với lr = 1e-3 phá huỷ trọng số đã học sẵn (chạy trên n mẫu, 1 epoch)."""
    sub = {"train": (data["train"][0][:n], data["train"][1][:n]), "val": data["val"],
           "test": data["test"]}
    res = {}
    for lr in lrs:
        print(f"--- lr = {lr} ---")
        res[lr] = fine_tune(sub, lr=lr, epochs=1, name="tmp", save=False)["accuracy"]
    for lr, a in res.items():
        print(f"lr={lr:<8} -> test accuracy {a:.4f}")
    return res
