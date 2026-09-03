"""
TT-10 — MLP CLASSIFIER (sklearn)
Đọc số viết tay trên séc / phiếu chuyển khoản

File này được XUẤT TỰ ĐỘNG từ notebooks/mlp_digits.ipynb (nguồn chân lý của
pipeline nằm ở notebook đó). Nếu cần sửa logic, sửa trong notebook rồi export
lại (Jupyter: File > Save and Export Notebook As > Executable Script), đừng
sửa trực tiếp ở đây rồi quên đồng bộ lại notebook.

Chạy:
    python src/train.py

Chỉnh DATASET / QUICK / CONFIDENCE_THRESHOLD ngay trong phần CẤU HÌNH bên dưới.
"""

import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib

# ====== CẤU HÌNH — chỉnh ở đây, không cần sửa gì bên dưới ======
DATASET = "mnist"     # "mnist" (chính thức) hoặc "digits" (chạy nhanh để kiểm tra pipeline)
QUICK = False          # True: nếu DATASET="mnist" và máy yếu, chỉ lấy 6000 mẫu để chạy thử nhanh
RANDOM_STATE = 42
CONFIDENCE_THRESHOLD = 0.99   # ngưỡng human-in-the-loop

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
REPORTS_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)

np.random.seed(RANDOM_STATE)
print(f"DATASET={DATASET} | QUICK={QUICK} | ngưỡng human-in-the-loop={CONFIDENCE_THRESHOLD}")

if DATASET == "digits":
    data = load_digits()
    X, y = data.data.astype(float), data.target.astype(int)
    img_shape = (8, 8)
elif DATASET == "mnist":
    print("Đang tải MNIST từ OpenML (lần đầu chậm, có cache cho lần sau)...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X.astype(float)
    y = y.astype(int)
    img_shape = (28, 28)
    if QUICK:
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X), size=6000, replace=False)
        X, y = X[idx], y[idx]
else:
    raise ValueError("DATASET phải là 'digits' hoặc 'mnist'")

n_features = X.shape[1]
n_classes = len(np.unique(y))
print(f"X={X.shape}, y={y.shape}, pixel range=[{X.min()},{X.max()}], n_classes={n_classes}")

fig, axes = plt.subplots(2, 5, figsize=(8, 3.5))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(X[i].reshape(img_shape), cmap="gray")
    ax.set_title(int(y[i]))
    ax.axis("off")
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(len(X_train), "mẫu train,", len(X_test), "mẫu test")

def make_mlp(hidden_layer_sizes=(128, 64), activation="relu",
             learning_rate_init=1e-3, max_iter=100):
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver="adam",
        alpha=1e-4,
        batch_size=128,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=True,       # ⭐ tự tách 10% validation, tránh overfit
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
        verbose=False,
    )

def count_params(n_features, hidden_layer_sizes, n_classes):
    layers = [n_features, *hidden_layer_sizes, n_classes]
    return sum(a * b + b for a, b in zip(layers[:-1], layers[1:]))

results_normalization = {}

# Baseline: Logistic Regression (trên dữ liệu đã chuẩn hoá)
t0 = time.time()
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train / 255.0, y_train)
acc_logreg = accuracy_score(y_test, logreg.predict(X_test / 255.0))
results_normalization["logistic_regression"] = {"accuracy": acc_logreg, "time_s": time.time() - t0}
print(f"[baseline] Logistic Regression accuracy = {acc_logreg:.4f}")

# MLP KHÔNG chuẩn hoá (pixel 0-255 nguyên gốc)
t0 = time.time()
mlp_raw = make_mlp(max_iter=50)
mlp_raw.fit(X_train, y_train)
acc_raw = accuracy_score(y_test, mlp_raw.predict(X_test))
results_normalization["mlp_khong_chuan_hoa"] = {"accuracy": acc_raw, "time_s": time.time() - t0, "n_iter": mlp_raw.n_iter_}
print(f"[KHÔNG chuẩn hoá] accuracy = {acc_raw:.4f} (n_iter={mlp_raw.n_iter_})")

# MLP CÓ chuẩn hoá /255  ← từ đây về sau luôn dùng bản đã chuẩn hoá
X_train_n, X_test_n = X_train / 255.0, X_test / 255.0
t0 = time.time()
mlp_norm = make_mlp()
mlp_norm.fit(X_train_n, y_train)
acc_norm = accuracy_score(y_test, mlp_norm.predict(X_test_n))
results_normalization["mlp_co_chuan_hoa"] = {"accuracy": acc_norm, "time_s": time.time() - t0, "n_iter": mlp_norm.n_iter_}
print(f"[CÓ chuẩn hoá]    accuracy = {acc_norm:.4f} (n_iter={mlp_norm.n_iter_})")

pd.DataFrame(results_normalization).T

ARCHITECTURES = [(64,), (128,), (128, 64), (256, 128, 64)]

arch_rows = []
arch_models = {}
for arch in ARCHITECTURES:
    t0 = time.time()
    mlp = make_mlp(hidden_layer_sizes=arch)
    mlp.fit(X_train_n, y_train)
    acc = accuracy_score(y_test, mlp.predict(X_test_n))
    elapsed = time.time() - t0
    n_params = count_params(n_features, arch, n_classes)
    arch_rows.append({"architecture": str(arch), "n_params": n_params,
                       "accuracy": round(acc, 4), "train_time_s": round(elapsed, 2),
                       "n_iter": mlp.n_iter_})
    arch_models[str(arch)] = mlp
    print(f"[{arch}] acc={acc:.4f}, params={n_params}, time={elapsed:.1f}s, n_iter={mlp.n_iter_}")

df_arch = pd.DataFrame(arch_rows)
df_arch

plt.figure(figsize=(8, 5))
for name, model in arch_models.items():
    plt.plot(model.loss_curve_, label=name)
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Loss curve theo kiến trúc (hidden_layer_sizes)")
plt.legend(); plt.tight_layout()
plt.savefig(REPORTS_DIR / "loss_curves.png")
plt.show()

# Model chính dùng để đánh giá các bước tiếp theo = kiến trúc (128, 64) như đề bài
best_model = arch_models["(128, 64)"]
joblib.dump(best_model, MODELS_DIR / "mlp_pipeline.joblib")
print("Đã lưu model chính ->", MODELS_DIR / "mlp_pipeline.joblib")

act_rows = []
act_curves = {}
for act in ["relu", "tanh", "logistic"]:
    t0 = time.time()
    mlp = make_mlp(activation=act)
    mlp.fit(X_train_n, y_train)
    acc = accuracy_score(y_test, mlp.predict(X_test_n))
    act_rows.append({"activation": act, "accuracy": round(acc, 4),
                      "n_iter": mlp.n_iter_, "train_time_s": round(time.time() - t0, 2)})
    act_curves[act] = mlp.loss_curve_
    print(f"[{act}] acc={acc:.4f}, n_iter={mlp.n_iter_}")

plt.figure(figsize=(8, 5))
for act, curve in act_curves.items():
    plt.plot(curve, label=act)
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Loss curve theo activation")
plt.legend(); plt.tight_layout()
plt.savefig(REPORTS_DIR / "activation_comparison.png")
plt.show()

pd.DataFrame(act_rows)

lr_rows = []
plt.figure(figsize=(8, 5))
for lr in [1e-2, 1e-3, 1e-4]:
    mlp = make_mlp(learning_rate_init=lr)
    mlp.fit(X_train_n, y_train)
    acc = accuracy_score(y_test, mlp.predict(X_test_n))
    lr_rows.append({"learning_rate": lr, "accuracy": round(acc, 4), "n_iter": mlp.n_iter_})
    plt.plot(mlp.loss_curve_, label=f"lr={lr}")
    print(f"[lr={lr}] acc={acc:.4f}, n_iter={mlp.n_iter_}")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Loss curve theo learning_rate_init")
plt.legend(); plt.tight_layout()
plt.savefig(REPORTS_DIR / "learning_rate_curves.png")
plt.show()

pd.DataFrame(lr_rows)

y_pred = best_model.predict(X_test_n)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(cm.shape[0])); ax.set_yticks(range(cm.shape[0]))
ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
ax.set_title("Ma trận nhầm lẫn 10x10")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
fig.colorbar(im); fig.tight_layout()
fig.savefig(REPORTS_DIR / "confusion_10x10.png")
plt.show()

# Cặp số hay nhầm nhất (bỏ đường chéo)
cm_off = cm.copy()
np.fill_diagonal(cm_off, 0)
flat_idx = np.argsort(cm_off, axis=None)[::-1][:5]
top_confusions = []
for idx in flat_idx:
    i, j = np.unravel_index(idx, cm_off.shape)
    if cm_off[i, j] > 0:
        top_confusions.append({"that": int(i), "du_doan": int(j), "so_luong": int(cm_off[i, j])})
print("Top cặp nhầm lẫn:", top_confusions)

wrong_idx = np.where(y_pred != y_test)[0]
n_show = min(20, len(wrong_idx))

if n_show == 0:
    print("Không có ảnh dự đoán sai nào!")
else:
    chosen = np.random.RandomState(RANDOM_STATE).choice(wrong_idx, size=n_show, replace=False)
    cols = 5
    rows_ = int(np.ceil(n_show / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(cols * 2, rows_ * 2))
    axes = np.array(axes).reshape(-1)
    for ax_i, idx in enumerate(chosen):
        axes[ax_i].imshow(X_test[idx].reshape(img_shape), cmap="gray")
        axes[ax_i].set_title(f"thực={y_test[idx]}/đoán={y_pred[idx]}", fontsize=8)
        axes[ax_i].axis("off")
    for ax_i in range(len(chosen), len(axes)):
        axes[ax_i].axis("off")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "anh_sai.png")
    plt.show()

proba = best_model.predict_proba(X_test_n)
confidence = proba.max(axis=1)
y_pred_hitl = proba.argmax(axis=1)

auto_mask = confidence >= CONFIDENCE_THRESHOLD
n_auto, n_manual = auto_mask.sum(), (~auto_mask).sum()
pct_auto = n_auto / len(y_test) * 100
pct_manual = n_manual / len(y_test) * 100
acc_auto = accuracy_score(y_test[auto_mask], y_pred_hitl[auto_mask]) if n_auto > 0 else float("nan")
acc_overall = accuracy_score(y_test, y_pred_hitl)

human_in_the_loop = {
    "threshold": CONFIDENCE_THRESHOLD,
    "pct_tu_dong": round(pct_auto, 2),
    "pct_chuyen_nguoi": round(pct_manual, 2),
    "accuracy_tren_phan_tu_dong": round(acc_auto, 4) if n_auto > 0 else None,
    "accuracy_tong_the_khong_loc": round(acc_overall, 4),
}
print(f"Ngưỡng {CONFIDENCE_THRESHOLD*100:.0f}%: {pct_auto:.1f}% séc tự động xử lý "
      f"(accuracy={acc_auto:.4f}), {pct_manual:.1f}% cần người kiểm tra")
human_in_the_loop

final_accuracy = accuracy_score(y_test, best_model.predict(X_test_n))
print(f"Accuracy cuối cùng — model (128, 64): {final_accuracy:.4f}")

all_results = {
    "dataset": DATASET,
    "n_train": len(X_train), "n_test": len(X_test),
    "baseline_va_chuan_hoa": results_normalization,
    "so_sanh_kien_truc": arch_rows,
    "so_sanh_activation": act_rows,
    "so_sanh_learning_rate": lr_rows,
    "top_cap_nham_lan": top_confusions,
    "human_in_the_loop_99": human_in_the_loop,
    "final_accuracy_128_64": round(final_accuracy, 4),
}
with open(REPORTS_DIR / "results_summary.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print("Đã lưu ->", REPORTS_DIR / "results_summary.json")
