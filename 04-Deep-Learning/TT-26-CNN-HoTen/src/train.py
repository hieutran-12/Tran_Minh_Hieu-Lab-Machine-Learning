"""Huấn luyện + đánh giá end-to-end: CNN từ đầu vs Transfer Learning vs Fine-tuning.

Chạy:
    python src/train.py                       # tự tải dữ liệu từ Kaggle (kagglehub)
    python src/train.py --data_dir /duong/dan/chest_xray
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from data_pipeline import (
    CLASS_NAMES,
    PROJECT_ROOT,
    SEED,
    compute_class_weights,
    count_table,
    get_data_root,
    list_images,
    load_image,
    make_dataset,
    plot_samples,
    split_train_val,
)
from model import (
    GradCAM,
    build_baseline_cnn,
    build_transfer_model,
    compile_model,
    unfreeze_top_layers,
)

STAGE_BASELINE = "CNN từ đầu (baseline)"
STAGE_FROZEN = "Transfer Learning (đóng băng)"
STAGE_FINETUNE = "Fine-tuning (lr=1e-5)"


# --------------------------------------------------------------------------- #
# Tiện ích
# --------------------------------------------------------------------------- #
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def predict_probs(model: tf.keras.Model, ds) -> np.ndarray:
    return model.predict(ds, verbose=0).ravel()


def compute_metrics(y_true, y_prob, thr: float = 0.5) -> dict:
    """Các chỉ số phân loại nhị phân tại ngưỡng thr (PNEUMONIA = dương tính)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= thr).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "threshold": float(thr),
        "recall": recall,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "precision": precision,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "accuracy": (tp + tn) / len(y_true),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def select_threshold(y_true, y_prob, target_recall: float = 0.97) -> float:
    """Ngưỡng CAO NHẤT mà recall trên tập validation vẫn >= target_recall.

    Ngưỡng cao nhất thoả điều kiện = ít báo động giả nhất trong các ngưỡng đạt recall.
    """
    _, tpr, thresholds = roc_curve(y_true, y_prob)
    ok = np.where(tpr >= target_recall)[0]
    if len(ok) == 0:  # không ngưỡng nào đạt -> lấy ngưỡng có recall cao nhất
        print(f"[CẢNH BÁO] Không có ngưỡng nào đạt recall >= {target_recall} trên validation.")
        return float(thresholds[int(np.argmax(tpr))])
    return float(thresholds[ok[0]])


# --------------------------------------------------------------------------- #
# Huấn luyện
# --------------------------------------------------------------------------- #
def train_stage(model, train_ds, val_ds, class_weight, epochs: int, lr: float, patience: int = 4) -> dict:
    """Compile với lr cho trước và train; tự dừng sớm theo val_auc, khôi phục trọng số tốt nhất."""
    compile_model(model, lr)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=patience, restore_best_weights=True
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=2,
    )
    return history.history


def validation_row(name: str, y_val, val_probs, history: dict) -> dict:
    m = compute_metrics(y_val, val_probs, 0.5)
    return {
        "Mô hình": name,
        "AUC (val)": round(m["auc"], 4),
        "Recall@0.5 (val)": round(m["recall"], 4),
        "Specificity@0.5 (val)": round(m["specificity"], 4),
        "Accuracy@0.5 (val)": round(m["accuracy"], 4),
        "Số epoch chạy": len(history["loss"]),
    }


# --------------------------------------------------------------------------- #
# Biểu đồ
# --------------------------------------------------------------------------- #
def plot_learning_curves(histories: dict[str, dict], out_path: str | Path | None = None):
    """Learning curves (loss / AUC / recall) cho từng giai đoạn huấn luyện."""
    rows = [("loss", "Loss"), ("auc", "AUC"), ("recall", "Recall")]
    fig, axes = plt.subplots(len(rows), len(histories), figsize=(4.6 * len(histories), 8), squeeze=False)
    for c, (name, h) in enumerate(histories.items()):
        for r, (key, label) in enumerate(rows):
            ax = axes[r, c]
            ax.plot(range(1, len(h[key]) + 1), h[key], label="train")
            ax.plot(range(1, len(h[f"val_{key}"]) + 1), h[f"val_{key}"], label="val")
            ax.set_xlabel("epoch")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(name, fontsize=10)
            if r == 0 and c == 0:
                ax.legend()
    fig.suptitle("Learning curves")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    return fig


def plot_confusion_matrix(metrics: dict, out_path: str | Path | None = None):
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=18,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(
        f"Ma trận nhầm lẫn — tập test (ngưỡng {metrics['threshold']:.3f})\n"
        f"SỐ CA VIÊM PHỔI BỊ BỎ SÓT: {metrics['fn']} / {metrics['fn'] + metrics['tp']}",
        fontsize=10,
    )
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    return fig


def _overlay(ax, image, cam=None):
    ax.imshow(image[..., 0], cmap="gray")
    if cam is not None:
        cam_up = tf.image.resize(cam[..., None], image.shape[:2], method="bilinear").numpy()[..., 0]
        ax.imshow(cam_up, cmap="jet", alpha=0.4)
    ax.axis("off")


def pick_gradcam_cases(y_true, y_pred, n: int = 6) -> list[int]:
    """Chọn n ca đa dạng: 2 TP, 2 TN, 1 FN, 1 FP (thiếu thì bù bằng TP/TN)."""
    rng = np.random.default_rng(SEED)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    groups = [
        np.where((y_true == 1) & (y_pred == 1))[0],  # TP
        np.where((y_true == 0) & (y_pred == 0))[0],  # TN
        np.where((y_true == 1) & (y_pred == 0))[0],  # FN
        np.where((y_true == 0) & (y_pred == 1))[0],  # FP
    ]
    chosen: list[int] = []
    for idx, quota in zip(groups, (2, 2, 1, 1)):
        if len(idx):
            chosen += [int(i) for i in rng.choice(idx, size=min(quota, len(idx)), replace=False)]
    leftover = [int(i) for i in np.concatenate(groups[:2]) if int(i) not in chosen]
    rng.shuffle(leftover)
    return (chosen + leftover[: max(0, n - len(chosen))])[:n]


def plot_gradcam_examples(model, df: pd.DataFrame, probs, thr: float, n: int = 6,
                          out_path: str | Path | None = None):
    """Hàng trên: ảnh gốc. Hàng dưới: Grad-CAM (đỏ = vùng làm tăng điểm PNEUMONIA)."""
    y_true = df.label.to_numpy()
    y_pred = (np.asarray(probs) >= thr).astype(int)
    cases = pick_gradcam_cases(y_true, y_pred, n)
    cam_engine = GradCAM(model)
    fig, axes = plt.subplots(2, len(cases), figsize=(3 * len(cases), 6.4), squeeze=False)
    tags = {(1, 1): "TP", (0, 0): "TN", (1, 0): "FN (BỎ SÓT)", (0, 1): "FP"}
    for c, i in enumerate(cases):
        image = load_image(df.path.iloc[i])
        cam = cam_engine.heatmap(image)
        _overlay(axes[0, c], image)
        _overlay(axes[1, c], image, cam)
        axes[0, c].set_title(
            f"{tags[(int(y_true[i]), int(y_pred[i]))]}\nThật: {CLASS_NAMES[y_true[i]]}, p={probs[i]:.2f}",
            fontsize=9,
        )
    fig.suptitle("Grad-CAM: model có nhìn vào vùng phổi không, hay nhìn chữ/nhãn ở góc phim?")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    return fig


def pick_errors(y_true, probs, thr: float, k: int = 10) -> list[int]:
    """k ca sai, ưu tiên ca BỎ SÓT (FN) có xác suất thấp nhất, rồi đến FP có xác suất cao nhất."""
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    y_pred = (probs >= thr).astype(int)
    fn = np.where((y_true == 1) & (y_pred == 0))[0]
    fp = np.where((y_true == 0) & (y_pred == 1))[0]
    fn = fn[np.argsort(probs[fn])]
    fp = fp[np.argsort(-probs[fp])]
    return [int(i) for i in np.concatenate([fn, fp])[:k]]


def plot_errors(df: pd.DataFrame, probs, thr: float, k: int = 10, out_path: str | Path | None = None):
    idx = pick_errors(df.label.to_numpy(), probs, thr, k)
    fig, axes = plt.subplots(2, 5, figsize=(15, 7), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), idx):
        image = load_image(df.path.iloc[i])
        ax.imshow(image[..., 0], cmap="gray")
        pred = int(probs[i] >= thr)
        ax.set_title(
            f"Thật: {CLASS_NAMES[int(df.label.iloc[i])]} | Đoán: {CLASS_NAMES[pred]}\np={probs[i]:.3f}",
            fontsize=9, color="red" if pred == 0 else "darkorange",
        )
    fig.suptitle(f"{len(idx)} ca dự đoán sai (đỏ = bỏ sót viêm phổi, cam = báo động giả)")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    return fig


# --------------------------------------------------------------------------- #
# Báo cáo kết quả
# --------------------------------------------------------------------------- #
def df_to_markdown(df: pd.DataFrame, index_name: str = "") -> str:
    df = df.reset_index().rename(columns={"index": index_name}) if index_name else df
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def build_results_markdown(original_counts, split_counts, comparison, best_name, thr, target_recall,
                           val_at_thr, test_at_thr, test_at_default) -> str:
    fn, tp = test_at_thr["fn"], test_at_thr["tp"]
    lines = [
        "### Dữ liệu",
        df_to_markdown(original_counts, "Tập gốc"),
        "",
        "Sau khi tự tách lại validation từ train (phân tầng + theo bệnh nhân):",
        "",
        df_to_markdown(split_counts, "Tập dùng để huấn luyện"),
        "",
        "### So sánh 3 cách tiếp cận (trên validation)",
        df_to_markdown(comparison),
        "",
        f"Mô hình được chọn (AUC validation cao nhất): **{best_name}** → `models/best_model.keras`.",
        "",
        f"### Ngưỡng quyết định (chọn trên validation, recall ≥ {target_recall})",
        f"- Ngưỡng = **{thr:.4f}** → recall validation = {val_at_thr['recall']:.4f}, "
        f"specificity validation = {val_at_thr['specificity']:.4f}",
        "",
        "### Kết quả trên tập TEST (đánh giá đúng 1 lần)",
        "| | Recall | Specificity | Precision | Accuracy | AUC | Bỏ sót (FN) | Báo động giả (FP) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, m in ((f"Ngưỡng chọn ({thr:.3f})", test_at_thr), ("Ngưỡng mặc định 0.5", test_at_default)):
        lines.append(
            f"| {label} | {m['recall']:.4f} | {m['specificity']:.4f} | {m['precision']:.4f} | "
            f"{m['accuracy']:.4f} | {m['auc']:.4f} | {m['fn']} | {m['fp']} |"
        )
    lines += [
        "",
        f"**Số ca viêm phổi bị bỏ sót trên tập test (ngưỡng chọn): {fn} / {fn + tp}.**",
        "",
        "Hình: `reports/learning_curves.png`, `reports/confusion_matrix.png`, "
        "`reports/gradcam_examples.png`, `reports/ca_du_doan_sai.png`.",
    ]
    return "\n".join(lines)


def update_readme_block(readme_path: Path, content: str) -> bool:
    """Ghi kết quả vào README giữa cặp marker RESULTS:START / RESULTS:END (nếu có)."""
    if not readme_path.exists():
        return False
    text = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(r"(<!-- RESULTS:START -->).*?(<!-- RESULTS:END -->)", re.DOTALL)
    if not pattern.search(text):
        return False
    readme_path.write_text(pattern.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(2)}", text), encoding="utf-8")
    return True


# --------------------------------------------------------------------------- #
# Luồng chính
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TT-26: CNN sàng lọc viêm phổi trên X-quang ngực")
    p.add_argument("--data_dir", default=None, help="Thư mục chest_xray (mặc định: ./data hoặc tự tải từ Kaggle)")
    p.add_argument("--output_dir", default=str(PROJECT_ROOT), help="Nơi ghi models/ và reports/")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs_baseline", type=int, default=20)
    p.add_argument("--epochs_frozen", type=int, default=10)
    p.add_argument("--epochs_finetune", type=int, default=20)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--target_recall", type=float, default=0.97)
    p.add_argument("--no_cache", action="store_true", help="Không cache ảnh trong RAM (chậm hơn, tốn ít RAM hơn)")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    out = Path(args.output_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    set_seed()
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU: {[g.name for g in gpus] if gpus else 'KHÔNG CÓ — sẽ chạy rất chậm trên CPU'}")

    # 1-2. Dữ liệu: đếm, tự tách lại validation
    root = get_data_root(args.data_dir)
    original = {s: list_images(root, s) for s in ("train", "val", "test")}
    original_counts = count_table(original)
    print("\n[1] Số ảnh theo tập gốc:\n", original_counts.to_string())
    train_df, val_df = split_train_val(original["train"], args.val_fraction)
    test_df = original["test"]
    split_counts = count_table({"train (mới)": train_df, "val (mới)": val_df, "test": test_df})
    print("\n[2] Sau khi tách lại validation:\n", split_counts.to_string())

    # 3-4. Ảnh mẫu, pipeline
    plot_samples(original["train"], 8, out / "reports" / "sample_images.png")
    plt.close("all")
    cache = not args.no_cache
    train_ds = make_dataset(train_df, args.batch_size, training=True, cache=cache)
    val_ds = make_dataset(val_df, args.batch_size, cache=cache)
    test_ds = make_dataset(test_df, args.batch_size, cache=cache)
    class_weight = compute_class_weights(train_df)
    print("class_weight:", class_weight)
    y_val = val_df.label.to_numpy()

    histories: dict[str, dict] = {}
    rows = []

    # 5. Baseline
    print("\n[5] CNN từ đầu")
    baseline = build_baseline_cnn()
    histories[STAGE_BASELINE] = train_stage(baseline, train_ds, val_ds, class_weight,
                                            args.epochs_baseline, 1e-3, args.patience)
    rows.append(validation_row(STAGE_BASELINE, y_val, predict_probs(baseline, val_ds), histories[STAGE_BASELINE]))

    # 6. Transfer learning, đóng băng backbone
    print("\n[6] Transfer Learning — giai đoạn 1 (đóng băng)")
    model = build_transfer_model()
    histories[STAGE_FROZEN] = train_stage(model, train_ds, val_ds, class_weight,
                                          args.epochs_frozen, 1e-3, args.patience)
    rows.append(validation_row(STAGE_FROZEN, y_val, predict_probs(model, val_ds), histories[STAGE_FROZEN]))
    frozen_weights = model.get_weights()

    # 7. Fine-tuning
    print("\n[7] Fine-tuning — giai đoạn 2 (lr = 1e-5)")
    unfreeze_top_layers(model, 30)
    histories[STAGE_FINETUNE] = train_stage(model, train_ds, val_ds, class_weight,
                                            args.epochs_finetune, 1e-5, args.patience)
    rows.append(validation_row(STAGE_FINETUNE, y_val, predict_probs(model, val_ds), histories[STAGE_FINETUNE]))

    # 8. Learning curves + so sánh + chọn model theo AUC validation
    plot_learning_curves(histories, out / "reports" / "learning_curves.png")
    plt.close("all")
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / "reports" / "model_comparison.csv", index=False)
    print("\n[8] So sánh trên validation:\n", comparison.to_string(index=False))
    best_name = comparison.loc[comparison["AUC (val)"].idxmax(), "Mô hình"]
    if best_name == STAGE_BASELINE:
        best_model = baseline
    else:
        if best_name == STAGE_FROZEN:
            model.set_weights(frozen_weights)
        best_model = model
    best_path = out / "models" / "best_model.keras"
    best_model.save(best_path)
    best = tf.keras.models.load_model(best_path, compile=False)  # dùng đúng file sẽ giao
    print(f"Model được chọn: {best_name} -> {best_path}")

    # 9. Chọn ngưỡng trên validation
    val_probs = predict_probs(best, val_ds)
    thr = select_threshold(y_val, val_probs, args.target_recall)
    val_at_thr = compute_metrics(y_val, val_probs, thr)
    print(f"\n[9] Ngưỡng = {thr:.4f} (recall val = {val_at_thr['recall']:.4f})")

    # 10. Đánh giá test đúng 1 lần
    test_probs = predict_probs(best, test_ds)
    y_test = test_df.label.to_numpy()
    test_at_thr = compute_metrics(y_test, test_probs, thr)
    test_at_default = compute_metrics(y_test, test_probs, 0.5)
    plot_confusion_matrix(test_at_thr, out / "reports" / "confusion_matrix.png")
    plt.close("all")
    pd.DataFrame({"path": test_df.path, "label": y_test, "prob": test_probs,
                  "pred": (test_probs >= thr).astype(int)}).to_csv(out / "reports" / "test_predictions.csv", index=False)
    print(f"\n[10] TEST @thr={thr:.4f}: recall={test_at_thr['recall']:.4f}, "
          f"accuracy={test_at_thr['accuracy']:.4f}, BỎ SÓT={test_at_thr['fn']}/{test_at_thr['fn'] + test_at_thr['tp']}")

    # 11-12. Grad-CAM và ca sai
    plot_gradcam_examples(best, test_df, test_probs, thr, 6, out / "reports" / "gradcam_examples.png")
    plot_errors(test_df, test_probs, thr, 10, out / "reports" / "ca_du_doan_sai.png")
    plt.close("all")

    results_md = build_results_markdown(original_counts, split_counts, comparison, best_name, thr,
                                        args.target_recall, val_at_thr, test_at_thr, test_at_default)
    (out / "reports" / "results.md").write_text(results_md, encoding="utf-8")
    update_readme_block(out / "README.md", results_md)
    summary = {"model": best_name, "threshold": thr, "validation": val_at_thr,
               "test": test_at_thr, "test_at_0.5": test_at_default}
    (out / "reports" / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nXong. Kết quả: {out / 'reports'}")
    return summary


if __name__ == "__main__":
    plt.switch_backend("Agg")  # chạy script (server không có màn hình): chỉ lưu ảnh, không mở cửa sổ
    main()
