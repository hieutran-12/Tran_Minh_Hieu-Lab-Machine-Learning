"""
train.py
--------
Pipeline end-to-end cho TT-25 — MLP với Keras: chấm điểm khách hàng tiềm năng
mua bảo hiểm ô tô.

Chạy toàn bộ pipeline:
    python -m src.train --data data/train.csv

Yêu cầu: đặt file train.csv (tải từ Kaggle) vào thư mục data/ trước khi chạy.
Xem README.md mục 3 (BỘ DỮ LIỆU) để biết nguồn tải.

Pipeline thực hiện đầy đủ các bước trong README.md mục 5:
  1. EDA nhanh theo Previously_Insured / Vehicle_Damage / nhóm tuổi
  2. Tiền xử lý (log1p, one-hot, scaling)
  3. Chia train/val/test có stratify
  4. Baseline LightGBM
  5-6. MLP cơ bản (có/không Dropout+BatchNorm)
  7. Learning curves
  8. So sánh 3 kiến trúc
  9. class_weight vs không dùng
  10. Embedding vs one-hot cho biến nhiều mức
  11. Ngưỡng theo Precision@3000
  12. Bảng kết luận MLP vs LightGBM
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.data import (
    TARGET, load_raw, split_data, build_dense_features,
    build_dense_features_with_highcard_onehot, build_embedding_features,
)
from src.model import build_mlp, build_mlp_embedding, default_callbacks
from src.evaluate import (
    pr_auc_score, precision_at_k, threshold_for_top_k,
    plot_learning_curves, plot_pr_curve, plot_bar_comparison,
)

REPORTS_DIR = "reports"
MODELS_DIR = "models"
TOP_K = 3000
EPOCHS = 60
BATCH_SIZE = 1024
RANDOM_STATE = 42


def step_eda(df: pd.DataFrame) -> dict:
    """Mục 1: tỉ lệ quan tâm (Response=1) theo vài biến then chốt."""
    df = df.copy()
    df["age_group"] = pd.cut(df["Age"], bins=[0, 25, 35, 45, 55, 65, 120],
                              labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"])
    eda = {
        "overall_response_rate": float(df[TARGET].mean()),
        "by_previously_insured": df.groupby("Previously_Insured")[TARGET].mean().round(4).to_dict(),
        "by_vehicle_damage": df.groupby("Vehicle_Damage")[TARGET].mean().round(4).to_dict(),
        "by_age_group": df.groupby("age_group", observed=True)[TARGET].mean().round(4).to_dict(),
    }
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "eda_summary.json"), "w", encoding="utf-8") as f:
        json.dump(eda, f, ensure_ascii=False, indent=2)
    print("[EDA] Tỉ lệ quan tâm tổng thể:", eda["overall_response_rate"])
    print("[EDA] Theo Previously_Insured:", eda["by_previously_insured"])
    print("[EDA] Theo Vehicle_Damage:", eda["by_vehicle_damage"])
    return eda


def step_baseline_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    """Mục 4: baseline cây bắt buộc, dùng để đối chứng trung thực với MLP."""
    t0 = time.time()
    class_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    clf = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=class_ratio,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    train_time = time.time() - t0
    proba_test = clf.predict_proba(X_test)[:, 1]
    result = {
        "model": "LightGBM (baseline)",
        "pr_auc": pr_auc_score(y_test, proba_test),
        f"precision_at_{TOP_K}": precision_at_k(y_test, proba_test, TOP_K),
        "train_seconds": round(train_time, 1),
        "n_params_or_trees": clf.best_iteration_ if clf.best_iteration_ else clf.n_estimators,
    }
    plot_pr_curve(y_test, proba_test, os.path.join(REPORTS_DIR, "pr_curve_lightgbm.png"), label="LightGBM")
    return result, clf


def _fit_mlp(model, X_train, y_train, X_val, y_val, class_weight=None, checkpoint_name="best.keras",
             epochs=EPOCHS):
    callbacks = default_callbacks(os.path.join(MODELS_DIR, checkpoint_name))
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0,
    )
    train_time = time.time() - t0
    return history, train_time


def step_mlp_plain_vs_bn_dropout(X_train, y_train, X_val, y_val, X_test, y_test) -> list:
    """Mục 5-6-7: MLP cơ bản (không Dropout/BN) so với có Dropout/BN, kèm learning curves."""
    n_features = X_train.shape[1]
    results = []

    model_plain = build_mlp(n_features, hidden_units=(128, 64), use_dropout_bn=False)
    hist_plain, t_plain = _fit_mlp(model_plain, X_train, y_train, X_val, y_val,
                                    checkpoint_name="mlp_plain.keras")
    plot_learning_curves(hist_plain, os.path.join(REPORTS_DIR, "learning_curves_plain.png"),
                          title="MLP cơ bản (không Dropout/BatchNorm)")
    proba_plain = model_plain.predict(X_test, verbose=0).ravel()
    results.append({
        "model": "MLP (không Dropout/BN)",
        "pr_auc": pr_auc_score(y_test, proba_plain),
        f"precision_at_{TOP_K}": precision_at_k(y_test, proba_plain, TOP_K),
        "train_seconds": round(t_plain, 1),
        "n_params_or_trees": model_plain.count_params(),
    })

    model_bn = build_mlp(n_features, hidden_units=(128, 64), use_dropout_bn=True)
    hist_bn, t_bn = _fit_mlp(model_bn, X_train, y_train, X_val, y_val,
                              checkpoint_name="mlp_bn_dropout.keras")
    plot_learning_curves(hist_bn, os.path.join(REPORTS_DIR, "learning_curves_bn_dropout.png"),
                          title="MLP + Dropout + BatchNorm")
    proba_bn = model_bn.predict(X_test, verbose=0).ravel()
    results.append({
        "model": "MLP (+ Dropout/BN)",
        "pr_auc": pr_auc_score(y_test, proba_bn),
        f"precision_at_{TOP_K}": precision_at_k(y_test, proba_bn, TOP_K),
        "train_seconds": round(t_bn, 1),
        "n_params_or_trees": model_bn.count_params(),
    })
    plot_pr_curve(y_test, proba_bn, os.path.join(REPORTS_DIR, "pr_curve_mlp_bn_dropout.png"), label="MLP+BN+Dropout")
    return results, model_bn


def step_architecture_comparison(X_train, y_train, X_val, y_val, X_test, y_test) -> list:
    """Mục 8: thử 3 kiến trúc (64) · (128,64) · (256,128,64)."""
    n_features = X_train.shape[1]
    archs = {"(64)": (64,), "(128,64)": (128, 64), "(256,128,64)": (256, 128, 64)}
    results = []
    for name, units in archs.items():
        model = build_mlp(n_features, hidden_units=units, use_dropout_bn=True)
        hist, t_train = _fit_mlp(model, X_train, y_train, X_val, y_val,
                                  checkpoint_name=f"mlp_arch_{name.replace(',', '_')}.keras")
        proba = model.predict(X_test, verbose=0).ravel()
        results.append({
            "model": f"MLP {name}",
            "pr_auc": pr_auc_score(y_test, proba),
            f"precision_at_{TOP_K}": precision_at_k(y_test, proba, TOP_K),
            "train_seconds": round(t_train, 1),
            "n_params_or_trees": model.count_params(),
        })
    plot_bar_comparison(
        [r["model"] for r in results], [r["pr_auc"] for r in results],
        os.path.join(REPORTS_DIR, "kien_truc_comparison.png"),
        ylabel="PR-AUC", title="So sánh 3 kiến trúc MLP",
    )
    return results


def step_class_weight_comparison(X_train, y_train, X_val, y_val, X_test, y_test) -> list:
    """Mục 9: class_weight vs không dùng, trên kiến trúc (128,64) + Dropout/BN."""
    n_features = X_train.shape[1]
    results = []

    model_no_cw = build_mlp(n_features, hidden_units=(128, 64), use_dropout_bn=True)
    _, t_no_cw = _fit_mlp(model_no_cw, X_train, y_train, X_val, y_val,
                           class_weight=None, checkpoint_name="mlp_no_class_weight.keras")
    proba_no_cw = model_no_cw.predict(X_test, verbose=0).ravel()
    results.append({
        "model": "MLP (không class_weight)",
        "pr_auc": pr_auc_score(y_test, proba_no_cw),
        f"precision_at_{TOP_K}": precision_at_k(y_test, proba_no_cw, TOP_K),
        "train_seconds": round(t_no_cw, 1),
    })

    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {0: weights[0], 1: weights[1]}
    model_cw = build_mlp(n_features, hidden_units=(128, 64), use_dropout_bn=True)
    _, t_cw = _fit_mlp(model_cw, X_train, y_train, X_val, y_val,
                        class_weight=class_weight, checkpoint_name="mlp_class_weight.keras")
    proba_cw = model_cw.predict(X_test, verbose=0).ravel()
    results.append({
        "model": "MLP (class_weight='balanced')",
        "pr_auc": pr_auc_score(y_test, proba_cw),
        f"precision_at_{TOP_K}": precision_at_k(y_test, proba_cw, TOP_K),
        "train_seconds": round(t_cw, 1),
    })
    return results


def step_embedding_comparison(df_train, df_val, df_test) -> list:
    """Mục 10: Embedding cho Region_Code/Policy_Sales_Channel vs one-hot toàn bộ."""
    results = []

    # (a) One-hot toàn bộ, kể cả biến nhiều mức
    (Xoh_train, y_train), (Xoh_val, y_val), (Xoh_test, y_test) = \
        build_dense_features_with_highcard_onehot(df_train, df_val, df_test)
    model_oh = build_mlp(Xoh_train.shape[1], hidden_units=(128, 64), use_dropout_bn=True)
    _, t_oh = _fit_mlp(model_oh, Xoh_train, y_train, Xoh_val, y_val,
                        checkpoint_name="mlp_onehot_highcard.keras")
    proba_oh = model_oh.predict(Xoh_test, verbose=0).ravel()
    results.append({
        "model": "MLP one-hot toàn bộ (kể cả biến nhiều mức)",
        "pr_auc": pr_auc_score(y_test, proba_oh),
        f"precision_at_{TOP_K}": precision_at_k(y_test, proba_oh, TOP_K),
        "n_params_or_trees": model_oh.count_params(),
        "n_input_features": int(Xoh_train.shape[1]),
    })

    # (b) Embedding cho Region_Code / Policy_Sales_Channel
    (Xemb_train, y_train2), (Xemb_val, y_val2), (Xemb_test, y_test2), meta = \
        build_embedding_features(df_train, df_val, df_test)
    model_emb = build_mlp_embedding(meta["dense_dim"], meta["vocab_sizes"], hidden_units=(128, 64))
    _, t_emb = _fit_mlp(model_emb, Xemb_train, y_train2, Xemb_val, y_val2,
                         checkpoint_name="mlp_embedding.keras")
    proba_emb = model_emb.predict(Xemb_test, verbose=0).ravel()
    total_input_dims = meta["dense_dim"] + sum(meta["vocab_sizes"].values())
    results.append({
        "model": "MLP + Embedding (Region_Code, Policy_Sales_Channel)",
        "pr_auc": pr_auc_score(y_test2, proba_emb),
        f"precision_at_{TOP_K}": precision_at_k(y_test2, proba_emb, TOP_K),
        "n_params_or_trees": model_emb.count_params(),
        "n_input_features": f"{meta['dense_dim']} dense + {len(meta['vocab_sizes'])} embedding inputs",
    })
    plot_bar_comparison(
        [r["model"] for r in results], [r["pr_auc"] for r in results],
        os.path.join(REPORTS_DIR, "embedding_vs_onehot.png"),
        ylabel="PR-AUC", title="Embedding vs one-hot cho biến nhiều mức",
    )
    return results


def write_conclusion_table(all_results: dict):
    """Mục 11-12: chọn ngưỡng theo Precision@3000 + bảng kết luận MLP vs LightGBM."""
    lines = ["# Bảng kết luận — TT-25 MLP Keras vs LightGBM\n"]
    lines.append(f"Ngưỡng chấm theo Precision@{TOP_K} khách hàng xếp hạng cao nhất "
                  f"(đội telesales gọi được {TOP_K} cuộc/ngày).\n")

    lines.append("## Baseline cây vs MLP tốt nhất\n")
    lines.append("| Model | PR-AUC | Precision@3000 | Thời gian train (s) | Tham số/cây |")
    lines.append("|---|---|---|---|---|")
    for r in all_results["baseline_vs_best_mlp"]:
        lines.append(f"| {r['model']} | {r['pr_auc']:.4f} | {r[f'precision_at_{TOP_K}']:.4f} | "
                      f"{r.get('train_seconds', '-')} | {r.get('n_params_or_trees', '-')} |")

    lines.append("\n## MLP cơ bản vs + Dropout/BatchNorm\n")
    lines.append("| Model | PR-AUC | Precision@3000 | Thời gian train (s) | Tham số |")
    lines.append("|---|---|---|---|---|")
    for r in all_results["plain_vs_bn"]:
        lines.append(f"| {r['model']} | {r['pr_auc']:.4f} | {r[f'precision_at_{TOP_K}']:.4f} | "
                      f"{r['train_seconds']} | {r['n_params_or_trees']} |")

    lines.append("\n## So sánh 3 kiến trúc\n")
    lines.append("| Kiến trúc | PR-AUC | Precision@3000 | Thời gian train (s) | Tham số |")
    lines.append("|---|---|---|---|---|")
    for r in all_results["architectures"]:
        lines.append(f"| {r['model']} | {r['pr_auc']:.4f} | {r[f'precision_at_{TOP_K}']:.4f} | "
                      f"{r['train_seconds']} | {r['n_params_or_trees']} |")

    lines.append("\n## class_weight vs không dùng\n")
    lines.append("| Model | PR-AUC | Precision@3000 | Thời gian train (s) |")
    lines.append("|---|---|---|---|")
    for r in all_results["class_weight"]:
        lines.append(f"| {r['model']} | {r['pr_auc']:.4f} | {r[f'precision_at_{TOP_K}']:.4f} | {r['train_seconds']} |")

    lines.append("\n## Embedding vs one-hot cho biến nhiều mức\n")
    lines.append("| Model | PR-AUC | Precision@3000 | Tham số | Số chiều đầu vào |")
    lines.append("|---|---|---|---|---|")
    for r in all_results["embedding"]:
        lines.append(f"| {r['model']} | {r['pr_auc']:.4f} | {r[f'precision_at_{TOP_K}']:.4f} | "
                      f"{r['n_params_or_trees']} | {r['n_input_features']} |")

    lines.append("\n## Kết luận\n")
    best_mlp = max(all_results["plain_vs_bn"] + all_results["architectures"]
                    + all_results["class_weight"] + all_results["embedding"],
                    key=lambda r: r["pr_auc"])
    lgbm = [r for r in all_results["baseline_vs_best_mlp"] if r["model"].startswith("LightGBM")][0]
    if lgbm["pr_auc"] >= best_mlp["pr_auc"]:
        verdict = (f"LightGBM (PR-AUC={lgbm['pr_auc']:.4f}) **ngang hoặc nhỉnh hơn** MLP tốt nhất "
                   f"({best_mlp['model']}, PR-AUC={best_mlp['pr_auc']:.4f}). Đúng với nhận định "
                   "chung: trên dữ liệu bảng dạng này, cây gradient boosting thường mạnh hơn hoặc "
                   "ngang MLP, trong khi MLP tốn nhiều công sức tinh chỉnh (kiến trúc, Dropout, "
                   "learning rate, embedding) hơn hẳn. Với bài toán này, LightGBM là lựa chọn "
                   "thực dụng hơn cho production; MLP phù hợp hơn khi cần kết hợp dữ liệu dạng "
                   "khác (văn bản, chuỗi thời gian) hoặc làm base model cho stacking.")
    else:
        verdict = (f"MLP tốt nhất ({best_mlp['model']}, PR-AUC={best_mlp['pr_auc']:.4f}) nhỉnh hơn "
                   f"LightGBM (PR-AUC={lgbm['pr_auc']:.4f}) trên tập test này. Cần xác nhận lại "
                   "bằng cross-validation trước khi kết luận chắc chắn, vì kết quả trên dữ liệu "
                   "bảng thường nghiêng về cây gradient boosting; chênh lệch nhỏ có thể do phương "
                   "sai giữa các lần chạy.")
    lines.append(verdict + "\n")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, "comparison_summary.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[REPORT] Đã ghi bảng kết luận vào {out_path}")
    return out_path


def main(data_path: str):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"[1/8] Đọc dữ liệu từ {data_path} ...")
    df = load_raw(data_path)
    step_eda(df)

    print("[2/8] Chia train/val/test (stratify) ...")
    df_train, df_val, df_test = split_data(df)

    print("[3/8] Tiền xử lý (one-hot + log1p + scaling) cho baseline & MLP dense ...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), _ = build_dense_features(df_train, df_val, df_test)

    print("[4/8] Baseline LightGBM ...")
    lgbm_result, _ = step_baseline_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test)

    print("[5/8] MLP cơ bản vs + Dropout/BatchNorm ...")
    plain_vs_bn_results, best_bn_model = step_mlp_plain_vs_bn_dropout(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    print("[6/8] So sánh 3 kiến trúc MLP ...")
    arch_results = step_architecture_comparison(X_train, y_train, X_val, y_val, X_test, y_test)

    print("[7/8] class_weight vs không dùng ...")
    cw_results = step_class_weight_comparison(X_train, y_train, X_val, y_val, X_test, y_test)

    print("[8/8] Embedding vs one-hot cho biến nhiều mức ...")
    emb_results = step_embedding_comparison(df_train, df_val, df_test)

    # Ngưỡng theo Precision@3000 trên model MLP (+Dropout/BN) tốt nhất
    proba_best = best_bn_model.predict(X_test, verbose=0).ravel()
    threshold = threshold_for_top_k(proba_best, TOP_K)
    with open(os.path.join(REPORTS_DIR, "threshold.json"), "w", encoding="utf-8") as f:
        json.dump({"top_k": TOP_K, "probability_threshold": threshold}, f, indent=2)
    print(f"[THRESHOLD] Ngưỡng xác suất cho top {TOP_K} khách hàng: {threshold:.4f}")

    all_results = {
        "baseline_vs_best_mlp": [
            lgbm_result,
            max(plain_vs_bn_results + arch_results + cw_results + emb_results, key=lambda r: r["pr_auc"]),
        ],
        "plain_vs_bn": plain_vs_bn_results,
        "architectures": arch_results,
        "class_weight": cw_results,
        "embedding": emb_results,
    }
    write_conclusion_table(all_results)
    print("\n✅ Hoàn tất pipeline. Xem báo cáo trong thư mục reports/ và model tốt nhất trong models/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TT-25 — MLP Keras cho chấm điểm khách hàng bảo hiểm ô tô")
    parser.add_argument("--data", type=str, default="data/train.csv",
                         help="Đường dẫn tới file train.csv tải từ Kaggle")
    args = parser.parse_args()
    main(args.data)
