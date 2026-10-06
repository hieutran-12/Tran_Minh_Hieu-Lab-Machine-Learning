"""TT-29 — TF-IDF phân loại ticket. Chạy toàn bộ 11 bước:

    python src/train.py                  # chạy đầy đủ
    python src/train.py --extensions     # thêm phần mở rộng (LSA, SGD partial_fit)

Kết quả: reports/*.png, reports/*.csv, reports/bao_cao.md, models/tfidf_pipeline.joblib
và các bảng kết quả được tự chèn vào README.md.
"""
import argparse
import json
import re
import time
import warnings
from itertools import product
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

from preprocess import (NGUONG_TIN_CAY, SEED, build_pipeline, load_20ng,
                        make_vectorizer)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]
REPORTS, MODELS = ROOT / "reports", ROOT / "models"
REPORTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

# Nhóm chủ đề lớn — dùng để gợi ý nguyên nhân ở bước 11
NHOM = {
    "alt.atheism": "tôn giáo", "soc.religion.christian": "tôn giáo", "talk.religion.misc": "tôn giáo",
    "comp.graphics": "máy tính", "comp.os.ms-windows.misc": "máy tính", "comp.sys.ibm.pc.hardware": "máy tính",
    "comp.sys.mac.hardware": "máy tính", "comp.windows.x": "máy tính",
    "rec.autos": "xe cộ", "rec.motorcycles": "xe cộ",
    "rec.sport.baseball": "thể thao", "rec.sport.hockey": "thể thao",
    "talk.politics.guns": "chính trị", "talk.politics.mideast": "chính trị", "talk.politics.misc": "chính trị",
    "sci.crypt": "khoa học", "sci.electronics": "khoa học", "sci.med": "khoa học", "sci.space": "khoa học",
    "misc.forsale": "rao vặt",
}
# Token thuộc header/footer — nếu lọt vào top từ nghĩa là còn rò rỉ
TU_RO_RI = {"newsgroups", "xref", "nntp", "posting", "host", "organization", "lines",
            "subject", "path", "message", "distribution", "reply", "writes", "wrote", "article"}


# ----------------------------------------------------------------- tiện ích
def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(REPORTS / name, dpi=130)
    plt.close(fig)


def md_table(df, fmt="{:.4f}"):
    """DataFrame -> bảng Markdown (không cần thư viện tabulate)."""
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.astype(object).iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                      for v in r.values) + " |")
    return "\n".join(out)


def _data():
    Xtr, ytr, names = load_20ng(True, "train")
    Xte, yte, _ = load_20ng(True, "test")
    return Xtr, ytr, Xte, yte, names


def _scores(y, p):
    return accuracy_score(y, p), f1_score(y, p, average="macro", zero_division=0)


# ----------------------------------------------------------------- Bước 1
def b1_ro_ri():
    """Chạy 2 lần: CÓ và KHÔNG remove headers/footers/quotes."""
    rows, evidence = [], {}
    for remove in (False, True):
        Xtr, ytr, names = load_20ng(remove, "train")
        Xte, yte, _ = load_20ng(remove, "test")
        pipe = make_pipeline(make_vectorizer(), LinearSVC(C=1.0, random_state=SEED)).fit(Xtr, ytr)
        acc, f1 = _scores(yte, pipe.predict(Xte))
        rows.append({"Cấu hình": "remove headers/footers/quotes" if remove else "KHÔNG remove (rò rỉ)",
                     "Accuracy": acc, "Macro-F1": f1})
        if not remove:   # bằng chứng: từ mạnh nhất của lớp đầu tiên khi còn header
            fn = np.array(pipe[0].get_feature_names_out())
            evidence = {"lop": names[0], "tu": list(fn[np.argsort(pipe[1].coef_[0])[::-1][:8]])}
    df = pd.DataFrame(rows)
    gap = df.Accuracy[0] - df.Accuracy[1]
    nhan_xet = (f"Khi KHÔNG bỏ headers, accuracy = {df.Accuracy[0]:.3f}; sau khi bỏ chỉ còn {df.Accuracy[1]:.3f} "
                f"(chênh {gap:.3f}). Phần chênh này là rò rỉ: mô hình đọc dòng 'Newsgroups: ...' "
                f"trong header — chính là đáp án. Ví dụ từ trọng số cao nhất của lớp '{evidence['lop']}' "
                f"trong bản rò rỉ: {', '.join(evidence['tu'])}. Từ bước 2 chỉ dùng bản đã remove.")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(["Có rò rỉ\n(không remove)", "Đã remove\n(đúng)"], df.Accuracy, color=["#d9534f", "#2e8b57"])
    ax.bar_label(bars, fmt="%.3f")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy (tập test)")
    ax.set_title("Rò rỉ do headers/footers/quotes")
    save_fig(fig, "leakage_comparison.png")
    df.to_csv(REPORTS / "leakage_comparison.csv", index=False)
    return df, nhan_xet


# ----------------------------------------------------------------- Bước 2
def b2_eda():
    Xtr, ytr, _, _, names = _data()
    counts = pd.Series(np.array(names)[ytr]).value_counts().sort_index().rename("Số văn bản")
    lens = pd.Series([len(t.split()) for t in Xtr])
    stats = lens.describe().round(1).rename("Số từ / văn bản").to_frame().T
    stats["Văn bản rỗng"] = int((lens == 0).sum())
    cv = CountVectorizer(stop_words="english", max_features=20)
    freq = np.asarray(cv.fit_transform(Xtr).sum(0)).ravel()
    top = pd.DataFrame({"Từ": cv.get_feature_names_out(), "Số lần": freq}).sort_values("Số lần", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    counts.plot.barh(ax=axes[0], color="#4c72b0"); axes[0].set_title("Số văn bản mỗi lớp (train)")
    axes[1].hist(lens.clip(upper=1500), bins=50, color="#55a868")
    axes[1].set_title("Độ dài văn bản (số từ, cắt ở 1500)"); axes[1].set_xlabel("số từ")
    axes[2].barh(top["Từ"][::-1], top["Số lần"][::-1], color="#c44e52"); axes[2].set_title("20 từ phổ biến nhất (bỏ stopword)")
    save_fig(fig, "eda.png")
    counts.to_csv(REPORTS / "eda_so_van_ban_moi_lop.csv")
    return counts.reset_index().rename(columns={"index": "Lớp"}), stats, top


# ----------------------------------------------------------------- Bước 3
def b3_baseline():
    Xtr, ytr, Xte, yte, _ = _data()
    models = {"DummyClassifier (lớp phổ biến nhất)": DummyClassifier(strategy="most_frequent"),
              "TF-IDF + Naive Bayes": make_pipeline(make_vectorizer(), MultinomialNB(alpha=0.1))}
    rows = []
    for name, m in models.items():
        acc, f1 = _scores(yte, m.fit(Xtr, ytr).predict(Xte))
        rows.append({"Baseline": name, "Accuracy": acc, "Macro-F1": f1})
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "baseline.csv", index=False)
    return df


# ----------------------------------------------------------------- Bước 4
def b4_fit_final():
    """TF-IDF + LinearSVC (calibrated). Lưu model; các bước 7–11 dùng lại kết quả này."""
    Xtr, ytr, Xte, yte, names = _data()
    pipe = build_pipeline()
    t0 = time.perf_counter()
    pipe.fit(Xtr, ytr)
    t_fit = time.perf_counter() - t0
    proba = pipe.predict_proba(Xte)
    pred = pipe.classes_[proba.argmax(1)]
    acc, f1 = _scores(yte, pred)
    rep = pd.DataFrame(classification_report(yte, pred, target_names=names, output_dict=True,
                                             zero_division=0)).T.round(3)
    rep.to_csv(REPORTS / "classification_report.csv")
    joblib.dump(pipe, MODELS / "tfidf_pipeline.joblib")
    (MODELS / "labels.json").write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
    return dict(pipe=pipe, proba=proba, pred=pred, acc=acc, f1=f1, t_fit=t_fit, report=rep,
                Xte=Xte, yte=yte, names=names)


# ----------------------------------------------------------------- Bước 5
def b5_khao_sat_vectorizer():
    """ngram_range × min_df × sublinear_tf -> số đặc trưng sinh ra × accuracy."""
    Xtr, ytr, Xte, yte, _ = _data()
    rows = []
    for ng, mdf, sub in product([(1, 1), (1, 2)], [1, 3, 5], [True, False]):
        vec = make_vectorizer(ngram_range=ng, min_df=mdf, sublinear_tf=sub, max_features=None)
        A = vec.fit_transform(Xtr)                      # chỉ fit trên train
        svc = LinearSVC(C=1.0, random_state=SEED).fit(A, ytr)
        acc, f1 = _scores(yte, svc.predict(vec.transform(Xte)))
        rows.append({"ngram_range": str(ng), "min_df": mdf, "sublinear_tf": sub,
                     "Số đặc trưng": A.shape[1], "Accuracy": acc, "Macro-F1": f1})
        print(f"   [b5] ngram={ng} min_df={mdf} sublinear={sub}: {A.shape[1]:,} đặc trưng, acc={acc:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "khao_sat_vectorizer.csv", index=False)
    return df


# ----------------------------------------------------------------- Bước 6
def b6_so_sanh_bo_phan_loai():
    Xtr, ytr, Xte, yte, _ = _data()
    vec = make_vectorizer()
    A, B = vec.fit_transform(Xtr), vec.transform(Xte)
    models = {"Naive Bayes (alpha=0.1)": MultinomialNB(alpha=0.1),
              "Logistic Regression (C=20)": LogisticRegression(C=20, max_iter=300),
              "LinearSVC (C=1)": LinearSVC(C=1.0, random_state=SEED)}
    rows = []
    for name, m in models.items():
        t0 = time.perf_counter(); m.fit(A, ytr); dt = time.perf_counter() - t0
        acc, f1 = _scores(yte, m.predict(B))
        rows.append({"Bộ phân loại": name, "Accuracy": acc, "Macro-F1": f1, "Thời gian train (s)": dt})
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "so_sanh_bo_phan_loai.csv", index=False)
    return df


# ----------------------------------------------------------------- Bước 7
def b7_top_tu(fin, k=15):
    """Top-k từ trọng số cao nhất mỗi lớp (trung bình hệ số của 3 fold calibrate)."""
    pipe, names = fin["pipe"], fin["names"]
    fn = np.array(pipe[0].get_feature_names_out())
    coef = np.mean([c.estimator.coef_ for c in pipe[-1].calibrated_classifiers_], axis=0)
    rows, flagged = [], []
    for i, name in enumerate(names):
        idx = np.argsort(coef[i])[::-1][:k]
        words = list(fn[idx])
        for r, (w, wt) in enumerate(zip(words, coef[i][idx]), 1):
            rows.append({"Lớp": name, "Hạng": r, "Từ": w, "Trọng số": round(float(wt), 4)})
        bad = [w for w in words if TU_RO_RI & set(w.split())]
        if bad:
            flagged.append(f"{name}: {', '.join(bad)}")
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "top_tu_moi_lop.csv", index=False)

    fig, axes = plt.subplots(5, 4, figsize=(24, 26))
    for ax, name in zip(axes.ravel(), names):
        d = df[df["Lớp"] == name].iloc[::-1]
        ax.barh(d["Từ"], d["Trọng số"], color="#4c72b0"); ax.set_title(name, fontsize=12)
        ax.tick_params(labelsize=9)
    fig.suptitle("Top 15 từ có trọng số cao nhất mỗi lớp (TF-IDF + LinearSVC)", fontsize=16)
    save_fig(fig, "top_tu_moi_lop.png")
    wide = df.pivot(index="Hạng", columns="Lớp", values="Từ")
    return wide, flagged


# ----------------------------------------------------------------- Bước 8
def b8_confusion(fin):
    yte, pred, names = fin["yte"], fin["pred"], fin["names"]
    cm = confusion_matrix(yte, pred)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế"); ax.set_title("Ma trận nhầm lẫn (chuẩn hoá theo hàng)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    save_fig(fig, "confusion_matrix.png")
    pairs = [{"Thực tế": names[i], "Bị nhầm thành": names[j], "Số ca": int(cm[i, j]),
              "Tỉ lệ nhầm": float(cmn[i, j])}
             for i in range(len(names)) for j in range(len(names)) if i != j and cm[i, j] > 0]
    df = pd.DataFrame(pairs).sort_values("Số ca", ascending=False).head(10).reset_index(drop=True)
    df.to_csv(REPORTS / "cap_nham_lan_nhieu_nhat.csv", index=False)
    return df


# ----------------------------------------------------------------- Bước 9
def b9_nguong_tin_cay(fin):
    """max(proba) < ngưỡng -> chuyển người xử lý."""
    conf = fin["proba"].max(1)
    correct = fin["pred"] == fin["yte"]
    rows = []
    for thr in [0.3, 0.4, 0.5, NGUONG_TIN_CAY, 0.7, 0.8, 0.9]:
        auto = conf >= thr
        acc_auto = correct[auto].mean() if auto.any() else float("nan")
        rows.append({"Ngưỡng": thr, "% tự động": 100 * auto.mean(), "Accuracy phần tự động": acc_auto,
                     "% cần người": 100 * (1 - auto.mean()),
                     "Accuracy phần cần người (nếu để máy quyết)": correct[~auto].mean() if (~auto).any() else float("nan")})
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "nguong_tin_cay.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["Ngưỡng"], df["% tự động"] / 100, "o-", label="% tự động xử lý")
    ax.plot(df["Ngưỡng"], df["Accuracy phần tự động"], "s-", label="Accuracy phần tự động")
    ax.axvline(NGUONG_TIN_CAY, ls="--", c="gray"); ax.text(NGUONG_TIN_CAY + 0.01, 0.05, "ngưỡng 0,6")
    ax.set_xlabel("Ngưỡng tin cậy"); ax.set_ylim(0, 1.05); ax.grid(alpha=.3); ax.legend()
    ax.set_title("Đánh đổi: tự động hoá vs độ chính xác")
    save_fig(fig, "nguong_tin_cay.png")
    return df


# ----------------------------------------------------------------- Bước 10
def b10_thoi_gian(fin, n_doc=300):
    pipe, Xte = fin["pipe"], fin["Xte"]
    docs = Xte[:n_doc]
    for d in docs[:10]:                                  # khởi động
        pipe.predict_proba([d])
    ts = []
    for d in docs:
        t0 = time.perf_counter(); pipe.predict_proba([d]); ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    df = pd.DataFrame([
        {"Chỉ số": "Train toàn bộ pipeline (giây)", "Giá trị": round(fin["t_fit"], 2)},
        {"Chỉ số": f"Dự đoán 1 ticket — trung bình (ms, {n_doc} ticket)", "Giá trị": round(ts.mean(), 3)},
        {"Chỉ số": "Dự đoán 1 ticket — trung vị (ms)", "Giá trị": round(float(np.median(ts)), 3)},
        {"Chỉ số": "Dự đoán 1 ticket — p95 (ms)", "Giá trị": round(float(np.percentile(ts, 95)), 3)},
        {"Chỉ số": "Đạt yêu cầu < 5 ms (trung bình)", "Giá trị": "ĐẠT" if ts.mean() < 5 else "KHÔNG ĐẠT"},
    ])
    df.to_csv(REPORTS / "thoi_gian.csv", index=False)
    return df


# ----------------------------------------------------------------- Bước 11
def goi_y_nguyen_nhan(text, true, pred, conf):
    if len(text.split()) < 20:
        return "Văn bản quá ngắn/rỗng — thiếu từ khoá để phân biệt"
    if NHOM[true] == NHOM[pred]:
        return f"Hai lớp cùng nhóm '{NHOM[true]}' — từ vựng chồng lấn mạnh"
    if conf < 0.4:
        return "Mô hình không chắc — nội dung nằm giữa nhiều chủ đề"
    return "Chứa từ khoá đặc trưng của lớp dự đoán, hoặc nhãn gốc nhiễu / chủ đề lệch"


def b11_ca_sai(fin, n=10):
    """10 ca sai: 5 ca sai với độ tin cậy cao nhất + 5 ca sai ngẫu nhiên. Cột nguyên nhân là GỢI Ý tự động."""
    Xte, yte, pred, names = fin["Xte"], fin["yte"], fin["pred"], fin["names"]
    conf = fin["proba"].max(1)
    wrong = np.where(pred != yte)[0]
    top = wrong[np.argsort(-conf[wrong])][: n // 2]
    rest = np.setdiff1d(wrong, top)
    rnd = np.random.RandomState(SEED).choice(rest, n - len(top), replace=False)
    rows = []
    for i in list(top) + list(rnd):
        snippet = re.sub(r"\s+", " ", Xte[i]).strip()[:200]
        rows.append({"Thực tế": names[yte[i]], "Dự đoán": names[pred[i]], "Độ tin cậy": float(conf[i]),
                     "Nguyên nhân (gợi ý)": goi_y_nguyen_nhan(Xte[i], names[yte[i]], names[pred[i]], conf[i]),
                     "Trích đoạn": snippet})
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "phan_tich_10_ca_sai.csv", index=False, encoding="utf-8-sig")
    return df


# ----------------------------------------------------------------- Hạn chế
def han_che(fin):
    """TF-IDF không hiểu NGHĨA: đồng nghĩa nhưng khác mặt chữ -> độ tương đồng ≈ 0."""
    vec = fin["pipe"][0]
    pairs = [("the car is very fast", "the automobile is really quick"),
             ("the car is very fast", "the car is very fast indeed"),
             ("great graphics card", "excellent video adapter")]
    rows = [{"Câu A": a, "Câu B": b,
             "Cosine TF-IDF": float(cosine_similarity(vec.transform([a]), vec.transform([b]))[0, 0])}
            for a, b in pairs]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- Mở rộng
def mo_rong(fin):
    """(1) SVD/LSA giảm chiều; (2) học trực tuyến SGDClassifier.partial_fit."""
    Xtr, ytr, Xte, yte, _ = _data()
    vec = make_vectorizer()
    A, B = vec.fit_transform(Xtr), vec.transform(Xte)
    rows = [{"Phương pháp": "TF-IDF + LinearSVC (50.000 chiều)", "Accuracy": fin["acc"],
             "Thời gian fit (s)": fin["t_fit"]}]
    t0 = time.perf_counter()
    lsa = make_pipeline(TruncatedSVD(300, random_state=SEED), Normalizer(copy=False)).fit(A)
    svc = LinearSVC(C=1.0, random_state=SEED).fit(lsa.transform(A), ytr)
    rows.append({"Phương pháp": "LSA 300 chiều + LinearSVC",
                 "Accuracy": accuracy_score(yte, svc.predict(lsa.transform(B))),
                 "Thời gian fit (s)": time.perf_counter() - t0})
    hv = HashingVectorizer(n_features=2 ** 18, ngram_range=(1, 2), alternate_sign=False, norm="l2")
    H, Ht = hv.transform(Xtr), hv.transform(Xte)
    sgd = SGDClassifier(loss="hinge", alpha=1e-5, random_state=SEED)
    chunks = np.array_split(np.arange(len(ytr)), 5)
    online = []
    for k, idx in enumerate(chunks, 1):
        sgd.partial_fit(H[idx], ytr[idx], classes=np.arange(20))
        online.append({"Đã học (số lô)": k, "Số ticket đã thấy": int(sum(len(c) for c in chunks[:k])),
                       "Accuracy": accuracy_score(yte, sgd.predict(Ht))})
    return pd.DataFrame(rows), pd.DataFrame(online)


# ----------------------------------------------------------------- Báo cáo
def inject_readme(text):
    p = ROOT / "README.md"
    if not p.exists():
        return
    s, a, b = p.read_text(encoding="utf-8"), "<!-- KET-QUA:BAT-DAU -->", "<!-- KET-QUA:KET-THUC -->"
    if a in s and b in s:
        pre, rest = s.split(a, 1)
        post = rest.split(b, 1)[1]
        p.write_text(f"{pre}{a}\n{text}\n{b}{post}", encoding="utf-8")


def main(extensions=False):
    plt.switch_backend("Agg")
    md = []

    def sec(title, *parts):
        md.append(f"### {title}\n")
        md.extend(parts)
        md.append("")

    t0 = time.perf_counter()
    print("[1/11] Chứng minh rò rỉ headers/footers/quotes ..."); df1, nx = b1_ro_ri()
    sec("Bước 1 — Bảng chứng minh rò rỉ", md_table(df1), "", nx, "", "![leak](reports/leakage_comparison.png)")
    print("[2/11] EDA ..."); _, st, top = b2_eda()
    sec("Bước 2 — EDA", md_table(st, "{:.1f}"), "", md_table(top, "{:,}"), "", "![eda](reports/eda.png)")
    print("[3/11] Baseline ..."); df3 = b3_baseline()
    sec("Bước 3 — Baseline", md_table(df3))
    print("[4/11] TF-IDF + LinearSVC (calibrated) ...")
    fin = b4_fit_final()
    sec("Bước 4 — TF-IDF + LinearSVC", f"Accuracy = **{fin['acc']:.4f}**, Macro-F1 = **{fin['f1']:.4f}** (tập test, 20 lớp).")
    print(f"       accuracy={fin['acc']:.4f}  macro-F1={fin['f1']:.4f}")
    print("[5/11] Khảo sát tham số vectorizer (12 tổ hợp, mất vài phút) ..."); df5 = b5_khao_sat_vectorizer()
    sec("Bước 5 — Khảo sát tham số vectorizer", md_table(df5))
    print("[6/11] So sánh 3 bộ phân loại ..."); df6 = b6_so_sanh_bo_phan_loai()
    sec("Bước 6 — So sánh bộ phân loại (cùng TF-IDF)", md_table(df6))
    print("[7/11] Top 15 từ mỗi lớp ..."); wide, flagged = b7_top_tu(fin)
    kiem = ("⚠️ Có từ nghi rò rỉ: " + "; ".join(flagged)) if flagged else \
           "✅ Không có token header/footer nào lọt vào top 15 của bất kỳ lớp nào."
    sec("Bước 7 — Top 15 từ đặc trưng mỗi lớp", kiem, "",
        "Danh sách đầy đủ: `reports/top_tu_moi_lop.csv` · ![top](reports/top_tu_moi_lop.png)")
    print("[8/11] Ma trận nhầm lẫn ..."); df8 = b8_confusion(fin)
    sec("Bước 8 — Các cặp lớp hay nhầm nhất", md_table(df8), "", "![cm](reports/confusion_matrix.png)")
    print("[9/11] Ngưỡng tin cậy ..."); df9 = b9_nguong_tin_cay(fin)
    sec("Bước 9 — Cơ chế ngưỡng tin cậy (max(proba) < 0,6 → chuyển người)", md_table(df9, "{:.3f}"),
        "", "![thr](reports/nguong_tin_cay.png)")
    print("[10/11] Đo thời gian ..."); df10 = b10_thoi_gian(fin)
    sec("Bước 10 — Thời gian", md_table(df10, "{}"))
    print("[11/11] Phân tích ca sai ..."); df11 = b11_ca_sai(fin)
    sec("Bước 11 — 10 ca sai", "*Cột nguyên nhân là gợi ý tự động theo luật đơn giản; cần đọc lại trích đoạn để kết luận.*",
        "", md_table(df11, "{:.2f}"))
    sec("Hạn chế — TF-IDF không hiểu NGHĨA", md_table(han_che(fin)),
        "", "Từ đồng nghĩa nhưng khác mặt chữ (car/automobile, fast/quick) cho độ tương đồng gần 0.")
    if extensions:
        print("[+] Mở rộng: LSA & SGD partial_fit ..."); a, b = mo_rong(fin)
        sec("Mở rộng", md_table(a), "", md_table(b))

    report = "\n".join(md)
    (REPORTS / "bao_cao.md").write_text("# Báo cáo kết quả TT-29\n\n" + report.replace("](reports/", "]("),
                                        encoding="utf-8")
    inject_readme(report)
    print(f"\nXong trong {time.perf_counter() - t0:.0f}s. Xem reports/, models/ và mục Kết quả trong README.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extensions", action="store_true", help="chạy thêm LSA và SGD partial_fit")
    main(ap.parse_args().extensions)
