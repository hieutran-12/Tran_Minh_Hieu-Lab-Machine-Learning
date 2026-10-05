"""Bước 8, 9, 11: bảng so sánh 4 cách, ma trận nhầm lẫn 4x4, human-in-the-loop, kết luận chi phí/lợi ích."""
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import config as C
from utils import doc_ket_qua, fit_temperature, xac_suat

ORDER = [("tfidf_svc", "TF-IDF + LinearSVC"), ("lstm", "BiLSTM"),
         ("transformer_scratch", "Transformer tự xây"), ("distilbert", "DistilBERT")]
README = C.ROOT / "README.md"
MARK = ("<!-- KET_QUA_START -->", "<!-- KET_QUA_END -->")


def nap_ket_qua():
    res = {}
    for key, label in ORDER:
        try:
            m, z = doc_ket_qua(key)
            res[label] = (m, z)
        except FileNotFoundError:
            print(f"[bỏ qua] chưa có kết quả của '{label}' — hãy chạy bước tương ứng trước.")
    if not res:
        raise SystemExit("Chưa có kết quả nào trong reports/. Chạy: python src/train.py all")
    return res


def _md(df):
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, r in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in r.values]) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ bước 8
def bang_so_sanh(res):
    rows = []
    for label, (m, _) in res.items():
        rows.append({"Model": label, "accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
                     "train_s": m["train_time_s"], "params": m["params"],
                     "predict_ms": m["predict_ms"], "device": m["device"]})
    df = pd.DataFrame(rows).set_index("Model")
    df.to_csv(C.REPORTS / "bang_so_sanh.csv")

    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    for a, (col, tit, log) in zip(ax, [("accuracy", "Accuracy (test)", False),
                                       ("train_s", "Thời gian train (s, log)", True),
                                       ("params", "Số tham số (log)", True),
                                       ("predict_ms", "Dự đoán 1 tin (ms, log)", True)]):
        a.bar(range(len(df)), df[col], color=plt.cm.tab10(range(len(df))))
        a.set_xticks(range(len(df)))
        a.set_xticklabels([n.replace(" ", "\n", 1) for n in df.index], fontsize=8)
        a.set_title(tit)
        if log:
            a.set_yscale("log")
        else:
            a.set_ylim(max(0, df[col].min() - 0.03), min(1, df[col].max() + 0.015))
        for i, v in enumerate(df[col]):
            a.text(i, v, f"{v:.3f}" if col == "accuracy" else f"{v:,.4g}", ha="center",
                   va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(C.REPORTS / "so_sanh_4_cach.png", dpi=130)
    plt.show()

    show = pd.DataFrame({"Accuracy": (df["accuracy"] * 100).map("{:.2f}%".format),
                         "F1-macro": (df["f1_macro"] * 100).map("{:.2f}%".format),
                         "Thời gian train (s)": df["train_s"].map("{:,.0f}".format),
                         "Số tham số": df["params"].map("{:,}".format),
                         "Dự đoán 1 tin (ms)": df["predict_ms"].map("{:.2f}".format),
                         "Thiết bị": df["device"]})
    show.index.name = "Model"
    return df, show


# ------------------------------------------------------------------ bước 9
def _ve_cm(ax, cm, title, chuan_hoa):
    mat = cm / cm.sum(1, keepdims=True) if chuan_hoa else cm
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=1 if chuan_hoa else None)
    for i in range(len(cm)):
        for j in range(len(cm)):
            txt = f"{mat[i, j]:.1%}" if chuan_hoa else f"{cm[i, j]}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if mat[i, j] > mat.max() * 0.55 else "black")
    ax.set_xticks(range(len(C.CLASSES)))
    ax.set_xticklabels(C.CLASSES, rotation=30, fontsize=8)
    ax.set_yticks(range(len(C.CLASSES)))
    ax.set_yticklabels(C.CLASSES, fontsize=8)
    ax.set_xlabel("dự đoán")
    ax.set_ylabel("thực tế")
    ax.set_title(title, fontsize=10)


def ma_tran_nham_lan(res, df):
    cms, pairs = {}, []
    for label, (_, z) in res.items():
        pred = z["test_logits"].argmax(1)
        cms[label] = confusion_matrix(z["test_y"], pred, labels=range(len(C.CLASSES)))
        cm = cms[label]
        ps = sorted(((cm[i, j] + cm[j, i], i, j) for i in range(4) for j in range(i + 1, 4)),
                    reverse=True)
        for n, i, j in ps[:3]:
            pairs.append({"Model": label, "cặp chuyên mục": f"{C.CLASSES[i]} ↔ {C.CLASSES[j]}",
                          "số tin nhầm (2 chiều)": int(n),
                          "% tổng số lỗi": f"{n / max(cm.sum() - np.trace(cm), 1):.1%}"})
    best = df["accuracy"].idxmax()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    _ve_cm(ax[0], cms[best], f"{best} — số tin", False)
    _ve_cm(ax[1], cms[best], f"{best} — tỉ lệ theo hàng (recall)", True)
    plt.tight_layout()
    plt.savefig(C.REPORTS / "confusion_4x4.png", dpi=130)
    plt.show()

    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), squeeze=False)
    for a, (label, cm) in zip(axes[0], cms.items()):
        _ve_cm(a, cm, label, True)
    plt.tight_layout()
    plt.savefig(C.REPORTS / "confusion_all_models.png", dpi=130)
    plt.show()

    pdf = pd.DataFrame(pairs).set_index("Model")
    pdf.to_csv(C.REPORTS / "cap_nham_lan.csv")
    return pdf


# ------------------------------------------------------------------ bước 11
def _hitl_row(conf, pred, y, th):
    auto = conf >= th
    p_auto = auto.mean()
    acc_auto = float((pred == y)[auto].mean()) if auto.any() else float("nan")
    n_ed = round(C.NEWS_PER_DAY * (1 - p_auto))
    return {"ngưỡng": th, "% tin tự động": p_auto, "accuracy phần tự động": acc_auto,
            "% tin cần biên tập viên": 1 - p_auto, "tin/ngày cho biên tập viên": n_ed,
            "số biên tập viên cần": math.ceil(n_ed / C.NEWS_PER_EDITOR),
            "accuracy toàn hệ thống": 1 - (1 - acc_auto) * p_auto if auto.any() else 1.0}


def human_in_the_loop(res):
    """Ngưỡng tin cậy: model chắc >= ngưỡng thì tự gán chuyên mục, còn lại chuyển biên tập viên.

    Xác suất được hiệu chỉnh bằng temperature scaling trên tập val (cùng cách cho cả 4 model),
    vì softmax thô của mạng nơ-ron thường quá tự tin, còn SVM không có xác suất.
    """
    rows, curves = [], {}
    for label, (_, z) in res.items():
        T = fit_temperature(z["val_logits"], z["val_y"])
        p = xac_suat(z["test_logits"], T)
        conf, pred, y = p.max(1), p.argmax(1), z["test_y"]
        for th in [0.5, 0.7, 0.8, 0.9, C.THRESHOLD, 0.99]:
            rows.append({"Model": label, "T": round(T, 2), **_hitl_row(conf, pred, y, th)})
        curves[label] = [_hitl_row(conf, pred, y, th) for th in np.linspace(0.4, 0.995, 40)]
    full = pd.DataFrame(rows).set_index("Model")
    full.to_csv(C.REPORTS / "bang_human_in_the_loop_full.csv")
    main = full[np.isclose(full["ngưỡng"], C.THRESHOLD)].drop(columns=["ngưỡng", "T"])
    main.to_csv(C.REPORTS / "bang_human_in_the_loop.csv")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for label, cv in curves.items():
        th = [r["ngưỡng"] for r in cv]
        ax[0].plot(th, [r["% tin tự động"] for r in cv], label=label)
        ax[1].plot(th, [r["accuracy phần tự động"] for r in cv], label=label)
    for a, t in zip(ax, ["% tin được tự động hoá", "accuracy của phần tự động"]):
        a.axvline(C.THRESHOLD, color="gray", ls="--")
        a.set_title(t)
        a.set_xlabel("ngưỡng tin cậy")
    ax[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(C.REPORTS / "hitl_curve.png", dpi=130)
    plt.show()

    show = pd.DataFrame({
        "% tin tự động": (main["% tin tự động"] * 100).map("{:.1f}%".format),
        "accuracy phần tự động": (main["accuracy phần tự động"] * 100).map("{:.2f}%".format),
        "tin/ngày cho biên tập viên": main["tin/ngày cho biên tập viên"].map("{:,}".format),
        "số biên tập viên": main["số biên tập viên cần"],
        "accuracy toàn hệ thống*": (main["accuracy toàn hệ thống"] * 100).map("{:.2f}%".format)})
    show.index.name = "Model"
    return full, show


# ------------------------------------------------------------------ kết luận
def ket_luan(df, hitl_full):
    """Sinh đoạn kết luận chi phí/lợi ích từ CHÍNH số đo của lần chạy này."""
    L = []
    svc, bert, tf_ = "TF-IDF + LinearSVC", "DistilBERT", "Transformer tự xây"
    n_test = None
    if svc in df.index and tf_ in df.index:
        g = (df.loc[tf_, "accuracy"] - df.loc[svc, "accuracy"]) * 100
        L.append(f"- **Transformer tự xây {'vượt' if g > 0 else 'thua'} baseline TF-IDF+SVC {abs(g):.2f} điểm %** "
                 f"({df.loc[tf_, 'accuracy']:.2%} vs {df.loc[svc, 'accuracy']:.2%}). "
                 + ("Kiến trúc hiện đại chưa chắc đã thắng phương pháp cổ điển khi dữ liệu vừa phải: "
                    "Transformer cần rất nhiều dữ liệu (hoặc tiền huấn luyện) mới phát huy." if g <= 0 else
                    "Transformer tự xây đã bắt kịp/vượt baseline, nhưng cần so với chi phí huấn luyện và độ trễ cao hơn "
                    "(xem bảng) trước khi chọn."))
    if svc in df.index and bert in df.index:
        a, b = df.loc[svc], df.loc[bert]
        gap = (b["accuracy"] - a["accuracy"]) * 100
        err_cut = ((1 - a["accuracy"]) - (1 - b["accuracy"])) / (1 - a["accuracy"]) * 100
        L.append(f"- **DistilBERT hơn baseline {gap:+.2f} điểm % accuracy** (giảm ~{err_cut:.0f}% số tin phân loại sai), "
                 f"đổi lại huấn luyện chậm ~{b['train_s'] / max(a['train_s'], 1e-9):.0f}×, "
                 f"~{b['params'] / a['params']:.0f}× tham số, dự đoán 1 tin chậm ~{b['predict_ms'] / max(a['predict_ms'], 1e-9):.0f}× "
                 f"(DistilBERT chạy trên {b['device'].upper()}).")
        if gap < 1:
            v = "chênh lệch dưới 1 điểm % — khó biện minh cho chi phí GPU; nên dùng TF-IDF+SVC."
        elif gap < 3:
            v = ("chênh lệch vài điểm % — đáng dùng nếu mỗi lỗi phân loại tốn kém hoặc đã có sẵn GPU; "
                 "nếu chỉ cần tiết kiệm chi phí vận hành, TF-IDF+SVC vẫn là lựa chọn hợp lý.")
        else:
            v = ("chênh lệch từ 3 điểm % trở lên — thường xứng đáng với chi phí GPU, "
                 "nhưng hãy đối chiếu với ngân sách vận hành thực tế của toà soạn.")
        L.append(f"- **Kết luận chi phí/lợi ích:** {v}")
    th = hitl_full[np.isclose(hitl_full["ngưỡng"], C.THRESHOLD)]
    if len(th):
        best = th["số biên tập viên cần"].idxmin()
        r = th.loc[best]
        L.append(f"- **Human-in-the-loop (ngưỡng {C.THRESHOLD:.0%}):** model tốt nhất về tải biên tập là *{best}*: "
                 f"tự động {r['% tin tự động']:.1%} tin, còn {int(r['tin/ngày cho biên tập viên']):,} tin/ngày "
                 f"(~{int(r['số biên tập viên cần'])} biên tập viên thay vì ~{math.ceil(C.NEWS_PER_DAY / C.NEWS_PER_EDITOR)} "
                 f"nếu làm thủ công).")
    return "\n".join(L)


def cap_nhat_readme(show, hitl_show, pairs, kl):
    if not README.exists():
        return
    txt = README.read_text(encoding="utf-8")
    if MARK[0] not in txt:
        return
    body = (f"\n### Bảng so sánh 4 cách (test set)\n\n{_md(show)}\n\n"
            f"![so sánh](reports/so_sanh_4_cach.png)\n\n"
            f"### Cặp chuyên mục hay bị nhầm\n\n{_md(pairs)}\n\n"
            f"![confusion](reports/confusion_4x4.png)\n\n"
            f"### Human-in-the-loop (ngưỡng {C.THRESHOLD:.0%}, {C.NEWS_PER_DAY:,} tin/ngày)\n\n{_md(hitl_show)}\n\n"
            f"\\* giả định biên tập viên sửa đúng 100% số tin được chuyển cho họ.\n\n"
            f"### Kết luận\n\n{kl}\n")
    new = re.sub(re.escape(MARK[0]) + r".*?" + re.escape(MARK[1]),
                 lambda _: MARK[0] + body + MARK[1], txt, flags=re.S)
    README.write_text(new, encoding="utf-8")


def run():
    res = nap_ket_qua()
    df, show = bang_so_sanh(res)
    print(_md(show), "\n")
    pairs = ma_tran_nham_lan(res, df)
    print(_md(pairs), "\n")
    full, hitl_show = human_in_the_loop(res)
    print(_md(hitl_show), "\n")
    kl = ket_luan(df, full)
    print(kl)
    (C.REPORTS / "bao_cao_ket_qua.md").write_text(
        f"## So sánh 4 cách\n\n{_md(show)}\n\n## Cặp hay nhầm\n\n{_md(pairs)}\n\n"
        f"## Human-in-the-loop\n\n{_md(hitl_show)}\n\n## Kết luận\n\n{kl}\n", encoding="utf-8")
    cap_nhat_readme(show, hitl_show, pairs, kl)
    return show, pairs, hitl_show, kl
