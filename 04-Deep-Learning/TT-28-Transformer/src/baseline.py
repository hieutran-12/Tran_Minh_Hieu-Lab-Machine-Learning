"""Baseline bắt buộc: TF-IDF + LinearSVC (mốc mà Transformer phải vượt qua)."""
import time

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from config import MODELS
from utils import bao_cao_phan_loai, do_ms_mot_tin, luu_ket_qua, set_seed, tinh_metrics


def run(data):
    set_seed()
    x_tr, y_tr = data["train"]
    x_val, y_val = data["val"]
    x_te, y_te = data["test"]

    t0 = time.perf_counter()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=300_000)
    xt = vec.fit_transform(x_tr)
    svc = LinearSVC(C=0.5).fit(xt, y_tr)
    train_time = time.perf_counter() - t0

    # "logits" của SVC = khoảng cách tới siêu phẳng (decision_function); so_sanh.py sẽ hiệu chỉnh thành xác suất
    val_logits = svc.decision_function(vec.transform(x_val))
    test_logits = svc.decision_function(vec.transform(x_te))

    ms = do_ms_mot_tin(lambda t: svc.predict(vec.transform([t])), x_te)
    n_params = int(svc.coef_.size + svc.intercept_.size)
    joblib.dump({"vectorizer": vec, "model": svc}, MODELS / "tfidf_svc.joblib")

    kq = luu_ket_qua("tfidf_svc",
                     {"ten": "TF-IDF + LinearSVC", "train_time_s": train_time, "params": n_params,
                      "predict_ms": ms, "device": "cpu", "epochs": None},
                     val_logits, y_val, test_logits, y_te)
    print(f"Val acc {tinh_metrics(y_val, val_logits)['accuracy']:.4f} | "
          f"Test acc {kq['accuracy']:.4f} | F1-macro {kq['f1_macro']:.4f} | "
          f"train {train_time:.1f}s | {ms:.2f} ms/tin")
    bao_cao_phan_loai(y_te, test_logits)
    return kq
