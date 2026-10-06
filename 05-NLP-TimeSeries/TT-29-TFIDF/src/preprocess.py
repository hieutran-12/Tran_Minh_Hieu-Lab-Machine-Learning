"""Tải dữ liệu 20 Newsgroups và dựng pipeline TF-IDF."""
from functools import lru_cache

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

SEED = 42
REMOVE = ("headers", "footers", "quotes")   # bắt buộc, nếu không sẽ rò rỉ nhãn
NGUONG_TIN_CAY = 0.6                         # max(proba) < 0,6 -> chuyển người xử lý


@lru_cache(maxsize=None)
def load_20ng(remove=True, subset="train"):
    """Trả về (văn bản, nhãn, tên lớp). remove=True -> bỏ headers/footers/quotes."""
    from sklearn.datasets import fetch_20newsgroups
    kw = {"remove": REMOVE} if remove else {}
    b = fetch_20newsgroups(subset=subset, shuffle=True, random_state=SEED, **kw)
    return list(b.data), np.asarray(b.target), list(b.target_names)


def make_vectorizer(**override):
    """TfidfVectorizer cấu hình chuẩn của đề bài; override để khảo sát tham số."""
    params = dict(lowercase=True, ngram_range=(1, 2),
                  min_df=3,            # bỏ từ < 3 văn bản (lỗi chính tả)
                  max_df=0.8,          # bỏ từ có mặt > 80% văn bản
                  sublinear_tf=True,   # 1 + log(TF)
                  max_features=50000)
    params.update(override)
    return TfidfVectorizer(**params)


def build_pipeline(C=1.0):
    """TF-IDF + LinearSVC bọc CalibratedClassifierCV để có predict_proba."""
    return make_pipeline(
        make_vectorizer(),
        CalibratedClassifierCV(LinearSVC(C=C, random_state=SEED), cv=3, ensemble=False),
    )


def tach_tu_vn(s):
    """Tiếng Việt: tách từ ghép ("học sinh" -> "học_sinh"). Cần: pip install underthesea.
    Dùng: make_vectorizer(preprocessor=tach_tu_vn)."""
    from underthesea import word_tokenize
    return word_tokenize(s, format="text")
