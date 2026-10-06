"""Demo dự đoán 1 ticket + định tuyến theo ngưỡng tin cậy.

    python src/predict.py "My graphics card driver crashes when I open the game"
"""
import json
import sys
from pathlib import Path

import joblib

from preprocess import NGUONG_TIN_CAY

MODELS = Path(__file__).resolve().parents[1] / "models"


def route(text, pipe=None, names=None):
    pipe = pipe or joblib.load(MODELS / "tfidf_pipeline.joblib")
    names = names or json.loads((MODELS / "labels.json").read_text(encoding="utf-8"))
    p = pipe.predict_proba([text])[0]
    top = p.argsort()[::-1][:3]
    conf = float(p[top[0]])
    return {"bo_phan": names[pipe.classes_[top[0]]], "do_tin_cay": round(conf, 3),
            "quyet_dinh": "TỰ ĐỘNG" if conf >= NGUONG_TIN_CAY else "CHUYỂN NGƯỜI XỬ LÝ",
            "top3": [(names[pipe.classes_[i]], round(float(p[i]), 3)) for i in top]}


if __name__ == "__main__":
    r = route(" ".join(sys.argv[1:]) or sys.stdin.read())
    print(json.dumps(r, ensure_ascii=False, indent=2))
