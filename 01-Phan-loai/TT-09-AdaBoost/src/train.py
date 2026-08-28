"""TT-09 — AdaBoost cho phát hiện xâm nhập mạng (NSL-KDD)."""
import time, json, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

from columns import COLUMNS, ATTACK_MAP

RNG = 42
REPORTS = "../reports"

# ---------- 1. Load ----------
train_raw = pd.read_csv("../data/KDDTrain+.txt", header=None, names=COLUMNS)
test_raw = pd.read_csv("../data/KDDTest+.txt", header=None, names=COLUMNS)
print(f"Train: {train_raw.shape}, Test (zero-day): {test_raw.shape}")

for df in (train_raw, test_raw):
    df["attack_type"] = df["label"].map(lambda x: ATTACK_MAP.get(x, "unknown"))
    df["y"] = (df["label"] != "normal").astype(int)
    df.drop(columns=["difficulty", "label"], inplace=True)

cat_cols = ["protocol_type", "service", "flag"]
num_cols = [c for c in train_raw.columns if c not in cat_cols + ["attack_type", "y"]]

X_train_full = train_raw.drop(columns=["attack_type", "y"])
y_train_full = train_raw["y"]
X_zeroday = test_raw.drop(columns=["attack_type", "y"])
y_zeroday = test_raw["y"]

# hold-out split from the TRAIN file for CV-style evaluation (same attack distribution)
X_train, X_cv, y_train, y_cv = train_test_split(
    X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=RNG
)

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols),
])

# ---------- 2. EDA ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
train_raw["attack_type"].value_counts().plot(kind="bar", ax=ax[0], color="#4C72B0")
ax[0].set_title("Phân bố loại tấn công (train)")
train_raw["y"].value_counts().rename({0: "normal", 1: "attack"}).plot(
    kind="pie", ax=ax[1], autopct="%1.1f%%", colors=["#55A868", "#C44E52"])
ax[1].set_title("Tỉ lệ normal / attack")
ax[1].set_ylabel("")
plt.tight_layout()
plt.savefig(f"{REPORTS}/eda_overview.png", dpi=120)
plt.close()
print("U2R chiếm:", (train_raw["attack_type"] == "u2r").mean() * 100, "% -> gộp nhị phân là hợp lý")

# ---------- 3. Baseline ----------
def make_pipe(clf):
    return Pipeline([("pre", pre), ("clf", clf)])

dummy = make_pipe(DummyClassifier(strategy="most_frequent"))
dummy.fit(X_train, y_train)
f1_dummy = f1_score(y_cv, dummy.predict(X_cv))

stump = make_pipe(DecisionTreeClassifier(max_depth=1, random_state=RNG))
stump.fit(X_train, y_train)
f1_stump = f1_score(y_cv, stump.predict(X_cv))
print(f"Baseline — Dummy F1: {f1_dummy:.4f} | 1 Stump F1: {f1_stump:.4f}")

# ---------- 4. AdaBoost (300 stumps) ----------
ada = make_pipe(AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=RNG),
    n_estimators=300, learning_rate=0.5, random_state=RNG,
))
t0 = time.time()
ada.fit(X_train, y_train)
fit_time = time.time() - t0
f1_ada = f1_score(y_cv, ada.predict(X_cv))
print(f"AdaBoost 300 stumps F1: {f1_ada:.4f} (fit {fit_time:.1f}s) — vs 1 stump: {f1_stump:.4f}")

# ---------- 5. F1 vs n_estimators ----------
Xt_cv = ada.named_steps["pre"].transform(X_cv)
f1_curve = [f1_score(y_cv, pred) for pred in ada.named_steps["clf"].staged_predict(Xt_cv)]
plt.figure(figsize=(7, 4))
plt.plot(range(1, len(f1_curve) + 1), f1_curve, color="#4C72B0")
plt.xlabel("Số vòng lặp (n_estimators)")
plt.ylabel("F1-score (holdout CV)")
plt.title("F1 theo số vòng lặp AdaBoost")
plt.tight_layout()
plt.savefig(f"{REPORTS}/f1_theo_vong_lap.png", dpi=120)
plt.close()

# ---------- 6. Noise experiment: flip 5% train labels ----------
rng = np.random.RandomState(RNG)
y_train_noisy = y_train.copy()
flip_idx = rng.choice(y_train_noisy.index, size=int(0.05 * len(y_train_noisy)), replace=False)
y_train_noisy.loc[flip_idx] = 1 - y_train_noisy.loc[flip_idx]

ada_noisy = make_pipe(AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=RNG),
    n_estimators=300, learning_rate=0.5, random_state=RNG))
ada_noisy.fit(X_train, y_train_noisy)
f1_ada_noisy = f1_score(y_cv, ada_noisy.predict(X_cv))

rf_clean = make_pipe(RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1))
rf_clean.fit(X_train, y_train)
f1_rf_clean = f1_score(y_cv, rf_clean.predict(X_cv))

rf_noisy = make_pipe(RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1))
rf_noisy.fit(X_train, y_train_noisy)
f1_rf_noisy = f1_score(y_cv, rf_noisy.predict(X_cv))

noise_table = pd.DataFrame({
    "model": ["AdaBoost", "RandomForest"],
    "F1_clean": [f1_ada, f1_rf_clean],
    "F1_5pct_noise": [f1_ada_noisy, f1_rf_noisy],
})
noise_table["drop_pct"] = (noise_table["F1_clean"] - noise_table["F1_5pct_noise"]) / noise_table["F1_clean"] * 100
print("\n=== THÍ NGHIỆM NHIỄU NHÃN (5%) ===")
print(noise_table.to_string(index=False))

plt.figure(figsize=(6, 4))
x = np.arange(2)
w = 0.35
plt.bar(x - w/2, noise_table["F1_clean"], w, label="Sạch", color="#55A868")
plt.bar(x + w/2, noise_table["F1_5pct_noise"], w, label="Nhiễu 5%", color="#C44E52")
plt.xticks(x, noise_table["model"])
plt.ylabel("F1-score")
plt.title("Ảnh hưởng của nhiễu nhãn: AdaBoost vs Random Forest")
plt.legend()
plt.tight_layout()
plt.savefig(f"{REPORTS}/thi_nghiem_nhieu.png", dpi=120)
plt.close()

# ---------- 7. So sánh 3 thuật toán ensemble ----------
gb = make_pipe(GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=RNG))
gb.fit(X_train, y_train)
f1_gb = f1_score(y_cv, gb.predict(X_cv))

compare_table = pd.DataFrame({
    "model": ["AdaBoost (300 stumps)", "GradientBoosting (TT-07 style)", "RandomForest (TT-03 style)"],
    "F1_cv": [f1_ada, f1_gb, f1_rf_clean],
    "Accuracy_cv": [
        accuracy_score(y_cv, ada.predict(X_cv)),
        accuracy_score(y_cv, gb.predict(X_cv)),
        accuracy_score(y_cv, rf_clean.predict(X_cv)),
    ],
})
print("\n=== SO SÁNH 3 THUẬT TOÁN ENSEMBLE (holdout CV) ===")
print(compare_table.to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(compare_table["model"], compare_table["F1_cv"], color=["#4C72B0", "#DD8452", "#55A868"])
ax.set_ylabel("F1-score (CV)")
ax.set_title("So sánh AdaBoost vs GradientBoosting vs RandomForest")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(f"{REPORTS}/so_sanh_ensemble.png", dpi=120)
plt.close()

# ---------- 8. Đánh giá trên tập test gốc (zero-day attacks) ----------
y_pred_zeroday = ada.predict(X_zeroday)
f1_zeroday = f1_score(y_zeroday, y_pred_zeroday)
acc_zeroday = accuracy_score(y_zeroday, y_pred_zeroday)
unseen_attacks = set(test_raw.loc[test_raw["y"] == 1, "attack_type"].unique())
print(f"\nF1 trên NSL-KDD test gốc (có tấn công lạ): {f1_zeroday:.4f} (so với CV: {f1_ada:.4f})")
print(f"Chênh lệch: {(f1_ada - f1_zeroday):.4f} — do các loại tấn công zero-day không có trong train")

# ---------- 9. Confusion matrix + alert fatigue ----------
cm = confusion_matrix(y_zeroday, y_pred_zeroday)
tn, fp, fn, tp = cm.ravel()
plt.figure(figsize=(4.5, 4))
plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xticks([0, 1], ["normal", "attack"])
plt.yticks([0, 1], ["normal", "attack"])
plt.xlabel("Dự đoán"); plt.ylabel("Thực tế")
plt.title("Ma trận nhầm lẫn — NSL-KDD test gốc")
plt.tight_layout()
plt.savefig(f"{REPORTS}/confusion_matrix.png", dpi=120)
plt.close()

fpr = fp / (fp + tn)

results = {
    "f1_dummy": f1_dummy, "f1_stump": f1_stump, "f1_ada_300": f1_ada,
    "fit_time_sec": fit_time,
    "noise_experiment": noise_table.to_dict(orient="records"),
    "ensemble_comparison": compare_table.to_dict(orient="records"),
    "f1_zeroday_test": f1_zeroday, "acc_zeroday_test": acc_zeroday,
    "cv_vs_test_gap": f1_ada - f1_zeroday,
    "confusion_matrix_zeroday": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "fpr": fpr,
}
with open(f"{REPORTS}/results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

joblib.dump(ada, "../models/adaboost.joblib")
print("\nĐã lưu model -> models/adaboost.joblib, báo cáo -> reports/")
