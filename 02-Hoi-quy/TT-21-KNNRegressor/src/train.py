"""TT-21 — KNN Regressor trên California Housing.

Chạy toàn bộ 11 bước của đề bài, xuất biểu đồ vào reports/, model vào models/
và bảng kết quả vào reports/ket_qua.md (đồng thời chèn vào README.md).

    python src/train.py            # chạy đầy đủ
    python src/train.py --nhanh    # bỏ bớt thí nghiệm đo thời gian 10x
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import r2_score, root_mean_squared_error  # noqa: E402
from sklearn.model_selection import KFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

try:  # chạy được cả `python src/train.py` lẫn `python -m src.train`
    from .data import (DAC_TRUNG, RANDOM_STATE, cac_can_tuong_tu, chia_du_lieu,
                       tai_du_lieu, tao_pipeline)
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.data import (DAC_TRUNG, RANDOM_STATE, cac_can_tuong_tu, chia_du_lieu,
                          tai_du_lieu, tao_pipeline)

GOC = Path(__file__).resolve().parent.parent
THU_MUC_BC = GOC / "reports"
THU_MUC_MODEL = GOC / "models"
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.autolayout": True})


# ----------------------------------------------------------------------------- tiện ích
def danh_gia(model, X_tr, y_tr, X_te, y_te, ten):
    """Fit, đo RMSE/R2 và thời gian train + thời gian dự đoán 1 căn."""
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    t_train = time.perf_counter() - t0

    du_doan_test = model.predict(X_te)
    mot_can = X_te.iloc[[0]]
    model.predict(mot_can)  # làm nóng
    t0 = time.perf_counter()
    for _ in range(20):
        model.predict(mot_can)
    t_predict1 = (time.perf_counter() - t0) / 20

    return {
        "Mô hình": ten,
        "RMSE_train": root_mean_squared_error(y_tr, model.predict(X_tr)),
        "RMSE_test": root_mean_squared_error(y_te, du_doan_test),
        "R2_test": r2_score(y_te, du_doan_test),
        "Thời gian train (s)": t_train,
        "Dự đoán 1 căn (ms)": t_predict1 * 1000,
    }


def in_bang(df, tieu_de):
    print(f"\n{'=' * 78}\n{tieu_de}\n{'=' * 78}")
    print(df.to_string(index=False))
    return df


def lam_tron(df):
    return df.round({c: 4 for c in df.select_dtypes("number").columns})


# ----------------------------------------------------------------------------- các bước
def buoc_1_du_lieu():
    tho = tai_du_lieu(loc_outlier=False)
    df = tai_du_lieu(loc_outlier=True)
    print(f"Dữ liệu gốc : {tho.shape[0]:,} dòng × {len(DAC_TRUNG)} đặc trưng")
    print(f"Sau lọc outlier (AveRooms>15, AveBedrms>4, AveOccup>10): "
          f"{df.shape[0]:,} dòng (bỏ {tho.shape[0] - df.shape[0]:,})")
    print(f"Tỉ lệ nhãn bị chặn trần 5.0: {(df['MedHouseVal'] >= 5.0).mean():.2%}")
    return df


def buoc_2_3_4_baseline(X_tr, y_tr, X_te, y_te):
    """Baseline + minh chứng bắt buộc phải chuẩn hoá."""
    ket_qua = [
        danh_gia(DummyRegressor(strategy="mean"), X_tr, y_tr, X_te, y_te, "Dummy (giá TB)"),
        danh_gia(Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())]),
                 X_tr, y_tr, X_te, y_te, "Linear Regression (TT-11)"),
        danh_gia(tao_pipeline(k=5, weights="uniform", chuan_hoa=False),
                 X_tr, y_tr, X_te, y_te, "KNN K=5 — KHÔNG chuẩn hoá"),
        danh_gia(tao_pipeline(k=5, weights="uniform", chuan_hoa=True),
                 X_tr, y_tr, X_te, y_te, "KNN K=5 — CÓ chuẩn hoá"),
    ]
    return in_bang(lam_tron(pd.DataFrame(ket_qua)),
                   "BƯỚC 2–4 · BASELINE VÀ TÁC ĐỘNG CỦA CHUẨN HOÁ")


def buoc_5_duong_cong_K(X_tr, y_tr, X_te, y_te, danh_sach_K=range(1, 51)):
    """RMSE train/validation/test theo K cho cả 2 kiểu weights.

    K được chọn bằng tập validation tách từ train — KHÔNG dùng test để chọn.
    """
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=RANDOM_STATE)

    dong = []
    for weights in ("uniform", "distance"):
        for k in danh_sach_K:
            pipe = tao_pipeline(k=k, weights=weights).fit(X_fit, y_fit)
            dong.append({
                "weights": weights, "K": k,
                "RMSE_train": root_mean_squared_error(y_fit, pipe.predict(X_fit)),
                "RMSE_val": root_mean_squared_error(y_val, pipe.predict(X_val)),
                "RMSE_test": root_mean_squared_error(y_te, pipe.predict(X_te)),
            })
    bang = pd.DataFrame(dong)
    tot = bang.loc[bang.RMSE_val.idxmin()]
    k_tot, weights_tot = int(tot.K), tot.weights

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, weights in zip(axes, ("uniform", "distance")):
        phan = bang[bang.weights == weights]
        ax.plot(phan.K, phan.RMSE_train, "o-", ms=3, label="RMSE train")
        ax.plot(phan.K, phan.RMSE_val, "s-", ms=3, label="RMSE validation")
        ax.plot(phan.K, phan.RMSE_test, "^--", ms=3, alpha=.7, label="RMSE test")
        k_w = int(phan.loc[phan.RMSE_val.idxmin(), "K"])
        ax.axvline(k_w, color="crimson", ls=":", label=f"K tốt nhất = {k_w}")
        ax.set(xlabel="K (số hàng xóm)", title=f"weights='{weights}'")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("RMSE (100.000 USD)")
    axes[0].annotate("K=1 → RMSE train = 0:\nhàng xóm gần nhất của một điểm train\nchính là nó",
                     xy=(1, 0), xytext=(8, axes[0].get_ylim()[1] * .45), fontsize=8.5,
                     color="gray", arrowprops=dict(arrowstyle="->", color="gray"))
    axes[1].text(.35, .12, "weights='distance' → RMSE train = 0 với MỌI K\n"
                           "(khoảng cách 0 ⇒ trọng số vô hạn cho chính nó)",
                 transform=axes[1].transAxes, fontsize=8.5, color="gray")
    fig.suptitle("Bước 5 · RMSE theo K — chọn K bằng validation, không bằng train")
    fig.savefig(THU_MUC_BC / "rmse_theo_K.png")
    plt.close(fig)
    rmse_train_k1 = bang.query(
    "weights == 'uniform' and K == 1"
    ).RMSE_train.iloc[0]

    print(f"\nBƯỚC 5 · K tốt nhất theo validation = {k_tot} (weights='{weights_tot}', "
        f"RMSE_val={bang.RMSE_val.min():.4f}). "
        f"RMSE train tại K=1/uniform = "
        f"{rmse_train_k1:.6f}")

    return bang, k_tot


def buoc_6_7_weights_metric(X_tr, y_tr, X_te, y_te, k):
    dong = []
    for weights in ("uniform", "distance"):
        for metric in ("euclidean", "manhattan"):
            kq = danh_gia(tao_pipeline(k=k, weights=weights, metric=metric),
                          X_tr, y_tr, X_te, y_te, f"{weights} · {metric}")
            dong.append(kq)
    bang = lam_tron(pd.DataFrame(dong))
    in_bang(bang, f"BƯỚC 6–7 · WEIGHTS VÀ METRIC (K={k})")
    tot = bang.loc[bang.RMSE_test.idxmin(), "Mô hình"].split(" · ")
    return bang, tot[0], tot[1]


def buoc_8_trong_so_vi_tri(X_tr, y_tr, X_te, y_te, k, weights, metric,
                           he_so=(1, 2, 3, 5)):
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=RANDOM_STATE)
    dong = []
    for w in he_so:
        pipe = tao_pipeline(k=k, weights=weights, metric=metric, w_vi_tri=w)
        pipe.fit(X_fit, y_fit)
        dong.append({"Hệ số Lat/Lon": w,
                     "RMSE_val": root_mean_squared_error(y_val, pipe.predict(X_val)),
                     "RMSE_test": root_mean_squared_error(y_te, pipe.predict(X_te)),
                     "R2_test": r2_score(y_te, pipe.predict(X_te))})
    bang = lam_tron(pd.DataFrame(dong))
    in_bang(bang, "BƯỚC 8 · THÍ NGHIỆM TRỌNG SỐ VỊ TRÍ")
    w_tot = float(bang.loc[bang.RMSE_val.idxmin(), "Hệ số Lat/Lon"])

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(bang["Hệ số Lat/Lon"], bang.RMSE_val, "o-", label="RMSE validation")
    ax.plot(bang["Hệ số Lat/Lon"], bang.RMSE_test, "s--", label="RMSE test")
    ax.axvline(w_tot, color="crimson", ls=":", label=f"hệ số tốt nhất = {w_tot:g}")
    ax.set(xlabel="Hệ số nhân cho Latitude/Longitude (sau chuẩn hoá)",
           ylabel="RMSE (100.000 USD)",
           title="Vị trí quan trọng đến mức nào?")
    ax.legend()
    fig.savefig(THU_MUC_BC / "trong_so_vi_tri.png")
    plt.close(fig)
    return bang, w_tot


def buoc_9_can_tuong_tu(pipe, X_tr, y_tr, X_te, y_te, so_can=3, k=5):
    print(f"\n{'=' * 78}\nBƯỚC 9 · CĂN CỨ DỰ ĐOÁN — {k} CĂN TƯƠNG TỰ\n{'=' * 78}")
    chon = X_te.sample(so_can, random_state=RANDOM_STATE).index
    fig, axes = plt.subplots(1, so_can, figsize=(4.4 * so_can, 4.2))
    ghi_chu = []

    for ax, i in zip(np.atleast_1d(axes), chon):
        x_q = X_te.loc[[i]]
        bang, du_doan = cac_can_tuong_tu(pipe, X_tr, y_tr, x_q, k=k)
        thuc = y_te.loc[i]
        print(f"\n— Căn #{i}: giá thực {thuc:.2f} · KNN ước tính {du_doan:.2f} "
              f"(100.000 USD)")
        print(x_q.round(2).to_string(index=False))
        print(f"  {k} căn tương tự KNN đã dùng:")
        print(bang.round(3).to_string(index=False))
        ghi_chu.append({"Căn": int(i), "Giá thực": round(float(thuc), 3),
                        "KNN ước tính": round(du_doan, 3),
                        "Giá các căn tương tự": [round(v, 2) for v in bang.Gia_thuc]})

        ax.scatter(X_tr.Longitude, X_tr.Latitude, s=2, c="lightgray", alpha=.5)
        ax.scatter(bang.Longitude, bang.Latitude, s=70, c="tab:blue",
                   label=f"{k} căn tương tự")
        ax.scatter(x_q.Longitude, x_q.Latitude, s=140, marker="*", c="crimson",
                   label="căn cần định giá")
        r = 0.6
        ax.set(xlim=(x_q.Longitude.iloc[0] - r, x_q.Longitude.iloc[0] + r),
               ylim=(x_q.Latitude.iloc[0] - r, x_q.Latitude.iloc[0] + r),
               xlabel="Longitude", ylabel="Latitude",
               title=f"Căn #{i} · thực {thuc:.2f} · dự đoán {du_doan:.2f}")
        ax.legend(fontsize=8, loc="upper right")

    fig.savefig(THU_MUC_BC / "can_tuong_tu_vi_du.png")
    plt.close(fig)
    return ghi_chu


def buoc_10_so_sanh(X_tr, y_tr, X_te, y_te, knn_pipe):
    dong = [
        danh_gia(Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())]),
                 X_tr, y_tr, X_te, y_te, "Linear Regression (TT-11)"),
        danh_gia(knn_pipe, X_tr, y_tr, X_te, y_te, "KNN Regressor (TT-21)"),
        danh_gia(RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE,
                                       n_jobs=-1),
                 X_tr, y_tr, X_te, y_te, "Random Forest"),
    ]
    bang = pd.DataFrame(dong)
    bang["Giải thích được?"] = ["Có — hệ số từng đặc trưng",
                                "Có — chỉ ra ĐÚNG các căn tương tự",
                                "Khó — 200 cây, chỉ có feature importance"]
    return in_bang(lam_tron(bang), "BƯỚC 10 · SO SÁNH 3 THUẬT TOÁN")


def buoc_11_thoi_gian(X_tr, y_tr, X_te, knn_pipe, he_so=(1, 5, 10)):
    """Nhân bản tập train (kèm nhiễu nhỏ) để đo độ chậm khi dữ liệu lớn dần."""
    rng = np.random.default_rng(RANDOM_STATE)
    lo = X_te.iloc[:100]
    dong = []
    for h in he_so:
        Xb = pd.concat([X_tr] + [X_tr * (1 + rng.normal(0, .01, X_tr.shape))
                                 for _ in range(h - 1)], ignore_index=True)
        yb = pd.concat([y_tr] * h, ignore_index=True)
        pipe = tao_pipeline(**knn_pipe_cau_hinh(knn_pipe)).fit(Xb, yb)

        pipe.predict(lo.iloc[[0]])
        t0 = time.perf_counter()
        for _ in range(20):
            pipe.predict(lo.iloc[[0]])
        t1 = (time.perf_counter() - t0) / 20 * 1000

        t0 = time.perf_counter()
        pipe.predict(lo)
        t100 = (time.perf_counter() - t0) * 1000

        dong.append({"Bội số dữ liệu": f"{h}×", "Số dòng train": len(Xb),
                     "Dự đoán 1 căn (ms)": t1, "Dự đoán 100 căn (ms)": t100})
    bang = lam_tron(pd.DataFrame(dong))
    in_bang(bang, "BƯỚC 11 · THỜI GIAN DỰ ĐOÁN KHI DỮ LIỆU LỚN DẦN")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(bang["Số dòng train"], bang["Dự đoán 100 căn (ms)"], "o-")
    for _, r in bang.iterrows():
        ax.annotate(r["Bội số dữ liệu"], (r["Số dòng train"], r["Dự đoán 100 căn (ms)"]),
                    textcoords="offset points", xytext=(6, -10))
    ax.set(xlabel="Số dòng trong tập train", ylabel="Thời gian dự đoán 100 căn (ms)",
           title="KNN chậm dần tuyến tính theo kích thước dữ liệu")
    fig.savefig(THU_MUC_BC / "thoi_gian_predict.png")
    plt.close(fig)
    return bang


def knn_pipe_cau_hinh(pipe):
    """Trích cấu hình KNN từ một pipeline đã dựng để tái tạo pipeline mới."""
    knn = pipe.named_steps["knn"]
    cau_hinh = {"k": knn.n_neighbors, "weights": knn.weights, "metric": knn.metric}
    if "vitri" in pipe.named_steps:
        cau_hinh["w_vi_tri"] = pipe.named_steps["vitri"].w
    return cau_hinh


def ghi_ket_qua(cac_bang, duong_dan_md):
    """Ghi reports/ket_qua.md và chèn vào README.md giữa 2 mốc đánh dấu."""
    khoi = ["<!-- KET-QUA:BEGIN --> <!-- Tự sinh bởi src/train.py — đừng sửa tay -->",
            f"\n> Sinh tự động lúc {time.strftime('%Y-%m-%d %H:%M')}\n"]
    for tieu_de, bang in cac_bang:
        khoi.append(f"\n**{tieu_de}**\n")
        khoi.append(bang.to_markdown(index=False))
    khoi.append("\n<!-- KET-QUA:END -->")
    noi_dung = "\n".join(khoi)
    duong_dan_md.write_text(noi_dung, encoding="utf-8")

    readme = GOC / "README.md"
    if readme.exists():
        van_ban = readme.read_text(encoding="utf-8")
        if "<!-- KET-QUA:BEGIN -->" in van_ban and "<!-- KET-QUA:END -->" in van_ban:
            dau = van_ban.index("<!-- KET-QUA:BEGIN -->")
            cuoi = van_ban.index("<!-- KET-QUA:END -->") + len("<!-- KET-QUA:END -->")
            readme.write_text(van_ban[:dau] + noi_dung + van_ban[cuoi:], encoding="utf-8")
            print("\n→ Đã cập nhật bảng kết quả trong README.md")


# ----------------------------------------------------------------------------- main
def main(nhanh=False):
    THU_MUC_BC.mkdir(exist_ok=True)
    THU_MUC_MODEL.mkdir(exist_ok=True)

    df = buoc_1_du_lieu()
    X_tr, X_te, y_tr, y_te = chia_du_lieu(df)
    print(f"Train: {len(X_tr):,} · Test: {len(X_te):,}")

    b_baseline = buoc_2_3_4_baseline(X_tr, y_tr, X_te, y_te)
    danh_sach_K = range(1, 51) if not nhanh else [1, 3, 5, 10, 15, 20, 30, 50]
    b_K, k_tot = buoc_5_duong_cong_K(X_tr, y_tr, X_te, y_te, danh_sach_K)
    b_wm, weights, metric = buoc_6_7_weights_metric(X_tr, y_tr, X_te, y_te, k_tot)
    b_vitri, w_tot = buoc_8_trong_so_vi_tri(X_tr, y_tr, X_te, y_te, k_tot, weights, metric)

    pipe = tao_pipeline(k=k_tot, weights=weights, metric=metric, w_vi_tri=w_tot)
    pipe.fit(X_tr, y_tr)
    print(f"\nCấu hình chốt: K={k_tot}, weights={weights}, metric={metric}, "
          f"hệ số vị trí={w_tot:g}")

    # Kiểm chứng bằng cross-validation 5-fold trên tập train
    cv = -cross_val_score(pipe, X_tr, y_tr, cv=KFold(5, shuffle=True,
                                                     random_state=RANDOM_STATE),
                          scoring="neg_root_mean_squared_error", n_jobs=-1)
    print(f"RMSE 5-fold CV trên train: {cv.mean():.4f} ± {cv.std():.4f}")

    vi_du = buoc_9_can_tuong_tu(pipe, X_tr, y_tr, X_te, y_te)
    b_cuoi = buoc_10_so_sanh(X_tr, y_tr, X_te, y_te, pipe)
    b_tg = buoc_11_thoi_gian(X_tr, y_tr, X_te, pipe, (1, 5) if nhanh else (1, 5, 10))

    joblib.dump(pipe, THU_MUC_MODEL / "knn_pipeline.joblib")
    (THU_MUC_MODEL / "cau_hinh.json").write_text(json.dumps(
        {"K": k_tot, "weights": weights, "metric": metric, "he_so_vi_tri": w_tot,
         "rmse_cv_train": round(float(cv.mean()), 4),
         "rmse_test": round(float(root_mean_squared_error(y_te, pipe.predict(X_te))), 4),
         "r2_test": round(float(r2_score(y_te, pipe.predict(X_te))), 4),
         "vi_du_can_tuong_tu": vi_du}, ensure_ascii=False, indent=2), encoding="utf-8")

    ghi_ket_qua([("Baseline & chuẩn hoá", b_baseline),
                 ("Weights × metric", b_wm),
                 ("Trọng số vị trí", b_vitri),
                 ("So sánh 3 thuật toán", b_cuoi),
                 ("Thời gian dự đoán theo kích thước dữ liệu", b_tg)],
                THU_MUC_BC / "ket_qua.md")

    b_K.to_csv(THU_MUC_BC / "rmse_theo_K.csv", index=False)
    print(f"\nXong. Model: models/knn_pipeline.joblib · Biểu đồ: reports/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nhanh", action="store_true", help="chạy rút gọn để kiểm tra")
    main(**vars(ap.parse_args()))
