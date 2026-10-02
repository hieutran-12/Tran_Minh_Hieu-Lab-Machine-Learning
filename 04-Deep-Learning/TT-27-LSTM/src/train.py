"""So sánh baseline naive / XGBoost / SimpleRNN-LSTM-GRU để dự báo lưu lượng giao thông theo giờ.

Chạy từ thư mục gốc dự án:  python -m src.train [--csv đường_dẫn] [--epochs 40]
"""
import argparse
import os
import random
import re
import time
from dataclasses import dataclass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from tensorflow.keras import layers

tf.get_logger().setLevel("ERROR")

from .data import ROOT, build_features, clean, load_raw
from .sequences import fit_scaler, make_samples, split_cuts

REPORTS, MODELS = ROOT / "reports", ROOT / "models"
NAIVE, SEAS, XGB = "Naive (hôm qua)", "Seasonal naive (tuần trước)", "XGBoost + lag"
RNN, LSTM, GRU = "SimpleRNN", "LSTM", "GRU"
KINDS = {RNN: layers.SimpleRNN, LSTM: layers.LSTM, GRU: layers.GRU}
LAGS = [0, 1, 2, 3, 4, 5, 6, 9, 12, 18, 23]  # lag (giờ) tính lùi từ thời điểm dự báo


@dataclass
class Ctx:
    hourly: pd.DataFrame
    scaled: pd.DataFrame
    scaler: object
    cuts: tuple
    report: dict
    epochs: int = 40
    seed: int = 42


# ----------------------------------------------------------------------------- chuẩn bị
def prepare(csv_path=None, epochs=40, seed=42):
    raw = load_raw(csv_path)
    hourly, report = clean(raw)
    feats = build_features(hourly)
    cuts = split_cuts(hourly)
    scaled, scaler = fit_scaler(feats, hourly, cuts[0])
    return Ctx(hourly, scaled, scaler, cuts, report, epochs, seed)


def samples(ctx, seq_len, horizon):
    return make_samples(ctx.scaled, ctx.hourly, ctx.cuts, seq_len, horizon)


def set_seed(seed):
    random.seed(seed), np.random.seed(seed), tf.keras.utils.set_random_seed(seed)


def save_fig(fig, name):
    REPORTS.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(REPORTS / name, dpi=130)
    plt.close(fig)
    return REPORTS / name


def ve_eda(ctx):
    """EDA: lưu lượng trung bình theo giờ trong ngày, theo thứ, theo tháng (chỉ giờ có quan sát thật)."""
    h = ctx.hourly
    d = h[h["valid"] & (h["is_filled"] == 0)]["traffic_volume"]
    weekend = d.index.dayofweek >= 5
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    d[~weekend].groupby(d.index[~weekend].hour).mean().plot(ax=ax[0], label="Ngày thường")
    d[weekend].groupby(d.index[weekend].hour).mean().plot(ax=ax[0], label="Cuối tuần")
    ax[0].set(title="Lưu lượng TB theo giờ trong ngày", xlabel="Giờ", ylabel="xe/giờ"), ax[0].legend()
    d.groupby(d.index.dayofweek).mean().set_axis(["T2", "T3", "T4", "T5", "T6", "T7", "CN"]).plot.bar(ax=ax[1], rot=0)
    ax[1].set(title="Theo thứ trong tuần", xlabel="")
    d.groupby(d.index.month).mean().plot.bar(ax=ax[2], rot=0)
    ax[2].set(title="Theo tháng", xlabel="Tháng")
    return save_fig(fig, "eda_theo_gio.png")


# ----------------------------------------------------------------------------- baseline
def actual(ctx):
    return ctx.hourly["traffic_volume"]


def naive_pred(ctx, times, lag_hours):
    """Baseline: lấy giá trị cách thời điểm đích `lag_hours` giờ (24 = hôm qua, 168 = tuần trước)."""
    v = actual(ctx).reindex(times - pd.Timedelta(hours=lag_hours)).to_numpy()
    return pd.Series(v, index=times)


def lag_matrix(ctx, times, h):
    """Đặc trưng lag cho XGBoost, chỉ dùng thông tin biết tại thời điểm dự báo (h <= 24)."""
    assert h <= 24, "lag 'hôm qua' chỉ hợp lệ khi h <= 24"
    hourly = ctx.hourly
    vol = actual(ctx).to_numpy()
    pos = hourly.index.get_indexer(times)  # vị trí thời điểm đích T

    def at(p):
        out = np.full(len(p), np.nan)
        ok = p >= 0
        out[ok] = vol[p[ok]]
        return out

    cols = {f"lag_{k}": at(pos - h - k) for k in LAGS}
    cols["cung_gio_hom_qua"] = at(pos - 24)
    cols["cung_gio_tuan_truoc"] = at(pos - 168)
    for c in ["temp", "rain_1h", "clouds_all"]:
        cols[c] = hourly[c].to_numpy()[pos - h]
    cols["gio"] = np.asarray(times.hour)
    cols["thu"] = np.asarray(times.dayofweek)
    cols["is_holiday"] = hourly["is_holiday"].to_numpy()[pos]  # lịch ngày lễ biết trước
    return pd.DataFrame(cols)


def train_xgb(ctx, data, h):
    t0 = time.time()
    (_, _, ttr), (_, _, tva), (_, _, tte) = data["train"], data["val"], data["test"]
    model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8,
                             colsample_bytree=0.8, early_stopping_rounds=30, tree_method="hist",
                             random_state=ctx.seed)
    model.fit(lag_matrix(ctx, ttr, h), actual(ctx).reindex(ttr).to_numpy(),
              eval_set=[(lag_matrix(ctx, tva, h), actual(ctx).reindex(tva).to_numpy())], verbose=False)
    pred = pd.Series(model.predict(lag_matrix(ctx, tte, h)), index=tte)
    return pred, {"params": np.nan, "train_s": time.time() - t0, "epochs": model.get_booster().num_boosted_rounds()}


# ----------------------------------------------------------------------------- mạng nơ-ron
def build_model(kind, seq_len, n_features):
    """2 tầng hồi quy (64 -> 32) + Dropout + Dense(1) KHÔNG activation (hồi quy)."""
    cell = KINDS[kind]
    model = tf.keras.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        cell(64, return_sequences=True),
        layers.Dropout(0.2),
        cell(32),  # return_sequences=False ở tầng cuối
        layers.Dropout(0.2),
        layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def train_nn(kind, ctx, data, seq_len):
    set_seed(ctx.seed)
    (Xtr, ytr, _), (Xva, yva, _), (Xte, _, tte) = data["train"], data["val"], data["test"]
    model = build_model(kind, seq_len, Xtr.shape[2])
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                 tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)]
    t0 = time.time()
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=ctx.epochs, batch_size=256,
                     shuffle=True,  # shuffle các CỬA SỔ khi train là hợp lệ (khác với shuffle khi chia)
                     callbacks=callbacks, verbose=0)
    secs = time.time() - t0
    mean, std = ctx.scaler.mean_[0], ctx.scaler.scale_[0]
    y_hat = np.clip(model.predict(Xte, batch_size=1024, verbose=0).ravel() * std + mean, 0, None)
    return model, pd.Series(y_hat, index=tte), {"params": model.count_params(), "train_s": secs,
                                                "epochs": len(hist.history["loss"])}


# ----------------------------------------------------------------------------- đánh giá
def common_index(*preds):
    idx = preds[0].dropna().index
    for p in preds[1:]:
        idx = idx.intersection(p.dropna().index)
    return idx


def score(ctx, pred, idx):
    e = pred.reindex(idx) - actual(ctx).reindex(idx)
    return {"MAE": float(e.abs().mean()), "RMSE": float(np.sqrt((e ** 2).mean()))}


def run_compare(ctx, seq_len=24, h=1):
    """Bước 6-8: 2 baseline naive + XGBoost + SimpleRNN/LSTM/GRU (dự báo 1 giờ tới)."""
    data = samples(ctx, seq_len, h)
    t = data["test"][2]
    zero = {"params": np.nan, "train_s": 0.0, "epochs": 0}
    preds = {NAIVE: naive_pred(ctx, t, 24), SEAS: naive_pred(ctx, t, 168)}
    info = {NAIVE: zero, SEAS: zero}
    print(f"[{XGB}] đang huấn luyện ...")
    preds[XGB], info[XGB] = train_xgb(ctx, data, h)
    models = {}
    for kind in KINDS:
        print(f"[{kind}] đang huấn luyện ...")
        models[kind], preds[kind], info[kind] = train_nn(kind, ctx, data, seq_len)
        print(f"    {info[kind]['epochs']} epoch, {info[kind]['train_s']:.0f}s")
    idx = common_index(*preds.values())
    rows = [{"Mô hình": k, **score(ctx, p, idx), "Thời gian train (s)": info[k]["train_s"],
             "Số tham số": info[k]["params"], "Số epoch/cây": info[k]["epochs"]} for k, p in preds.items()]
    table = pd.DataFrame(rows)
    params = table["Số tham số"].copy()
    table["Số tham số"] = params.map(lambda v: "-" if pd.isna(v) else f"{int(v):,}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].bar(table["Mô hình"], table["MAE"], color=["#999", "#666", "#e69f00", "#56b4e9", "#009e73", "#cc79a7"])
    ax[0].set(title=f"MAE trên test ({len(idx)} giờ chung)", ylabel="xe/giờ")
    nn = table[table["Mô hình"].isin([XGB, RNN, LSTM, GRU])]
    ax[1].bar(nn["Mô hình"], nn["Thời gian train (s)"], color="#56b4e9"), ax[1].set(title="Thời gian train (s)")
    nn = table[table["Mô hình"].isin([RNN, LSTM, GRU])]
    ax[2].bar(nn["Mô hình"], params[nn.index], color="#009e73"), ax[2].set(title="Số tham số")
    for a in ax:
        a.tick_params(axis="x", rotation=25)
    save_fig(fig, "rnn_lstm_gru.png")
    return {"table": table, "preds": preds, "info": info, "models": models, "idx": idx, "data": data}


def run_window_sweep(ctx, cmp, windows=(6, 12, 24, 48), h=1):
    """Bước 9: khảo sát độ dài cửa sổ cho LSTM."""
    preds, info = {}, {}
    for w in windows:
        if w == 24:
            preds[w], info[w] = cmp["preds"][LSTM], cmp["info"][LSTM]
            continue
        print(f"[LSTM] cửa sổ {w}h ...")
        _, preds[w], info[w] = train_nn(LSTM, ctx, samples(ctx, w, h), w)
    idx = common_index(*preds.values(), cmp["preds"][SEAS])
    table = pd.DataFrame([{"Cửa sổ (giờ)": w, **score(ctx, p, idx), "Thời gian train (s)": info[w]["train_s"]}
                          for w, p in preds.items()])
    seas = score(ctx, cmp["preds"][SEAS], idx)["MAE"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(table["Cửa sổ (giờ)"], table["MAE"], "o-", label="LSTM")
    ax.axhline(seas, color="gray", ls="--", label="Seasonal naive")
    ax.set(title="MAE theo độ dài cửa sổ", xlabel="Số giờ quá khứ", ylabel="MAE (xe/giờ)", xticks=list(windows))
    ax.legend()
    save_fig(fig, "do_dai_cua_so.png")
    return {"table": table, "seasonal_mae": seas}


def run_horizons(ctx, cmp, horizons=(1, 3, 6), seq_len=24):
    """Bước 11: dự báo nhiều bước (mỗi mức h một mô hình LSTM + một XGBoost)."""
    P = {}
    for h in horizons:
        if h == 1:
            P[h] = {k: cmp["preds"][k] for k in (LSTM, XGB, SEAS, NAIVE)}
            continue
        print(f"[h={h}] LSTM + XGBoost ...")
        data = samples(ctx, seq_len, h)
        t = data["test"][2]
        P[h] = {LSTM: train_nn(LSTM, ctx, data, seq_len)[1], XGB: train_xgb(ctx, data, h)[0],
                SEAS: naive_pred(ctx, t, 168), NAIVE: naive_pred(ctx, t, 24)}
    idx = common_index(*[p for d in P.values() for p in d.values()])
    rows = []
    for h in horizons:
        r = {"Dự báo trước (giờ)": h}
        r.update({k: score(ctx, P[h][k], idx)["MAE"] for k in (LSTM, XGB, SEAS, NAIVE)})
        rows.append(r)
    table = pd.DataFrame(rows)
    table["LSTM tăng so với 1h (%)"] = (table[LSTM] / table[LSTM].iloc[0] - 1) * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    table.set_index("Dự báo trước (giờ)")[[LSTM, XGB, SEAS, NAIVE]].plot.bar(ax=ax, rot=0)
    ax.set(title=f"MAE theo số giờ dự báo trước ({len(idx)} giờ chung)", ylabel="MAE (xe/giờ)")
    save_fig(fig, "du_bao_nhieu_buoc.png")
    return {"table": table, "n": len(idx)}


def ve_tuan_cuoi(ctx, cmp):
    """Bước 10: dự báo vs thực tế trên 1 tuần cuối của tập test."""
    end = cmp["preds"][LSTM].index.max()
    rng = pd.date_range(end - pd.Timedelta(hours=167), end, freq="h")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(rng, actual(ctx).reindex(rng), "k", lw=2, label="Thực tế")
    ax.plot(rng, cmp["preds"][LSTM].reindex(rng), color="#009e73", label="LSTM (1h tới)")
    ax.plot(rng, cmp["preds"][SEAS].reindex(rng), color="gray", ls="--", label="Seasonal naive")
    ax.set(title="Dự báo vs thực tế — 1 tuần cuối của tập test", ylabel="xe/giờ"), ax.legend()
    return save_fig(fig, "du_bao_vs_thuc_te.png")


def phan_tich_loi(ctx, cmp):
    """Bước 12: LSTM sai nhiều nhất vào giờ nào, ngày nào?"""
    idx = cmp["idx"]
    err = (cmp["preds"][LSTM].reindex(idx) - actual(ctx).reindex(idx)).abs()
    by_hour = err.groupby(idx.hour).mean()
    by_dow = err.groupby(idx.dayofweek).mean().set_axis(["T2", "T3", "T4", "T5", "T6", "T7", "CN"])
    by_day = err.groupby(idx.normalize()).agg(["mean", "size"])
    worst_days = by_day[by_day["size"] >= 12]["mean"].nlargest(5)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    by_hour.plot.bar(ax=ax[0], rot=0), ax[0].set(title="MAE của LSTM theo giờ trong ngày", xlabel="Giờ")
    by_dow.plot.bar(ax=ax[1], rot=0), ax[1].set(title="MAE của LSTM theo thứ")
    save_fig(fig, "phan_tich_loi.png")
    return {"worst_hours": by_hour.nlargest(3), "worst_dow": by_dow.nlargest(2), "worst_days": worst_days}


# ----------------------------------------------------------------------------- báo cáo
def md_table(df):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "-" if np.isnan(v) else f"{v:,.1f}"
        return f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False))


def conclusions(cmp, sweep, hor):
    t = cmp["table"].set_index("Mô hình")
    mae, secs = t["MAE"], t["Thời gian train (s)"]
    out = []
    if mae[LSTM] < mae[SEAS]:
        out.append(f"LSTM **thắng** seasonal naive: MAE {mae[LSTM]:.0f} so với {mae[SEAS]:.0f} xe/giờ "
                   f"(giảm {(1 - mae[LSTM] / mae[SEAS]) * 100:.1f}%).")
    else:
        out.append(f"LSTM **không thắng** seasonal naive (MAE {mae[LSTM]:.0f} vs {mae[SEAS]:.0f} xe/giờ) → "
                   "kết luận trung thực: bài này không cần deep learning.")
    ratio = secs[LSTM] / max(secs[XGB], 1e-3)
    if mae[XGB] <= mae[LSTM] * 1.05:
        out.append(f"XGBoost + lag features cho MAE {mae[XGB]:.0f} (LSTM: {mae[LSTM]:.0f}) và train nhanh hơn "
                   f"~{ratio:.0f} lần → nên ưu tiên XGBoost khi triển khai.")
    else:
        out.append(f"LSTM (MAE {mae[LSTM]:.0f}) tốt hơn XGBoost (MAE {mae[XGB]:.0f}) "
                   f"{(1 - mae[LSTM] / mae[XGB]) * 100:.1f}%, đổi lại train chậm hơn ~{ratio:.0f} lần.")
    nn = mae[[RNN, LSTM, GRU]]
    out.append(f"Trong 3 kiến trúc hồi quy, **{nn.idxmin()}** có MAE thấp nhất ({nn.min():.0f}); "
               f"SimpleRNN {mae[RNN]:.0f} · LSTM {mae[LSTM]:.0f} · GRU {mae[GRU]:.0f}.")
    best_w = sweep["table"].loc[sweep["table"]["MAE"].idxmin()]
    out.append(f"Cửa sổ tốt nhất: **{int(best_w['Cửa sổ (giờ)'])} giờ** (MAE {best_w['MAE']:.0f}).")
    last = hor["table"].iloc[-1]
    inc, far = last["LSTM tăng so với 1h (%)"], int(last["Dự báo trước (giờ)"])
    out.append(f"LSTM dự báo {far}h tới có MAE " + (f"tăng {inc:.0f}%" if inc > 0 else f"không tăng ({inc:.0f}%)")
               + " so với 1h tới.")
    out.append("Mức tham chiếu của đề bài cho 1 giờ tới: MAE ~250–400 xe/giờ.")
    return out


def build_markdown(ctx, cmp, sweep, hor, err, img=""):
    rep = "\n".join(f"- {k}: {v}" for k, v in ctx.report.items())
    days = ", ".join(f"{d:%Y-%m-%d} ({v:.0f})" for d, v in err["worst_days"].items())
    hours = ", ".join(f"{h}h ({v:.0f})" for h, v in err["worst_hours"].items())
    dows = ", ".join(f"{d} ({v:.0f})" for d, v in err["worst_dow"].items())
    return f"""### Xử lý dữ liệu
{rep}

### So sánh với baseline (dự báo 1 giờ tới, MAE/RMSE đơn vị xe/giờ)
{md_table(cmp['table'])}

![so sánh]({img}rnn_lstm_gru.png)

### Khảo sát độ dài cửa sổ (LSTM)
{md_table(sweep['table'])}

![cửa sổ]({img}do_dai_cua_so.png)

### Dự báo nhiều bước
{md_table(hor['table'])}

![nhiều bước]({img}du_bao_nhieu_buoc.png)

### Dự báo vs thực tế (1 tuần cuối tập test)
![tuần cuối]({img}du_bao_vs_thuc_te.png)

### Phân tích lỗi (LSTM, 1h tới)
- Giờ sai nhiều nhất: {hours}
- Thứ sai nhiều nhất: {dows}
- Ngày sai nhiều nhất: {days}

![lỗi]({img}phan_tich_loi.png)

### Kết luận
""" + "\n".join(f"- {c}" for c in conclusions(cmp, sweep, hor))


def save_outputs(ctx, cmp, sweep, hor, err):
    """Lưu mô hình LSTM, scaler, báo cáo kết quả và điền bảng kết quả vào README."""
    MODELS.mkdir(exist_ok=True)
    cmp["models"][LSTM].save(MODELS / "lstm_traffic.keras")
    joblib.dump(ctx.scaler, MODELS / "scaler.joblib")
    (REPORTS / "ket_qua.md").write_text(build_markdown(ctx, cmp, sweep, hor, err), encoding="utf-8")
    readme = ROOT / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        body = build_markdown(ctx, cmp, sweep, hor, err, img="reports/")
        text = re.sub(r"(<!-- KET_QUA_START -->).*?(<!-- KET_QUA_END -->)",
                      lambda m: f"{m.group(1)}\n{body}\n{m.group(2)}", text, flags=re.S)
        readme.write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", help="đường dẫn file csv/csv.gz (mặc định: tự tải từ UCI)")
    ap.add_argument("--epochs", type=int, default=40, help="số epoch tối đa (có early stopping)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ctx = prepare(args.csv, args.epochs, args.seed)
    for k, v in ctx.report.items():
        print(f"{k}: {v}")
    print(f"Mốc chia thời gian: val từ {ctx.cuts[0]}, test từ {ctx.cuts[1]}")
    ve_eda(ctx)
    cmp = run_compare(ctx)
    print(md_table(cmp["table"]))
    sweep = run_window_sweep(ctx, cmp)
    hor = run_horizons(ctx, cmp)
    ve_tuan_cuoi(ctx, cmp)
    err = phan_tich_loi(ctx, cmp)
    save_outputs(ctx, cmp, sweep, hor, err)
    print(f"Xong. Xem {REPORTS / 'ket_qua.md'}")


if __name__ == "__main__":
    main()
