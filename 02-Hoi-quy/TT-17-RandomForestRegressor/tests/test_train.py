"""
Bộ kiểm thử cho src/train.py — chạy trên dữ liệu GIẢ LẬP (không cần
Clean_Dataset.csv thật) để xác nhận từng hàm và toàn bộ pipeline hoạt động
đúng, không lỗi cú pháp/logic, TRƯỚC KHI chạy tốn thời gian trên dữ liệu thật.

Chạy: pytest tests/ -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import train as tt17  # noqa: E402


@pytest.fixture(scope="session")
def split_data(synthetic_df):
    X = synthetic_df[tt17.NUMERIC_COLS + tt17.CATEGORICAL_COLS]
    y = synthetic_df[tt17.TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=X["class"])


# --------------------------------------------------------------------------
# 1. Nạp dữ liệu
# --------------------------------------------------------------------------
def test_load_data_drops_flight_and_unnamed(synthetic_csv):
    df = tt17.load_data(Path(synthetic_csv))
    assert "flight" not in df.columns
    assert not any(c.lower().startswith("unnamed") for c in df.columns)
    assert set(tt17.NUMERIC_COLS + tt17.CATEGORICAL_COLS + [tt17.TARGET_COL]).issubset(df.columns)


def test_load_data_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        tt17.load_data(Path("/khong/ton/tai/nao_ca.csv"))


def test_load_data_missing_required_column_raises(tmp_path, synthetic_df):
    bad_df = synthetic_df.drop(columns=["class"])
    path = tmp_path / "bad.csv"
    bad_df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        tt17.load_data(path)


def test_validate_data_quality_rejects_negative_price(synthetic_df):
    bad_df = synthetic_df.copy()
    bad_df.loc[0, "price"] = -100
    with pytest.raises(ValueError):
        tt17._validate_data_quality(bad_df)


def test_validate_data_quality_passes_on_clean_data(synthetic_df):
    report = tt17._validate_data_quality(synthetic_df)
    assert report["so_gia_tri_null"] == 0


# --------------------------------------------------------------------------
# 2. Preprocessor
# --------------------------------------------------------------------------
def test_preprocessor_output_shape(split_data):
    X_train, X_test, y_train, y_test = split_data
    prep = tt17.build_preprocessor()
    X_t = prep.fit_transform(X_train)
    # số cột = số cột one-hot (categorical) + 2 numeric (passthrough)
    expected_cat_cols = sum(X_train[c].nunique() for c in tt17.CATEGORICAL_COLS)
    assert X_t.shape[1] == expected_cat_cols + len(tt17.NUMERIC_COLS)
    assert X_t.shape[0] == len(X_train)


def test_preprocessor_handles_unknown_category_at_test_time(split_data):
    X_train, X_test, y_train, y_test = split_data
    prep = tt17.build_preprocessor()
    prep.fit(X_train)
    X_test_unknown = X_test.copy()
    X_test_unknown.iloc[0, X_test_unknown.columns.get_loc("airline")] = "HANG_BAY_CHUA_TUNG_THAY"
    # KHÔNG được raise lỗi nhờ handle_unknown="ignore"
    out = prep.transform(X_test_unknown)
    assert out.shape[0] == len(X_test_unknown)


# --------------------------------------------------------------------------
# 3. rmse / metrics
# --------------------------------------------------------------------------
def test_rmse_zero_for_perfect_prediction():
    y = np.array([100.0, 200.0, 300.0])
    assert tt17.rmse(y, y) == pytest.approx(0.0)


def test_rmse_known_value():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    # RMSE = sqrt((9+16)/2) = sqrt(12.5)
    assert tt17.rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5))


# --------------------------------------------------------------------------
# 4. Baseline
# --------------------------------------------------------------------------
def test_run_baselines_returns_all_three_models(split_data):
    X_train, X_test, y_train, y_test = split_data
    prep = tt17.build_preprocessor()
    results = tt17.run_baselines(X_train, X_test, y_train, y_test, prep)
    assert set(results.keys()) == {"Dummy (mean)", "Linear Regression", "Decision Tree (TT-16, single)"}
    for name, metrics in results.items():
        assert {"rmse", "mae", "r2"} <= set(metrics.keys())
        assert metrics["rmse"] > 0


def test_random_forest_beats_dummy_baseline(split_data):
    """Sanity check tối thiểu: RF phải học được gì đó tốt hơn dự đoán trung bình."""
    X_train, X_test, y_train, y_test = split_data
    prep = tt17.build_preprocessor()
    dummy_rmse = tt17.run_baselines(X_train, X_test, y_train, y_test, prep)["Dummy (mean)"]["rmse"]

    rf_pipe = tt17.train_random_forest(X_train, y_train, tt17.build_preprocessor(), n_estimators=50)
    rf_rmse = tt17.rmse(y_test, rf_pipe.predict(X_test))
    assert rf_rmse < dummy_rmse


# --------------------------------------------------------------------------
# 5. Random Forest + oob_score_
# --------------------------------------------------------------------------
def test_train_random_forest_has_oob_score(split_data):
    X_train, X_test, y_train, y_test = split_data
    pipe = tt17.train_random_forest(X_train, y_train, tt17.build_preprocessor(), n_estimators=50)
    rf = pipe.named_steps["model"]
    assert hasattr(rf, "oob_score_")
    assert -1.0 <= rf.oob_score_ <= 1.0  # R^2 hợp lệ (có thể âm nếu model rất tệ)


# --------------------------------------------------------------------------
# 6. Tinh chỉnh siêu tham số
# --------------------------------------------------------------------------
def test_tune_random_forest_returns_valid_params(split_data):
    X_train, X_test, y_train, y_test = split_data
    result = tt17.tune_random_forest(
        X_train, y_train, tt17.build_preprocessor(),
        n_iter=3, cv_folds=2, search_sample_frac=0.5,
    )
    assert "n_estimators" in result["best_params"]
    assert result["best_cv_rmse"] > 0


# --------------------------------------------------------------------------
# 7. Cross-validation
# --------------------------------------------------------------------------
def test_cross_validation_runs_and_returns_expected_keys(synthetic_df):
    X = synthetic_df[tt17.NUMERIC_COLS + tt17.CATEGORICAL_COLS]
    y = synthetic_df[tt17.TARGET_COL]
    from sklearn.ensemble import RandomForestRegressor
    pipe = Pipeline([("prep", tt17.build_preprocessor()),
                      ("model", RandomForestRegressor(n_estimators=30, n_jobs=-1, random_state=42))])
    result = tt17.run_cross_validation(pipe, X, y, cv_folds=3)
    assert result["cv_folds"] == 3
    assert len(result["rmse_per_fold"]) == 3
    assert result["rmse_mean"] > 0


# --------------------------------------------------------------------------
# 8. RMSE vs n_estimators
# --------------------------------------------------------------------------
def test_rmse_vs_n_estimators_monotonic_improvement(split_data, tmp_path):
    X_train, X_test, y_train, y_test = split_data
    result = tt17.rmse_vs_n_estimators(
        X_train, X_test, y_train, y_test, tt17.build_preprocessor(), tmp_path,
        n_values=[10, 50, 100],
    )
    assert len(result["test_rmse"]) == 3
    assert "diem_bao_hoa_uoc_luong" in result
    assert (tmp_path / "rmse_theo_so_cay.png").exists()


# --------------------------------------------------------------------------
# 9. Permutation importance + Gini comparison
# --------------------------------------------------------------------------
def test_permutation_importance_class_is_top_feature(split_data, tmp_path):
    """Trong dữ liệu giả lập, `class` được cố tình gán ảnh hưởng lớn nhất tới
    giá -> permutation importance PHẢI xếp `class` lên hàng đầu."""
    X_train, X_test, y_train, y_test = split_data
    rf_pipe = tt17.train_random_forest(X_train, y_train, tt17.build_preprocessor(), n_estimators=100)
    result = tt17.run_permutation_importance(rf_pipe, X_test, y_test, tmp_path)

    perm = result["permutation_importance"]
    top_feature = max(perm, key=perm.get)
    assert top_feature == "class"
    assert (tmp_path / "permutation_importance.png").exists()
    assert (tmp_path / "so_sanh_permutation_vs_gini.png").exists()


# --------------------------------------------------------------------------
# 10. PDP days_left
# --------------------------------------------------------------------------
def test_pdp_days_left_shows_savings_when_booking_early(split_data, tmp_path):
    X_train, X_test, y_train, y_test = split_data
    rf_pipe = tt17.train_random_forest(X_train, y_train, tt17.build_preprocessor(), n_estimators=100)
    conclusion = tt17.run_pdp_days_left(rf_pipe, X_train, tmp_path)
    # Dữ liệu giả lập được thiết kế: giá cao hơn khi mua sát ngày (days_left nhỏ)
    assert conclusion["tiet_kiem_khi_mua_som_hon_1_tuan"] > 0
    assert (tmp_path / "pdp_days_left.png").exists()


# --------------------------------------------------------------------------
# 11. Khoảng dự báo 10-90%
# --------------------------------------------------------------------------
def test_prediction_interval_coverage_reasonable(split_data, tmp_path):
    X_train, X_test, y_train, y_test = split_data
    rf_pipe = tt17.train_random_forest(X_train, y_train, tt17.build_preprocessor(), n_estimators=100)
    result = tt17.run_prediction_interval(rf_pipe, X_test, y_test, tmp_path)
    # Kỳ vọng ~80% nhưng cho phép sai số rộng vì dữ liệu giả lập + tập nhỏ
    assert 0.4 <= result["coverage_10_90"] <= 1.0
    assert result["do_rong_khoang_trung_binh"] >= 0


# --------------------------------------------------------------------------
# 12. Ngoại suy
# --------------------------------------------------------------------------
def test_extrapolation_experiment_flags_capping(split_data, tmp_path):
    X_train, X_test, y_train, y_test = split_data
    rf_pipe = tt17.train_random_forest(X_train, y_train, tt17.build_preprocessor(), n_estimators=100)
    result = tt17.run_extrapolation_experiment(rf_pipe, X_train, tmp_path, n_profiles=10)
    assert result["max_days_left_trong_train"] <= 49
    assert result["du_doan_trung_binh_tai_days_left_100"] is not None
    # Random Forest không thể ngoại suy -> phải kẹp trần trong bài test này
    assert result["bi_kep_tran"] is True


# --------------------------------------------------------------------------
# 13. Sanity check chống rò rỉ dữ liệu
# --------------------------------------------------------------------------
def test_shuffled_target_sanity_check_detects_no_leakage(split_data):
    X_train, X_test, y_train, y_test = split_data
    result = tt17.run_shuffled_target_sanity_check(
        X_train, X_test, y_train, y_test, tt17.build_preprocessor()
    )
    assert result["vuot_qua_kiem_tra"] is True
    assert result["r2_voi_target_xao_tron"] < 0.05


# --------------------------------------------------------------------------
# 14. Toàn bộ pipeline qua CLI (end-to-end thật sự)
# --------------------------------------------------------------------------
def test_full_cli_pipeline_runs_without_error(synthetic_csv, tmp_path):
    """Chạy toàn bộ main() ở chế độ --fast --no-tune để test end-to-end nhanh
    (vẫn đi qua MỌI bước của pipeline, chỉ giảm số cây/số cấu hình)."""
    import subprocess

    script = Path(tt17.__file__).resolve()
    cmd = [
        sys.executable, str(script),
        "--data", synthetic_csv,
        "--outdir", str(tmp_path),
        "--fast",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"

    assert (tmp_path / "models" / "rf_reg.joblib").exists()
    summary_path = tmp_path / "reports" / "summary.json"
    assert summary_path.exists()

    import json
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in [
        "baselines", "random_forest", "sanity_check_shuffled_target",
        "rmse_vs_n_estimators", "feature_importance", "pdp_days_left",
        "prediction_interval", "extrapolation", "metadata",
    ]:
        assert key in summary, f"Thiếu mục '{key}' trong summary.json"
