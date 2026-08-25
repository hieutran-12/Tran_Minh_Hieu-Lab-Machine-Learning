
import argparse
import os

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    f1_score,
    log_loss,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, OneHotEncoder


FEATURE_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


CAT_COLS = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


NUM_COLS = [
    "age",
    "capital_loss",
    "hours_per_week",
]


ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DEFAULT_MODEL_PATH = os.path.join(
    ROOT_DIR,
    "models",
    "gb_pipeline.joblib",
)

DEFAULT_REPORTS_DIR = os.path.join(
    ROOT_DIR,
    "reports",
)


# =========================================================
# LOAD DATA
# =========================================================

def load_data(data_path: str) -> pd.DataFrame:

    df = pd.read_csv(
        data_path,
        header=None,
        sep=r",\s*",
        names=FEATURE_NAMES,
        na_values="?",
        engine="python",
    )

    # Adult dataset đôi khi có khoảng trắng
    df["income"] = df["income"].str.strip()

    # Loại bỏ feature đã quyết định không dùng
    df = df.drop(
        columns=[
            "education_num",
            "fnlwgt",
        ]
    )

    return df


# =========================================================
# BUILD PIPELINE
# =========================================================

def build_pipeline() -> Pipeline:

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CAT_COLS,
            ),

            (
                "capital_gain",
                Binarizer(threshold=0),
                ["capital_gain"],
            ),

            (
                "num",
                "passthrough",
                NUM_COLS,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),

            (
                "model",
                GradientBoostingClassifier(
                    random_state=42,
                ),
            ),
        ]
    )

    return pipeline


# =========================================================
# EVALUATE
# =========================================================

def evaluate(
    name: str,
    y_true,
    y_pred,
) -> None:

    print(f"\n===== {name} =====")

    print(
        "Precision:",
        precision_score(
            y_true,
            y_pred,
            pos_label=">50K",
        ),
    )

    print(
        "Recall:",
        recall_score(
            y_true,
            y_pred,
            pos_label=">50K",
        ),
    )

    print(
        "F1-score:",
        f1_score(
            y_true,
            y_pred,
            pos_label=">50K",
        ),
    )


# =========================================================
# GRID SEARCH + HEATMAP
# =========================================================

def tune_model(
    pipeline,
    x_train,
    y_train,
    out_path: str,
):

    f1_scorer = make_scorer(
        f1_score,
        pos_label=">50K",
    )

    param_grid = {
        "model__learning_rate": [
            0.01,
            0.05,
            0.1,
        ],

        "model__n_estimators": [
            50,
            100,
            200,
        ],
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring=f1_scorer,
        n_jobs=-1,
    )

    grid.fit(
        x_train,
        y_train,
    )

    results = pd.DataFrame(
        grid.cv_results_
    )

    pivot = results.pivot(
        index="param_model__learning_rate",
        columns="param_model__n_estimators",
        values="mean_test_score",
    )

    plt.figure(
        figsize=(7, 5)
    )

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
    )

    plt.xlabel("n_estimators")
    plt.ylabel("learning_rate")

    plt.title(
        "F1-score CV theo learning_rate và n_estimators"
    )

    plt.savefig(
        out_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nBest params:",
        grid.best_params_,
    )

    print(
        "Best CV F1:",
        grid.best_score_,
    )

    # QUAN TRỌNG:
    # Trả về model tốt nhất
    return grid.best_estimator_


# =========================================================
# LOSS CURVE
# =========================================================

def plot_loss_theo_so_cay(
    pipeline,
    x_train,
    y_train,
    x_test,
    y_test,
    out_path: str,
):

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    model = pipeline.named_steps[
        "model"
    ]

    X_train_processed = preprocessor.transform(
        x_train
    )

    X_test_processed = preprocessor.transform(
        x_test
    )

    train_loss = []
    test_loss = []

    for (
        train_proba,
        test_proba,
    ) in zip(
        model.staged_predict_proba(
            X_train_processed
        ),
        model.staged_predict_proba(
            X_test_processed
        ),
    ):

        train_loss.append(
            log_loss(
                y_train,
                train_proba,
                labels=model.classes_,
            )
        )

        test_loss.append(
            log_loss(
                y_test,
                test_proba,
                labels=model.classes_,
            )
        )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        range(1, len(train_loss) + 1),
        train_loss,
        label="Train Loss",
    )

    plt.plot(
        range(1, len(test_loss) + 1),
        test_loss,
        label="Validation Loss",
    )

    plt.xlabel(
        "Số lượng cây"
    )

    plt.ylabel(
        "Log Loss"
    )

    plt.title(
        "Gradient Boosting: Train vs Validation Loss"
    )

    plt.legend()

    plt.grid()

    plt.savefig(
        out_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


# =========================================================
# BIAS REPORT
# =========================================================

def bias_report(
    pipeline,
    x_test,
    y_test,
    group_col: str,
):

    df_report = x_test.copy()

    df_report["y_true"] = y_test.values

    df_report["y_pred"] = pipeline.predict(
        x_test
    )

    rows = []

    for group, sub in df_report.groupby(
        group_col
    ):

        selection_rate = (
            sub["y_pred"] == ">50K"
        ).mean()

        recall = recall_score(
            sub["y_true"],
            sub["y_pred"],
            pos_label=">50K",
            zero_division=0,
        )

        precision = precision_score(
            sub["y_true"],
            sub["y_pred"],
            pos_label=">50K",
            zero_division=0,
        )

        rows.append(
            {
                group_col: group,

                "Số lượng":
                    len(sub),

                "Tỉ lệ dự đoán >50K":
                    round(selection_rate, 3),

                "Recall":
                    round(recall, 3),

                "Precision":
                    round(precision, 3),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "Tỉ lệ dự đoán >50K",
        ascending=False,
    )


# =========================================================
# PLOT BIAS
# =========================================================

def plot_bias_by_group(
    sex_report,
    race_report,
    out_path: str,
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    sex_report.plot(
        kind="bar",
        x="sex",
        y="Tỉ lệ dự đoán >50K",
        ax=axes[0],
        legend=False,
    )

    axes[0].set_title(
        "Tỉ lệ dự đoán >50K theo giới tính"
    )

    race_report.plot(
        kind="bar",
        x="race",
        y="Tỉ lệ dự đoán >50K",
        ax=axes[1],
        legend=False,
    )

    axes[1].set_title(
        "Tỉ lệ dự đoán >50K theo chủng tộc"
    )

    plt.tight_layout()

    plt.savefig(
        out_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train Gradient Boosting "
            "trên Adult Income Dataset"
        )
    )

    parser.add_argument(
        "--data-path",
        required=True,
        help="Đường dẫn tới adult.data",
    )

    parser.add_argument(
        "--model-out",
        default=DEFAULT_MODEL_PATH,
    )

    parser.add_argument(
        "--reports-dir",
        default=DEFAULT_REPORTS_DIR,
    )

    args = parser.parse_args()


    # Tạo thư mục output

    model_dir = os.path.dirname(
        args.model_out
    )

    if model_dir:
        os.makedirs(
            model_dir,
            exist_ok=True,
        )

    os.makedirs(
        args.reports_dir,
        exist_ok=True,
    )


    # -----------------------------------------------------
    # 1. Load raw data
    # -----------------------------------------------------

    df = load_data(
        args.data_path
    )

    X = df.drop(
        columns=["income"]
    )

    y = df["income"]


    # -----------------------------------------------------
    # 2. Train / Test split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    )


    # -----------------------------------------------------
    # 3. Build pipeline
    # -----------------------------------------------------

    pipeline = build_pipeline()


    # -----------------------------------------------------
    # 4. Hyperparameter tuning
    # -----------------------------------------------------

    best_pipeline = tune_model(
        pipeline,
        X_train,
        y_train,
        os.path.join(
            args.reports_dir,
            "lr_vs_nestimators.png",
        ),
    )


    # -----------------------------------------------------
    # 5. Evaluate
    # -----------------------------------------------------

    y_pred = best_pipeline.predict(
        X_test
    )

    evaluate(
        "Gradient Boosting",
        y_test,
        y_pred,
    )


    # -----------------------------------------------------
    # 6. Loss curve
    # -----------------------------------------------------

    plot_loss_theo_so_cay(
        best_pipeline,
        X_train,
        y_train,
        X_test,
        y_test,
        os.path.join(
            args.reports_dir,
            "loss_theo_so_cay.png",
        ),
    )


    # -----------------------------------------------------
    # 7. Bias report
    # -----------------------------------------------------

    sex_report = bias_report(
        best_pipeline,
        X_test,
        y_test,
        "sex",
    )

    race_report = bias_report(
        best_pipeline,
        X_test,
        y_test,
        "race",
    )

    plot_bias_by_group(
        sex_report,
        race_report,
        os.path.join(
            args.reports_dir,
            "bias_by_group.png",
        ),
    )

    print(
        "\nBias theo sex:\n",
        sex_report,
    )

    print(
        "\nBias theo race:\n",
        race_report,
    )


    # -----------------------------------------------------
    # 8. SAVE FINAL PIPELINE
    # -----------------------------------------------------

    joblib.dump(
        best_pipeline,
        args.model_out,
    )

    print(
        f"\nĐã lưu model tại: "
        f"{args.model_out}"
    )

    print(
        f"Đã lưu reports tại: "
        f"{args.reports_dir}"
    )


if __name__ == "__main__":
    main()