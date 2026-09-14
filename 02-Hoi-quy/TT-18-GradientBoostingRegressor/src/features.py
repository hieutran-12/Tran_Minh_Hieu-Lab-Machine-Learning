"""
features.py
------------
Data cleaning, missing-value handling, and feature engineering for the
Ames Housing AVM (Automated Valuation Model) pipeline.

The Ames dataset has two fundamentally different kinds of missing data
(see the official data dictionary):

1. "Feature absent" (NaN is meaningful, e.g. PoolQC=NaN -> no pool).
   These columns are filled with the literal string 'None' (or 0 for the
   matching numeric column, e.g. GarageArea=NaN -> 0) and must NEVER be
   dropped, because the missingness itself is signal.

2. "Genuinely missing" values (a handful of numeric fields such as
   LotFrontage / MasVnrArea) which are imputed with the median.

Ordinal quality columns (Po < Fa < TA < Gd < Ex, plus 'None' where the
feature doesn't exist) are encoded with OrdinalEncoder so the model sees
the true rank instead of an arbitrary one-hot split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

TARGET = "SalePrice"
ID_COL = "Id"

# --- 1. Columns where NaN means "this feature does not exist on the house" ---
# Categorical "none" columns -> filled with the string 'None'
NONE_CATEGORICAL_COLS = [
    "PoolQC", "Alley", "Fence", "FireplaceQu", "GarageType", "GarageFinish",
    "GarageQual", "GarageCond", "BsmtQual", "BsmtCond", "BsmtExposure",
    "BsmtFinType1", "BsmtFinType2", "MasVnrType", "MiscFeature",
]
# Numeric "none" columns -> filled with 0 (e.g. GarageArea=NaN -> no garage -> 0)
NONE_NUMERIC_COLS = [
    "GarageArea", "GarageCars", "GarageYrBlt", "TotalBsmtSF", "BsmtFinSF1",
    "BsmtFinSF2", "BsmtUnfSF", "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
]

# --- 2. Genuinely-missing numeric columns -> median imputation ---
MEDIAN_IMPUTE_COLS = ["LotFrontage"]

# --- 3. Ordinal quality columns, worst -> best. 'None' is added as the floor
#        automatically for any column that also appears in NONE_CATEGORICAL_COLS.
ORDINAL_COLS = {
    "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
    "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "PoolQC": ["None", "Fa", "TA", "Gd", "Ex"],
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    "GarageFinish": ["None", "Unf", "RFn", "Fin"],
}


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the raw Kaggle train.csv (or an equally-shaped local export)."""
    df = pd.read_csv(csv_path)
    return df


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a table of columns with missing values, split into the two
    business categories described in the module docstring."""
    na_counts = df.isna().sum()
    na_counts = na_counts[na_counts > 0].sort_values(ascending=False)
    rows = []
    for col, n in na_counts.items():
        if col in NONE_CATEGORICAL_COLS or col in NONE_NUMERIC_COLS:
            category = "khong_co_tien_ich (NaN co y nghia)"
        elif col in MEDIAN_IMPUTE_COLS:
            category = "thieu_that (se dien median)"
        else:
            category = "thieu_that (chua phan loai - kiem tra)"
        rows.append({"cot": col, "so_dong_thieu": n,
                      "ty_le_%": round(100 * n / len(df), 1), "phan_loai": category})
    return pd.DataFrame(rows)


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the two missing-value strategies. Never drops a column."""
    df = df.copy()
    for col in NONE_CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    for col in NONE_NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    for col in MEDIAN_IMPUTE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    # Any remaining stragglers (rare categorical fields like Electrical):
    # impute categorical with mode, numeric with median, rather than dropping rows/cols.
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode(dropna=True).iloc[0])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the domain features called out in the brief:
    TotalSF, house age at sale, and whether it was remodeled."""
    df = df.copy()
    flr1 = df.get("1stFlrSF", 0)
    flr2 = df.get("2ndFlrSF", 0)
    bsmt = df.get("TotalBsmtSF", 0)
    df["TotalSF"] = flr1 + flr2 + bsmt
    if "YrSold" in df.columns and "YearBuilt" in df.columns:
        df["TuoiNha"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    if "YearRemodAdd" in df.columns and "YearBuilt" in df.columns:
        df["DaSuaChua"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    return df


def split_columns(df: pd.DataFrame, target: str = TARGET, id_col: str = ID_COL):
    """Classify the remaining feature columns into ordinal / nominal / numeric."""
    feature_cols = [c for c in df.columns if c not in (target, id_col)]
    ordinal_cols = [c for c in ORDINAL_COLS if c in feature_cols]
    numeric_cols = [c for c in feature_cols
                     if c not in ordinal_cols and pd.api.types.is_numeric_dtype(df[c])]
    nominal_cols = [c for c in feature_cols
                     if c not in ordinal_cols and c not in numeric_cols]
    return numeric_cols, ordinal_cols, nominal_cols


def build_preprocessor(df: pd.DataFrame, target: str = TARGET, id_col: str = ID_COL) -> ColumnTransformer:
    """Build the sklearn ColumnTransformer:
    - numeric  -> passthrough (already cleaned upstream)
    - ordinal  -> OrdinalEncoder, category order preserved (quality rank kept)
    - nominal  -> OneHotEncoder (no ordinal meaning to preserve)
    """
    numeric_cols, ordinal_cols, nominal_cols = split_columns(df, target, id_col)

    ordinal_categories = [ORDINAL_COLS[c] for c in ordinal_cols]

    transformers = []
    if numeric_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_cols))
    if ordinal_cols:
        transformers.append((
            "ord",
            OrdinalEncoder(categories=ordinal_categories,
                            handle_unknown="use_encoded_value", unknown_value=-1),
            ordinal_cols,
        ))
    if nominal_cols:
        transformers.append((
            "nom",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            nominal_cols,
        ))

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return preprocessor, (numeric_cols, ordinal_cols, nominal_cols)
