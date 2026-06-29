import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(continuous_cols, discrete_cols):
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        [("continuous", StandardScaler(), continuous_cols), ("discrete", enc, discrete_cols)],
        remainder="drop",
    )


def make_xy(df, continuous_cols, discrete_cols, passfail_col, tmax_col):
    x = df[continuous_cols + discrete_cols].copy()
    y_class = df[passfail_col].astype(int).to_numpy()
    y_tmax = df[tmax_col].astype(float).to_numpy()
    return x, y_class, y_tmax


def make_extra_targets(df, other_regression_cols, time_feature_cols):
    cols = (other_regression_cols or []) + (time_feature_cols or [])
    if not cols:
        return None, []
    return df[cols].astype(float).to_numpy(), cols
