import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _coerce_numeric_columns(df, cols):
    if not cols:
        return df.copy()
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_preprocessor(continuous_cols, discrete_cols):
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer([
        ("continuous", StandardScaler(), continuous_cols),
        ("discrete", enc, discrete_cols),
    ], remainder="drop")

def make_xy(df, continuous_cols, discrete_cols, passfail_col, tmax_col):
    x = df[continuous_cols + discrete_cols].copy()
    y_class = df[passfail_col].astype(int).to_numpy()
    y_tmax = _coerce_numeric_columns(df, [tmax_col])[tmax_col].to_numpy(dtype=float)
    return x, y_class, y_tmax

def resolve_extra_target_cols(other_regression_cols, time_feature_cols, target_enable_map=None):
    enabled = target_enable_map or {}
    cols = []
    seen = set()

    for col in (other_regression_cols or []):
        if col not in seen:
            cols.append(col)
            seen.add(col)

    for col in (time_feature_cols or []):
        if col in seen:
            continue
        if enabled and bool(enabled.get(col, False)) is False:
            continue
        cols.append(col)
        seen.add(col)

    return cols

def make_extra_targets(df, other_regression_cols, time_feature_cols, target_enable_map=None):
    cols = resolve_extra_target_cols(other_regression_cols, time_feature_cols, target_enable_map=target_enable_map)
    if not cols:
        return None, []
    safe_df = _coerce_numeric_columns(df, cols)
    valid_cols = [col for col in cols if safe_df[col].notna().any()]
    if not valid_cols:
        return None, []
    return safe_df[valid_cols].to_numpy(dtype=float), valid_cols
