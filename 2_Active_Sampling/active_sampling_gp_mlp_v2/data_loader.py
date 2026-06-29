from pathlib import Path
import pandas as pd


def load_labeled_data(csv_path):
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Input CSV not found: {p.resolve()}")
    return pd.read_csv(p)


def validate_required_columns(df, continuous_cols, discrete_cols, passfail_col, tmax_col, other_regression_cols=None, time_feature_cols=None):
    required = continuous_cols + discrete_cols + [passfail_col, tmax_col]
    required += other_regression_cols or []
    required += time_feature_cols or []
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns:\n" + "\n".join(f"- {c}" for c in missing))


def validate_passfail_labels(df, passfail_col, pass_label, fail_label):
    vals = set(df[passfail_col].dropna().unique().tolist())
    allowed = {pass_label, fail_label}
    if not vals.issubset(allowed):
        raise ValueError(f"{passfail_col} must contain only PASS={pass_label}, FAIL={fail_label}. Found: {sorted(vals)}")
