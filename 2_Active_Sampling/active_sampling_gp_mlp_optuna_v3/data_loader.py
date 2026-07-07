from pathlib import Path
import csv
import pandas as pd

def load_labeled_data(csv_path):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path.resolve()}")
    # Auto-detect delimiter so both CSV and TSV inputs are supported.
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"]).delimiter
        return pd.read_csv(path, sep=delimiter)
    except csv.Error:
        return pd.read_csv(path)

def validate_required_columns(df, continuous_cols, discrete_cols, passfail_col, tmax_col, other_regression_cols=None, time_feature_cols=None):
    required = continuous_cols + discrete_cols + [passfail_col, tmax_col]
    required += other_regression_cols or []
    required += time_feature_cols or []
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns:\n" + "\n".join(f"- {c}" for c in missing))

def validate_passfail_labels(df, passfail_col, pass_label, fail_label):
    values = set(df[passfail_col].dropna().unique().tolist())
    allowed = {pass_label, fail_label}
    if not values.issubset(allowed):
        raise ValueError(f"{passfail_col} must contain only NoTP={pass_label}, TP={fail_label}. Found: {sorted(values)}")
