from pathlib import Path
import csv
import pandas as pd

def load_labeled_data(csv_path):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path.resolve()}")
    # Auto-detect delimiter so both CSV and TSV inputs are supported.
    # Try multiple encodings for both sniffing and reading.
    encodings = ["utf-8-sig", "cp949", "euc-kr", "latin-1"]
    sample = ""
    for enc in encodings:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                sample = f.read(4096)
            break
        except UnicodeDecodeError:
            continue
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"]).delimiter
    except csv.Error:
        delimiter = ","
    # Try reading with each encoding until success
    for enc in encodings:
        try:
            return pd.read_csv(path, sep=delimiter, encoding=enc)
        except UnicodeDecodeError:
            continue
    # Final fallback with errors='replace' to avoid crash
    return pd.read_csv(path, sep=delimiter, encoding="utf-8", errors="replace")

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
