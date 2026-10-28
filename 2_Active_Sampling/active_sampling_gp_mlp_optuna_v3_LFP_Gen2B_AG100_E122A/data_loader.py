from pathlib import Path
import csv
import pandas as pd

def load_labeled_data(csv_path, cfg=None):
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
            df = pd.read_csv(path, sep=delimiter, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        df = pd.read_csv(path, sep=delimiter, encoding="utf-8", encoding_errors="replace")
    df.columns = df.columns.astype(str).str.replace(r"\s+", "", regex=True)
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names after removing whitespace.")
    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].map(lambda value: "".join(value.split()) if isinstance(value, str) else value)
    if cfg is not None:
        numeric_cols = list(cfg.BASE_CONTINUOUS_COLS) + [cfg.TPNoTP_COL, cfg.TMAX_COL]
        numeric_cols += list(getattr(cfg, "OTHER_REGRESSION_COLS", [])) + list(getattr(cfg, "TIME_FEATURE_COLS", []))
        for column in dict.fromkeys(numeric_cols):
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="raise")
        for column in cfg.BASE_CONTINUOUS_COLS:
            if column != "G_PreForce" and column in df.columns and column in cfg.EXCLUDED_REFERENCE_RANGES:
                df[column] = df[column] + float(cfg.EXCLUDED_REFERENCE_RANGES[column]["center"])
    return df

def load_candidate_data(csv_path, cfg):
    df = load_labeled_data(csv_path)
    input_cols = sorted(list(cfg.BASE_CONTINUOUS_COLS) + list(cfg.DISCRETE_COLS))
    separator_col = f"Unnamed:{len(input_cols)}"
    if (
        len(df.columns) <= len(input_cols)
        or df.columns[:len(input_cols)].tolist() != input_cols
        or df.columns[len(input_cols)] != separator_col
        or df[separator_col].notna().any()
    ):
        return df
    centers = getattr(cfg, "EXCLUDED_REFERENCE_RANGES", {})
    for col in cfg.BASE_CONTINUOUS_COLS:
        if col != "G_PreForce" and col in centers:
            df[col] = df[col].astype(float) + float(centers[col]["center"])
    return df.drop(columns=[separator_col])

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
