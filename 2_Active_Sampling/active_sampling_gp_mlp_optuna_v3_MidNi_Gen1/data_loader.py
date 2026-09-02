from pathlib import Path
import csv
import pandas as pd


def _normalize_input_columns(df):
    out = df.copy()

    # Remove fully empty rows (e.g., trailing blank CSV records).
    out = out.dropna(how="all").reset_index(drop=True)

    # Known header aliases used by source datasets.
    rename_map = {
        "G_CoolingLPM": "G_CoolantLPM",
        "G_Coolant_LPM": "G_CoolantLPM",
        "Coolant_LPM": "G_CoolantLPM",
    }
    applicable = {src: dst for src, dst in rename_map.items() if src in out.columns and dst not in out.columns}
    if applicable:
        out = out.rename(columns=applicable)

    # If a discrete column is omitted but config defines exactly one fixed level,
    # synthesize it so continuous-only studies do not need a redundant CSV column.
    try:
        import config

        discrete_cols = list(getattr(config, "DISCRETE_COLS", []))
        continuous_cols = list(getattr(config, "CONTINUOUS_COLS", []))
        passfail_col = getattr(config, "PASSFAIL_COL", None)
        tmax_col = getattr(config, "TMAX_COL", None)
        discrete_levels = dict(getattr(config, "DISCRETE_LEVELS", {}))

        # Drop trailer/meta rows if all core modeling columns are empty.
        core_cols = [c for c in (continuous_cols + discrete_cols + [passfail_col, tmax_col]) if c and c in out.columns]
        if core_cols:
            out = out.loc[~out[core_cols].isna().all(axis=1)].reset_index(drop=True)

        for col in discrete_cols:
            if col not in out.columns:
                levels = list(discrete_levels.get(col, []))
                if len(levels) == 1:
                    out[col] = levels[0]
    except Exception:
        pass

    return out

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
            return _normalize_input_columns(pd.read_csv(path, sep=delimiter, encoding=enc))
        except UnicodeDecodeError:
            continue
    # Final fallback with errors='replace' to avoid crash
    return _normalize_input_columns(pd.read_csv(path, sep=delimiter, encoding="utf-8", errors="replace"))

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
