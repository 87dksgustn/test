import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create Tmax prediction-error Top-N table using row-aligned comparison "
            "between Itr dataset and CV OOF diagnostics."
        )
    )
    parser.add_argument(
        "--itr-csv",
        type=Path,
        default=Path("Itr_11_Dataset.csv"),
        help="Path to iteration dataset CSV (ground truth).",
    )
    parser.add_argument(
        "--cv-oof-csv",
        type=Path,
        default=Path("outputs/Itr_12/Performance/cv_oof_diagnostics.csv"),
        help="Path to CV OOF diagnostics CSV containing oof_tmax_pred.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top N rows by absolute Tmax error.",
    )
    parser.add_argument(
        "--notp-only",
        action="store_true",
        help="Use NoTP rows only (TP_NoTP == 0), matching Tmax plot domain.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/Itr_12/Performance/tmax_error_top10_rows.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be a positive integer.")

    itr_df = pd.read_csv(args.itr_csv).reset_index(drop=True)
    cv_df = pd.read_csv(args.cv_oof_csv).reset_index(drop=True)

    n = min(len(itr_df), len(cv_df))
    if n == 0:
        raise ValueError("Input CSV is empty.")

    # Row-aligned comparison: do not join on Eval# because Eval# is not unique.
    base = pd.DataFrame(
        {
            "data_row_no": np.arange(n),
            "excel_row_no": np.arange(n) + 2,
            "Eval#": pd.to_numeric(itr_df.loc[: n - 1, "Eval#"], errors="coerce"),
            "TP_NoTP": pd.to_numeric(cv_df.loc[: n - 1, "TP_NoTP"], errors="coerce"),
            "Time_MaxT": cv_df.loc[: n - 1, "Time_MaxT"],
            "actual_tmax": pd.to_numeric(itr_df.loc[: n - 1, "MaxT_TB"], errors="coerce"),
            "pred_tmax": pd.to_numeric(cv_df.loc[: n - 1, "oof_tmax_pred"], errors="coerce"),
        }
    )

    valid = base[np.isfinite(base["actual_tmax"]) & np.isfinite(base["pred_tmax"])].copy()
    if args.notp_only:
        valid = valid[valid["TP_NoTP"] == 0].copy()

    if len(valid) == 0:
        raise ValueError("No valid rows after filtering. Check input files or --notp-only option.")

    valid["abs_err"] = (valid["actual_tmax"] - valid["pred_tmax"]).abs()
    top = valid.sort_values("abs_err", ascending=False).head(args.top_n).copy()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print(f"Saved: {args.output_csv}")
    print(f"Rows used (after filter): {len(valid)}")
    print(top[["excel_row_no", "actual_tmax", "pred_tmax", "abs_err", "Eval#", "Time_MaxT"]].to_string(index=False))


if __name__ == "__main__":
    main()
