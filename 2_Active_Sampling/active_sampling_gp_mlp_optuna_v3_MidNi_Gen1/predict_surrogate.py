from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import config
from surrogate_bundle import get_bundle_info, load_surrogate_bundle, predict_with_bundle


def build_parser():
    parser = argparse.ArgumentParser(description="Run predictions with a saved surrogate bundle.")
    parser.add_argument(
        "--bundle",
        default=str(getattr(config, "OUTPUT_DIR", Path("outputs")) / "latest_surrogate_bundle.pkl"),
        help="Path to saved surrogate bundle (.pkl). Defaults to outputs/latest_surrogate_bundle.pkl",
    )
    parser.add_argument("--input-csv", help="Optional CSV of input rows to score.")
    parser.add_argument("--output-csv", help="Where to save predictions when --input-csv is used.")
    parser.add_argument("--show-bundle-info", action="store_true", help="Print model/version metadata before prediction.")
    # Direct-input arguments are derived from config so they always match the active study.
    for col in getattr(config, "BASE_CONTINUOUS_COLS", []):
        parser.add_argument(f"--{col}", type=float)
    for col in getattr(config, "DISCRETE_COLS", []):
        parser.add_argument(f"--{col}", type=str)
    return parser


def read_input_csv_with_fallback(csv_path):
    last_err = None
    for enc in ["utf-8-sig", "cp949", "euc-kr", "latin-1"]:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except Exception as e:
            last_err = e
    raise ValueError(f"Could not read CSV file '{csv_path}'. Last error: {last_err}")


def print_bundle_info(bundle):
    info = get_bundle_info(bundle)
    print("[Bundle Info]")
    print(json.dumps(info, indent=2, ensure_ascii=False))


def friendly_error_message(exc):
    msg = str(exc)
    if "Missing input columns" in msg:
        return msg + "\nHint: Check CSV header names exactly."
    if "out-of-range" in msg:
        return msg + "\nHint: Keep continuous inputs inside config bounds."
    if "invalid level" in msg:
        return msg + "\nHint: Use one of the allowed categorical levels shown in the message."
    if "Surrogate bundle not found" in msg:
        return msg + "\nHint: Run 0_main_active_sampling.py first to create outputs/latest_surrogate_bundle.pkl"
    if "Could not read CSV file" in msg:
        return msg + "\nHint: Confirm file path and CSV encoding (utf-8-sig/cp949/euc-kr)."
    return msg


def build_single_input_df(args):
    required = list(getattr(config, "BASE_CONTINUOUS_COLS", [])) + list(getattr(config, "DISCRETE_COLS", []))
    values = {name: getattr(args, name) for name in required}
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError("Missing direct input argument(s): " + ", ".join(missing))
    return pd.DataFrame([values])


def main():
    args = build_parser().parse_args()
    try:
        bundle_path = Path(args.bundle)
        if not bundle_path.exists():
            raise FileNotFoundError(f"Surrogate bundle not found: {bundle_path}")

        bundle = load_surrogate_bundle(bundle_path)
        if args.show_bundle_info:
            print_bundle_info(bundle)

        if args.input_csv:
            input_df = read_input_csv_with_fallback(args.input_csv)
            pred_df = predict_with_bundle(bundle, input_df)
            out_df = pd.concat([input_df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
            output_csv = Path(args.output_csv) if args.output_csv else Path(args.input_csv).with_name(Path(args.input_csv).stem + "_predictions.csv")
            out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            print(f"Saved predictions: {output_csv}")
            return

        input_df = build_single_input_df(args)
        pred_df = predict_with_bundle(bundle, input_df)
        result = {**input_df.iloc[0].to_dict(), **pred_df.iloc[0].to_dict()}
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        print("[ERROR] Prediction failed")
        print(friendly_error_message(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
