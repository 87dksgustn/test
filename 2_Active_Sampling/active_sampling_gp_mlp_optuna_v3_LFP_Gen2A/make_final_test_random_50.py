import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from data_loader import load_labeled_data
from discrete_space import generate_valid_discrete_combinations


N_OUTPUT = 50
RANDOM_SEED = 42
OUTPUT_SUBDIR = Path("final_test_doe")
OUTPUT_CSV_NAME = "final_test_random_50.csv"
OUTPUT_REPORT_NAME = "final_test_random_50_report.json"


def _row_key(row, continuous_cols, discrete_cols):
    cont = tuple(np.round(row[continuous_cols].to_numpy(dtype=float), 10).tolist())
    disc = tuple(row[discrete_cols].astype(str).tolist())
    return cont + disc


def _build_existing_keys(df, continuous_cols, discrete_cols):
    keys = set()
    for _, row in df[continuous_cols + discrete_cols].iterrows():
        keys.add(_row_key(row, continuous_cols, discrete_cols))
    return keys


def _sample_batch(rng, n_rows, valid_combos, continuous_cols, bounds, discrete_cols):
    combo_idx = rng.integers(0, len(valid_combos), size=n_rows)
    sampled_combos = valid_combos.iloc[combo_idx].reset_index(drop=True)

    out = {}
    for col in continuous_cols:
        lo, hi = bounds[col]
        out[col] = rng.uniform(float(lo), float(hi), size=n_rows)

    for col in discrete_cols:
        out[col] = sampled_combos[col].to_numpy()

    out_df = pd.DataFrame(out)
    out_df.insert(0, "discrete_combo_id", sampled_combos["discrete_combo_id"].to_numpy())
    return out_df


def generate_final_test_random(n_output, seed):
    base_df = load_labeled_data(config.INPUT_CSV)
    valid_combos = generate_valid_discrete_combinations(config.DISCRETE_LEVELS, config.DISCRETE_COLS, config.S_PREFIX)

    rng = np.random.default_rng(seed)
    existing_keys = _build_existing_keys(base_df, config.BASE_CONTINUOUS_COLS, config.DISCRETE_COLS)

    selected_rows = []
    selected_keys = set(existing_keys)

    max_loops = 20
    for _ in range(max_loops):
        needed = n_output - len(selected_rows)
        if needed <= 0:
            break

        batch = _sample_batch(
            rng,
            max(needed * 6, 200),
            valid_combos,
            config.BASE_CONTINUOUS_COLS,
            config.CONTINUOUS_BOUNDS,
            config.DISCRETE_COLS,
        )

        for _, row in batch.iterrows():
            key = _row_key(row, config.BASE_CONTINUOUS_COLS, config.DISCRETE_COLS)
            if key in selected_keys:
                continue
            selected_rows.append(row)
            selected_keys.add(key)
            if len(selected_rows) >= n_output:
                break

    if len(selected_rows) < n_output:
        raise RuntimeError(f"Could not generate {n_output} unique random samples. Generated: {len(selected_rows)}")

    out = pd.DataFrame(selected_rows).reset_index(drop=True)

    for c1, c2, new_col in config.INTERACTION_TERMS:
        out[new_col] = pd.to_numeric(out[c1], errors="coerce") * pd.to_numeric(out[c2], errors="coerce")

    out.insert(0, "final_test_rank", np.arange(1, len(out) + 1))
    return out


def save_outputs(final_df, n_output, seed):
    output_dir = config.OUTPUT_DIR / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    out_csv = output_dir / OUTPUT_CSV_NAME
    out_report = output_dir / OUTPUT_REPORT_NAME

    final_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    report = {
        "method": "pure_random_uniform",
        "n_output": int(len(final_df)),
        "requested_n_output": int(n_output),
        "seed": int(seed),
        "continuous_cols": list(config.BASE_CONTINUOUS_COLS),
        "continuous_bounds": {k: list(v) for k, v in config.CONTINUOUS_BOUNDS.items() if k in config.BASE_CONTINUOUS_COLS},
        "discrete_cols": list(config.DISCRETE_COLS),
        "discrete_levels": config.DISCRETE_LEVELS,
        "discrete_counts": {
            col: final_df[col].value_counts().to_dict() for col in config.DISCRETE_COLS
        },
        "interaction_terms": list(config.INTERACTION_TERMS),
        "note": "No predictive model, acquisition score, or optimization logic used.",
    }

    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_csv, out_report


def main():
    final_df = generate_final_test_random(n_output=N_OUTPUT, seed=RANDOM_SEED)
    out_csv, out_report = save_outputs(final_df, n_output=N_OUTPUT, seed=RANDOM_SEED)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_report}")
    print(f"Total rows: {len(final_df)}")
    print(final_df.head(10))


if __name__ == "__main__":
    main()
