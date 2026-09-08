from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a final test DOE using simple random sampling only."
    )
    parser.add_argument(
        "--n-cases",
        type=int,
        default=50,
        help="Number of random test cases to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(config.RANDOM_SEED),
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def sample_continuous(rng: np.random.Generator, n_cases: int) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    for col in config.BASE_CONTINUOUS_COLS:
        lo, hi = config.CONTINUOUS_BOUNDS[col]
        data[col] = rng.uniform(float(lo), float(hi), size=n_cases)
    return pd.DataFrame(data)


def sample_discrete(rng: np.random.Generator, n_cases: int) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    for col in config.DISCRETE_COLS:
        levels = list(config.DISCRETE_LEVELS[col])
        if not levels:
            raise ValueError(f"No discrete levels configured for column: {col}")
        data[col] = rng.choice(levels, size=n_cases, replace=True)
    return pd.DataFrame(data)


def main() -> None:
    args = parse_args()
    n_cases = int(args.n_cases)
    seed = int(args.seed)

    if n_cases <= 0:
        raise ValueError("--n-cases must be a positive integer.")

    rng = np.random.default_rng(seed)

    cont_df = sample_continuous(rng, n_cases)
    disc_df = sample_discrete(rng, n_cases)

    out_df = pd.concat([cont_df, disc_df], axis=1)

    for c1, c2, new_col in config.INTERACTION_TERMS:
        out_df[new_col] = pd.to_numeric(out_df[c1], errors="coerce") * pd.to_numeric(
            out_df[c2], errors="coerce"
        )

    out_df.insert(0, "final_test_rank", np.arange(1, len(out_df) + 1))

    output_dir = Path(config.OUTPUT_DIR) / "final_test_doe"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"final_test_dataset_random_{n_cases}.csv"
    report_path = output_dir / f"final_test_dataset_random_{n_cases}_report.json"

    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    report = {
        "method": "simple_random_sampling",
        "note": "No surrogate model, acquisition, or active-sampling logic used.",
        "n_cases": int(n_cases),
        "seed": int(seed),
        "continuous_cols": list(config.BASE_CONTINUOUS_COLS),
        "continuous_bounds": {
            col: [float(config.CONTINUOUS_BOUNDS[col][0]), float(config.CONTINUOUS_BOUNDS[col][1])]
            for col in config.BASE_CONTINUOUS_COLS
        },
        "discrete_cols": list(config.DISCRETE_COLS),
        "discrete_levels": {col: list(config.DISCRETE_LEVELS[col]) for col in config.DISCRETE_COLS},
    }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved random test dataset: {csv_path}")
    print(f"Saved generation report: {report_path}")


if __name__ == "__main__":
    main()
