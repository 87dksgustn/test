from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config


REFERENCE_CSV = Path("outputs/final_test_doe/initial_DOE_best_seed_greedy_maximin.csv")
N_OUTPUT = 50
RANDOM_SEED = 42
OVERSAMPLE_FACTOR = 120

CONTINUOUS_ALIAS = {
    "A_Cell_D": ["A_Cell_D", "Cell_D"],
    "C_Barrier_Thx": ["C_Barrier_Thx", "Barrier_Thx"],
    "E_Barrier_Outer_Thx": ["E_Barrier_Outer_Thx", "Barrier_Outer_Thx"],
    "F_ThermalResin_Thx": ["F_ThermalResin_Thx", "ThermalResin_Thx"],
}


def _pick_existing_col(columns: list[str], candidates: list[str]) -> str:
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(f"None of candidate columns exist: {candidates}")


def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=0)
    sd = df.std(axis=0).replace(0.0, 1.0)
    return (df - mu) / sd


def _min_dist_to_selected(x: np.ndarray, selected_idx: list[int], cand_idx: np.ndarray) -> np.ndarray:
    if not selected_idx:
        return np.full(len(cand_idx), np.inf)
    s = x[np.array(selected_idx)]
    c = x[cand_idx]
    d2 = ((c[:, None, :] - s[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(d2.min(axis=1))


def greedy_maximin_subset(df_cont: pd.DataFrame, n_pick: int, seed: int) -> np.ndarray:
    x = _zscore(df_cont).to_numpy(dtype=float)
    n = x.shape[0]
    rng = np.random.default_rng(seed)

    if n_pick >= n:
        return np.arange(n, dtype=int)

    center = np.median(x, axis=0)
    d_center = np.sqrt(((x - center) ** 2).sum(axis=1))
    first = int(np.argmin(d_center))

    selected = [first]
    remaining = np.array([i for i in range(n) if i != first], dtype=int)

    while len(selected) < n_pick:
        dmin = _min_dist_to_selected(x, selected, remaining)
        max_d = dmin.max()
        ties = remaining[np.flatnonzero(np.isclose(dmin, max_d))]
        chosen = int(rng.choice(ties))
        selected.append(chosen)
        remaining = remaining[remaining != chosen]

    return np.array(selected, dtype=int)


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    ref = pd.read_csv(REFERENCE_CSV)
    ref = ref.loc[:, ~ref.columns.astype(str).str.startswith("Unnamed")].copy()

    base_cols = list(config.BASE_CONTINUOUS_COLS)
    cont_src = {
        target: _pick_existing_col(ref.columns.tolist(), CONTINUOUS_ALIAS[target])
        for target in base_cols
    }

    ref_renamed = ref.rename(columns={v: k for k, v in cont_src.items()}).copy()
    for col in base_cols:
        ref_renamed[col] = pd.to_numeric(ref_renamed[col], errors="coerce")

    ref_renamed = ref_renamed.dropna(subset=base_cols).reset_index(drop=True)
    if len(ref_renamed) < 2:
        raise ValueError("Reference CSV has too few valid rows for distribution-based DOE generation.")

    n_pool = max(N_OUTPUT * OVERSAMPLE_FACTOR, 1000)
    src_idx = rng.integers(0, len(ref_renamed), size=n_pool)
    pool = ref_renamed.iloc[src_idx][base_cols].reset_index(drop=True).copy()

    for col in base_cols:
        q75, q25 = np.percentile(ref_renamed[col], [75, 25])
        iqr = float(q75 - q25)
        sigma = max(iqr * 0.05, 1e-6)
        pool[col] = pool[col].to_numpy(dtype=float) + rng.normal(0.0, sigma, size=n_pool)
        lo, hi = config.CONTINUOUS_BOUNDS[col]
        pool[col] = pool[col].clip(lower=float(lo), upper=float(hi))

    picked_idx = greedy_maximin_subset(pool[base_cols], N_OUTPUT, RANDOM_SEED)
    out = pool.iloc[picked_idx].reset_index(drop=True)

    # Discrete policy A: sample from current config levels.
    out["B_Barrier_Type"] = "Si1"
    d_levels = list(config.DISCRETE_LEVELS["D_Barrier_Outer_Type"])
    out["D_Barrier_Outer_Type"] = rng.choice(d_levels, size=len(out), replace=True)

    for c1, c2, new_col in config.INTERACTION_TERMS:
        out[new_col] = pd.to_numeric(out[c1], errors="coerce") * pd.to_numeric(out[c2], errors="coerce")

    output_dir = config.OUTPUT_DIR / "final_test_doe"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "holdout_doe_50_from_initial_distribution.csv"
    report_path = output_dir / "holdout_doe_50_from_initial_distribution_report.json"

    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    report = {
        "reference_csv": str(REFERENCE_CSV),
        "n_reference": int(len(ref_renamed)),
        "n_output": int(len(out)),
        "seed": RANDOM_SEED,
        "method": "bootstrap_jitter_then_greedy_maximin",
        "source_column_mapping": {"continuous": cont_src},
        "continuous_cols": base_cols,
        "discrete_policy": "config_level_uniform_sampling",
        "discrete_counts": {
            "B_Barrier_Type": out["B_Barrier_Type"].value_counts().to_dict(),
            "D_Barrier_Outer_Type": out["D_Barrier_Outer_Type"].value_counts().to_dict(),
        },
        "summary_ref": ref_renamed[base_cols].describe().to_dict(),
        "summary_output": out[base_cols].describe().to_dict(),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
