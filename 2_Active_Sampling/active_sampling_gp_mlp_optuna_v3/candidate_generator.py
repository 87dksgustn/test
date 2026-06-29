import numpy as np
import pandas as pd
from scipy.stats import qmc

def filter_excluded_reference_ranges(df, excluded_reference_ranges):
    if not excluded_reference_ranges:
        return df.reset_index(drop=True)
    mask = pd.Series(True, index=df.index)
    for col, info in excluded_reference_ranges.items():
        c = float(info["center"]); hw = float(info["half_width"])
        mask &= ~df[col].between(c - hw, c + hw, inclusive="both")
    return df.loc[mask].reset_index(drop=True)

def generate_lhs_continuous_candidates(n, continuous_cols, continuous_bounds, seed):
    sampler = qmc.LatinHypercube(d=len(continuous_cols), seed=seed)
    x_unit = sampler.random(n=n)
    lows = np.array([continuous_bounds[c][0] for c in continuous_cols], dtype=float)
    highs = np.array([continuous_bounds[c][1] for c in continuous_cols], dtype=float)
    return pd.DataFrame(qmc.scale(x_unit, lows, highs), columns=continuous_cols)

def generate_candidate_pool(valid_combos, continuous_cols, continuous_bounds, discrete_cols, candidates_per_combo, excluded_reference_ranges, seed=42):
    parts = []
    for i, row in valid_combos.reset_index(drop=True).iterrows():
        cont = generate_lhs_continuous_candidates(candidates_per_combo, continuous_cols, continuous_bounds, seed + i * 1009)
        for col in discrete_cols:
            cont[col] = row[col]
        cont["discrete_combo_id"] = row["discrete_combo_id"]
        parts.append(cont)
    pool = pd.concat(parts, ignore_index=True)
    return filter_excluded_reference_ranges(pool, excluded_reference_ranges)
