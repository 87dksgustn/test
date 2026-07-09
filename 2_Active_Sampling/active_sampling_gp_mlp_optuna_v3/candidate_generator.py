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
    combos = valid_combos.reset_index(drop=True)
    n_combo = len(combos)
    n_total = n_combo * int(candidates_per_combo)

    cont_all = np.empty((n_total, len(continuous_cols)), dtype=float)
    for i in range(n_combo):
        start = i * int(candidates_per_combo)
        end = start + int(candidates_per_combo)
        cont = generate_lhs_continuous_candidates(
            candidates_per_combo,
            continuous_cols,
            continuous_bounds,
            seed + i * 1009,
        )
        cont_all[start:end, :] = cont.to_numpy(dtype=float)

    out = {col: cont_all[:, j] for j, col in enumerate(continuous_cols)}
    for col in discrete_cols:
        out[col] = np.repeat(combos[col].to_numpy(), int(candidates_per_combo))
    out["discrete_combo_id"] = np.repeat(combos["discrete_combo_id"].to_numpy(), int(candidates_per_combo))

    pool = pd.DataFrame(out)
    return filter_excluded_reference_ranges(pool, excluded_reference_ranges)
