import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.neighbors import NearestNeighbors

def filter_excluded_reference_ranges(df, excluded_reference_ranges):
    if not excluded_reference_ranges:
        return df.reset_index(drop=True)
    mask = pd.Series(True, index=df.index)
    for col, info in excluded_reference_ranges.items():
        c = float(info["center"]); hw = float(info["half_width"])
        mask &= ~df[col].between(c - hw, c + hw, inclusive="both")
    return df.loc[mask].reset_index(drop=True)


def filter_near_existing_data(candidate_pool, labeled_df, continuous_cols, discrete_cols, min_distance=0.05):
    """
    Filter out candidates that are too close to existing labeled data.
    
    Args:
        candidate_pool: DataFrame of candidate points
        labeled_df: DataFrame of existing labeled data
        continuous_cols: list of continuous feature column names
        discrete_cols: list of discrete feature column names  
        min_distance: minimum normalized distance threshold (default 0.05)
    
    Returns:
        Filtered candidate pool with near-duplicates removed
    """
    if len(labeled_df) == 0 or len(candidate_pool) == 0:
        return candidate_pool.reset_index(drop=True)
    
    # Group by discrete combo for efficiency
    combo_col = "discrete_combo_id"
    keep_mask = np.ones(len(candidate_pool), dtype=bool)
    
    for combo_id, cand_group in candidate_pool.groupby(combo_col):
        # Get existing data for this combo
        existing = labeled_df[labeled_df[combo_col] == combo_id]
        if len(existing) == 0:
            continue
        
        # Get continuous features only
        cand_cont = cand_group[continuous_cols].to_numpy(dtype=float)
        exist_cont = existing[continuous_cols].to_numpy(dtype=float)
        
        # Normalize by bounds (0-1 scale)
        cand_norm = cand_cont.copy()
        exist_norm = exist_cont.copy()
        for j, col in enumerate(continuous_cols):
            col_min = cand_cont[:, j].min()
            col_max = cand_cont[:, j].max()
            if col_max > col_min:
                cand_norm[:, j] = (cand_cont[:, j] - col_min) / (col_max - col_min)
                exist_norm[:, j] = (exist_cont[:, j] - col_min) / (col_max - col_min)
        
        # Find nearest neighbor distance to existing data
        nn = NearestNeighbors(n_neighbors=1).fit(exist_norm)
        dists, _ = nn.kneighbors(cand_norm)
        
        # Mark candidates that are too close
        too_close = dists.ravel() < min_distance
        cand_indices = cand_group.index.to_numpy()
        keep_mask[cand_indices[too_close]] = False
    
    n_filtered = (~keep_mask).sum()
    if n_filtered > 0:
        print(f"[INFO] Filtered {n_filtered} candidates too close to existing data (min_dist={min_distance})")
    
    return candidate_pool.loc[keep_mask].reset_index(drop=True)

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
