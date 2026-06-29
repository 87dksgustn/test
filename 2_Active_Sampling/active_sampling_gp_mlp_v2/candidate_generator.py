import numpy as np
import pandas as pd
from scipy.stats import qmc


def filter_excluded_reference_ranges(
    df: pd.DataFrame,
    excluded_reference_ranges: dict,
) -> pd.DataFrame:
    """
    Removes candidates where ANY continuous variable falls into its excluded range.
    Existing labeled data should not be filtered with this function.
    """
    if not excluded_reference_ranges:
        return df.reset_index(drop=True)

    mask = pd.Series(True, index=df.index)

    for col, info in excluded_reference_ranges.items():
        center = float(info["center"])
        half_width = float(info["half_width"])
        low = center - half_width
        high = center + half_width
        mask &= ~df[col].between(low, high, inclusive="both")

    return df.loc[mask].reset_index(drop=True)


def generate_lhs_continuous_candidates(
    n: int,
    continuous_cols: list[str],
    continuous_bounds: dict[str, tuple[float, float]],
    seed: int,
) -> pd.DataFrame:
    sampler = qmc.LatinHypercube(d=len(continuous_cols), seed=seed)
    x_unit = sampler.random(n=n)

    lows = np.array([continuous_bounds[c][0] for c in continuous_cols], dtype=float)
    highs = np.array([continuous_bounds[c][1] for c in continuous_cols], dtype=float)

    x = qmc.scale(x_unit, lows, highs)
    return pd.DataFrame(x, columns=continuous_cols)


def generate_candidate_pool(
    valid_combos: pd.DataFrame,
    continuous_cols: list[str],
    continuous_bounds: dict[str, tuple[float, float]],
    discrete_cols: list[str],
    candidates_per_combo: int,
    excluded_reference_ranges: dict,
    seed: int = 42,
) -> pd.DataFrame:
    all_parts = []

    for i, row in valid_combos.reset_index(drop=True).iterrows():
        combo_seed = seed + i * 1009
        cont = generate_lhs_continuous_candidates(
            n=candidates_per_combo,
            continuous_cols=continuous_cols,
            continuous_bounds=continuous_bounds,
            seed=combo_seed,
        )

        for col in discrete_cols:
            cont[col] = row[col]

        cont["discrete_combo_id"] = row["discrete_combo_id"]
        all_parts.append(cont)

    pool = pd.concat(all_parts, axis=0, ignore_index=True)
    pool = filter_excluded_reference_ranges(pool, excluded_reference_ranges)

    return pool.reset_index(drop=True)
