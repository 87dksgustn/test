import math
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def bucket_counts(batch_size: int, bucket_ratio: dict[str, float]) -> dict[str, int]:
    total_ratio = sum(bucket_ratio.values())
    raw = {k: batch_size * v / total_ratio for k, v in bucket_ratio.items()}

    counts = {k: int(math.floor(v)) for k, v in raw.items()}
    remaining = batch_size - sum(counts.values())

    # Add remaining by largest fractional part.
    frac_order = sorted(raw.keys(), key=lambda k: raw[k] - counts[k], reverse=True)
    for k in frac_order[:remaining]:
        counts[k] += 1

    return counts


def get_acq_col_for_bucket(bucket: str) -> str:
    if bucket == "boundary":
        return "acq_boundary"
    if bucket == "pass_high_tmax":
        return "acq_pass_high_tmax"
    if bucket == "uncertainty_sparse":
        return "acq_uncertainty_sparse"
    if bucket == "random_check":
        return "random_score"
    raise ValueError(f"Unknown bucket: {bucket}")


def combo_counts_after_selection(labeled_df: pd.DataFrame, selected_df: pd.DataFrame) -> dict:
    base = labeled_df["discrete_combo_id"].value_counts().to_dict()
    if len(selected_df):
        add = selected_df["discrete_combo_id"].value_counts().to_dict()
        for k, v in add.items():
            base[k] = base.get(k, 0) + v
    return base


def is_far_enough_from_selected(
    candidate_row,
    candidate_x,
    selected_indices: list[int],
    x_pool,
    min_batch_distance: float,
) -> bool:
    if not selected_indices:
        return True

    selected_x = x_pool[selected_indices]
    dist = np.linalg.norm(selected_x - candidate_x, axis=1)
    return bool(np.min(dist) >= min_batch_distance)


def greedy_select_from_sorted(
    pool_df: pd.DataFrame,
    x_pool,
    labeled_df: pd.DataFrame,
    selected_indices: list[int],
    n_select: int,
    acq_col: str,
    max_samples_per_combo: int,
    min_batch_distance: float,
    rng: np.random.Generator,
) -> list[int]:
    chosen = []

    if acq_col == "random_score":
        order = rng.permutation(pool_df.index.to_numpy())
    else:
        order = pool_df.sort_values(acq_col, ascending=False).index.to_numpy()

    current_selected_df = pool_df.loc[selected_indices] if selected_indices else pool_df.iloc[0:0].copy()

    for idx in order:
        if idx in selected_indices or idx in chosen:
            continue

        combo_id = pool_df.at[idx, "discrete_combo_id"]

        current_counts = combo_counts_after_selection(
            labeled_df=labeled_df,
            selected_df=pool_df.loc[selected_indices + chosen] if selected_indices or chosen else pool_df.iloc[0:0],
        )

        if current_counts.get(combo_id, 0) >= max_samples_per_combo:
            continue

        if not is_far_enough_from_selected(
            candidate_row=pool_df.loc[idx],
            candidate_x=x_pool[idx],
            selected_indices=selected_indices + chosen,
            x_pool=x_pool,
            min_batch_distance=min_batch_distance,
        ):
            continue

        chosen.append(idx)

        if len(chosen) >= n_select:
            break

    return chosen


def select_batch(
    scored_pool: pd.DataFrame,
    x_pool_transformed,
    labeled_df: pd.DataFrame,
    batch_size: int,
    bucket_ratio: dict[str, float],
    max_samples_per_combo: int,
    min_batch_distance: float,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    pool = scored_pool.copy().reset_index(drop=True)
    pool["random_score"] = rng.random(len(pool))

    counts = bucket_counts(batch_size, bucket_ratio)
    selected_indices: list[int] = []
    selected_buckets = {}

    for bucket, n in counts.items():
        if n <= 0:
            continue

        acq_col = get_acq_col_for_bucket(bucket)

        chosen = greedy_select_from_sorted(
            pool_df=pool,
            x_pool=x_pool_transformed,
            labeled_df=labeled_df,
            selected_indices=selected_indices,
            n_select=n,
            acq_col=acq_col,
            max_samples_per_combo=max_samples_per_combo,
            min_batch_distance=min_batch_distance,
            rng=rng,
        )

        selected_indices.extend(chosen)
        for idx in chosen:
            selected_buckets[idx] = bucket

    # Fill any shortage using overall best mixed score.
    if len(selected_indices) < batch_size:
        pool["acq_mixed_fill"] = (
            pool["acq_boundary"]
            + pool["acq_pass_high_tmax"]
            + pool["acq_uncertainty_sparse"]
        ) / 3.0

        fill_chosen = greedy_select_from_sorted(
            pool_df=pool,
            x_pool=x_pool_transformed,
            labeled_df=labeled_df,
            selected_indices=selected_indices,
            n_select=batch_size - len(selected_indices),
            acq_col="acq_mixed_fill",
            max_samples_per_combo=max_samples_per_combo,
            min_batch_distance=min_batch_distance,
            rng=rng,
        )

        selected_indices.extend(fill_chosen)
        for idx in fill_chosen:
            selected_buckets[idx] = "fill_mixed"

    selected = pool.loc[selected_indices].copy().reset_index(drop=True)
    selected["selected_bucket"] = [
        selected_buckets.get(idx, "unknown") for idx in selected_indices
    ]

    selected.insert(0, "sampling_rank", range(1, len(selected) + 1))
    return selected
