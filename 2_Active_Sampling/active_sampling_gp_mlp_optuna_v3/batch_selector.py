import math
import numpy as np

def bucket_counts(batch_size, bucket_ratio):
    total = sum(bucket_ratio.values())
    raw = {k: batch_size*v/total for k,v in bucket_ratio.items()}
    counts = {k: int(math.floor(v)) for k,v in raw.items()}
    rem = batch_size - sum(counts.values())
    order = sorted(raw.keys(), key=lambda k: raw[k]-counts[k], reverse=True)
    for k in order[:rem]: counts[k] += 1
    return counts

def acq_col(bucket):
    return {"boundary":"acq_boundary", "notp_high_tmax":"acq_notp_high_tmax", "uncertainty_sparse":"acq_uncertainty_sparse", "random_check":"random_score"}[bucket]

def combo_counts_after(labeled_df, selected_df):
    d = labeled_df["discrete_combo_id"].value_counts().to_dict()
    if len(selected_df):
        for k,v in selected_df["discrete_combo_id"].value_counts().to_dict().items(): d[k] = d.get(k,0)+v
    return d

def far_enough(x, selected_idx, x_pool, min_dist):
    if not selected_idx: return True
    return bool(np.min(np.linalg.norm(x_pool[selected_idx] - x, axis=1)) >= min_dist)

def far_enough_local(pool, idx, selected_idx, cols, min_dist):
    if not selected_idx:
        return True
    x = pool.loc[idx, cols].to_numpy(dtype=float)
    y = pool.loc[selected_idx, cols].to_numpy(dtype=float)
    return bool(np.min(np.linalg.norm(y - x, axis=1)) >= min_dist)

def greedy(pool, x_pool, labeled_df, selected, n, col, max_per_combo, min_dist, rng, local_rule=None):
    chosen = []
    order = rng.permutation(pool.index.to_numpy()) if col == "random_score" else pool.sort_values(col, ascending=False).index.to_numpy()
    local_cols = None
    local_min_dist = None
    if isinstance(local_rule, dict):
        local_cols = local_rule.get("cols")
        local_min_dist = local_rule.get("min_dist")
    for idx in order:
        if idx in selected or idx in chosen: continue
        current = combo_counts_after(labeled_df, pool.loc[selected+chosen] if selected or chosen else pool.iloc[0:0])
        if current.get(pool.at[idx, "discrete_combo_id"], 0) >= max_per_combo: continue
        if not far_enough(x_pool[idx], selected+chosen, x_pool, min_dist): continue
        if local_cols and local_min_dist is not None and not far_enough_local(pool, idx, selected+chosen, local_cols, float(local_min_dist)):
            continue
        chosen.append(idx)
        if len(chosen) >= n: break
    return chosen

def select_batch(scored_pool, x_pool_transformed, labeled_df, batch_size, bucket_ratio, max_samples_per_combo, min_batch_distance, seed=42, bucket_distance_multiplier=None, bucket_local_distance_rules=None):
    rng = np.random.default_rng(seed)
    pool = scored_pool.copy().reset_index(drop=True)
    pool["random_score"] = rng.random(len(pool))
    bucket_distance_multiplier = bucket_distance_multiplier or {}
    bucket_local_distance_rules = bucket_local_distance_rules or {}
    selected = []; buckets = {}
    for bucket, n in bucket_counts(batch_size, bucket_ratio).items():
        min_dist_for_bucket = float(min_batch_distance) * float(bucket_distance_multiplier.get(bucket, 1.0))
        ch = greedy(
            pool,
            x_pool_transformed,
            labeled_df,
            selected,
            n,
            acq_col(bucket),
            max_samples_per_combo,
            min_dist_for_bucket,
            rng,
            bucket_local_distance_rules.get(bucket),
        )
        selected += ch
        for i in ch: buckets[i] = bucket
    if len(selected) < batch_size:
        pool["acq_mixed_fill"] = (pool["acq_boundary"] + pool["acq_notp_high_tmax"] + pool["acq_uncertainty_sparse"])/3
        ch = greedy(pool, x_pool_transformed, labeled_df, selected, batch_size-len(selected), "acq_mixed_fill", max_samples_per_combo, min_batch_distance, rng)
        selected += ch
        for i in ch: buckets[i] = "fill_mixed"
    out = pool.loc[selected].copy().reset_index(drop=True)
    out["selected_bucket"] = [buckets.get(i, "unknown") for i in selected]
    out.insert(0, "sampling_rank", range(1, len(out)+1))
    return out
