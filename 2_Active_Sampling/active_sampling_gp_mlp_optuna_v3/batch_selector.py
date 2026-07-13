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

def far_enough_local(local_pool, idx, selected_idx, min_dist):
    if not selected_idx:
        return True
    x = local_pool[idx]
    y = local_pool[selected_idx]
    return bool(np.min(np.linalg.norm(y - x, axis=1)) >= min_dist)

def bin_label_for_value(v, bins, labels):
    for i in range(len(labels)):
        if bins[i] <= float(v) < bins[i + 1]:
            return labels[i]
    return None

def greedy(pool, x_pool, labeled_df, selected, n, col, max_per_combo, min_dist, rng, local_rule=None, bin_quota_rule=None, ptp_bound=None):
    chosen = []
    order = rng.permutation(pool.index.to_numpy()) if col == "random_score" else pool.sort_values(col, ascending=False).index.to_numpy()
    local_cols = None
    local_min_dist = None
    quota_col = None
    quota_bins = None
    quota_labels = None
    quota_by_label = None
    quota_counts = {}
    if isinstance(local_rule, dict):
        local_cols = local_rule.get("cols")
        local_min_dist = local_rule.get("min_dist")
    local_pool = None
    if local_cols and local_min_dist is not None:
        local_pool = pool.loc[:, local_cols].to_numpy(dtype=float)
    if isinstance(bin_quota_rule, dict):
        quota_col = bin_quota_rule.get("col")
        quota_bins = list(bin_quota_rule.get("bins", []))
        quota_labels = list(bin_quota_rule.get("labels", []))
        quota_by_label = dict(bin_quota_rule.get("quota_by_label", {}))
        quota_counts = {k: 0 for k in quota_by_label.keys()}
    ptp_min = None
    ptp_max = None
    if isinstance(ptp_bound, dict):
        ptp_min = ptp_bound.get("min")
        ptp_max = ptp_bound.get("max")

    # Keep incremental combo counts to avoid repeated value_counts in the hot loop.
    current_counts = labeled_df["discrete_combo_id"].value_counts().to_dict()
    if selected:
        for k, v in pool.loc[selected, "discrete_combo_id"].value_counts().to_dict().items():
            current_counts[k] = current_counts.get(k, 0) + int(v)

    for idx in order:
        if idx in selected or idx in chosen: continue
        cid = pool.at[idx, "discrete_combo_id"]
        if current_counts.get(cid, 0) >= max_per_combo: continue
        if "p_tp" in pool.columns:
            p = float(pool.at[idx, "p_tp"])
            if ptp_min is not None and p < float(ptp_min):
                continue
            if ptp_max is not None and p > float(ptp_max):
                continue
        quota_label = None
        if quota_col and quota_bins and quota_labels and quota_by_label:
            quota_label = bin_label_for_value(pool.at[idx, quota_col], quota_bins, quota_labels)
            if quota_label is None:
                continue
            if quota_counts.get(quota_label, 0) >= int(quota_by_label.get(quota_label, 0)):
                continue
        if not far_enough(x_pool[idx], selected+chosen, x_pool, min_dist): continue
        if local_pool is not None and not far_enough_local(local_pool, idx, selected+chosen, float(local_min_dist)):
            continue
        chosen.append(idx)
        if quota_label is not None:
            quota_counts[quota_label] = quota_counts.get(quota_label, 0) + 1
        current_counts[cid] = current_counts.get(cid, 0) + 1
        if len(chosen) >= n: break
    return chosen

def _detect_sparse_zones_in_barrier_thx(selected_indices, pool, full_range=None, n_bins=5, sparsity_threshold=0.15):
    """
    Analyze barrier_thx distribution in selected samples against a fixed full range.
    
    Args:
        full_range: (min, max) tuple. If None, uses pool's actual min/max.
                   Set to (0.25, 2.5) to cover entire physical range.
    
    Returns: list of zones with metadata including actual_count, required_count
    """
    if "C_Barrier_Thx" not in pool.columns or len(selected_indices) < 2:
        return []
    
    selected_barrier_thx = pool.loc[selected_indices, "C_Barrier_Thx"].values
    
    # Use fixed range or actual range
    if full_range is not None:
        min_val, max_val = full_range
    else:
        min_val, max_val = selected_barrier_thx.min(), selected_barrier_thx.max()
    
    if min_val == max_val:
        return []
    
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    zones = []
    total_count = len(selected_indices)
    
    for i in range(n_bins):
        bin_start, bin_end = bin_edges[i], bin_edges[i + 1]
        count = np.sum((selected_barrier_thx >= bin_start) & (selected_barrier_thx < bin_end))
        density = count / total_count if total_count > 0 else 0
        is_sparse = density < sparsity_threshold
        zone_label = f"{bin_start:.2f}-{bin_end:.2f}"
        zones.append({
            "label": zone_label,
            "bin_start": bin_start,
            "bin_end": bin_end,
            "actual_count": count,
            "density": density,
            "is_sparse": is_sparse
        })
    
    return zones

def _ensure_sparse_zone_coverage(selected, pool, x_pool, labeled_df, batch_size, min_batch_distance, rng, full_range=None, sparse_min_samples=3):
    """
    After initial greedy selection, ensure ALL zones (even empty ones) have minimum coverage.
    
    Uses fixed full_range to cover entire barrier_thx domain, not just selected sample range.
    """
    if "C_Barrier_Thx" not in pool.columns:
        return selected
    
    # Use full range or detect from pool
    if full_range is not None:
        min_val, max_val = full_range
    else:
        min_val, max_val = pool["C_Barrier_Thx"].min(), pool["C_Barrier_Thx"].max()
    
    if min_val == max_val:
        return selected
    
    n_bins = 5
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    zones = []
    
    # Analyze current coverage
    selected_barrier_thx = pool.loc[selected, "C_Barrier_Thx"].values
    total_count = len(selected)
    
    for i in range(n_bins):
        bin_start, bin_end = bin_edges[i], bin_edges[i + 1]
        count = np.sum((selected_barrier_thx >= bin_start) & (selected_barrier_thx < bin_end))
        density = count / total_count if total_count > 0 else 0
        is_sparse = (density < 0.15) or (count < sparse_min_samples)  # Either sparse OR insufficient
        
        zones.append({
            "bin_start": bin_start,
            "bin_end": bin_end,
            "actual_count": count,
            "is_sparse": is_sparse
        })
    
    selected_set = set(selected)
    
    # For each zone that needs more samples, add from that zone
    for zone in zones:
        if not zone["is_sparse"]:
            continue  # Already has enough
        
        needed = sparse_min_samples - zone["actual_count"]
        if needed <= 0:
            continue
        
        # Find candidates in this zone
        zone_mask = (pool["C_Barrier_Thx"] >= zone["bin_start"]) & (pool["C_Barrier_Thx"] < zone["bin_end"])
        zone_candidates = pool[zone_mask].index.tolist()
        zone_candidates = [idx for idx in zone_candidates if idx not in selected_set]
        
        if not zone_candidates:
            continue
        
        # Greedily add from this zone
        added_count = 0
        for idx in zone_candidates:
            if added_count >= needed:
                break
            if idx in selected_set:
                continue
            
            cid = pool.at[idx, "discrete_combo_id"]
            # Don't violate combo limit too much
            if len([s for s in selected if pool.at[s, "discrete_combo_id"] == cid]) >= 5:
                continue
            
            if far_enough(x_pool[idx], selected, x_pool, min_batch_distance):
                selected.append(idx)
                selected_set.add(idx)
                added_count += 1
    
    return selected

def select_batch(scored_pool, x_pool_transformed, labeled_df, batch_size, bucket_ratio, max_samples_per_combo, min_batch_distance, seed=42, bucket_distance_multiplier=None, bucket_local_distance_rules=None, bucket_bin_quota_rules=None, bucket_ptp_bounds=None, enable_sparse_coverage=True, sparse_min_samples=3, barrier_thx_full_range=None):
    rng = np.random.default_rng(seed)
    pool = scored_pool.copy().reset_index(drop=True)
    pool["random_score"] = rng.random(len(pool))
    bucket_distance_multiplier = bucket_distance_multiplier or {}
    bucket_local_distance_rules = bucket_local_distance_rules or {}
    bucket_bin_quota_rules = bucket_bin_quota_rules or {}
    bucket_ptp_bounds = bucket_ptp_bounds or {}
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
            bucket_bin_quota_rules.get(bucket),
            bucket_ptp_bounds.get(bucket),
        )
        selected += ch
        for i in ch: buckets[i] = bucket
    if len(selected) < batch_size:
        pool["acq_mixed_fill"] = (pool["acq_boundary"] + pool["acq_notp_high_tmax"] + pool["acq_uncertainty_sparse"])/3
        ch = greedy(pool, x_pool_transformed, labeled_df, selected, batch_size-len(selected), "acq_mixed_fill", max_samples_per_combo, min_batch_distance, rng)
        selected += ch
        for i in ch: buckets[i] = "fill_mixed"
    
    # Ensure sparse zone coverage with full range support
    if enable_sparse_coverage and len(selected) < batch_size:
        selected = _ensure_sparse_zone_coverage(selected, pool, x_pool_transformed, labeled_df, batch_size, min_batch_distance, rng, barrier_thx_full_range, sparse_min_samples)
    
    out = pool.loc[selected].copy().reset_index(drop=True)
    out["selected_bucket"] = [buckets.get(i, "unknown") for i in selected]
    out.insert(0, "sampling_rank", range(1, len(out)+1))
    
    return out
