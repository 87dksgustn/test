import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

import config
from data_loader import load_labeled_data
from discrete_space import attach_discrete_combo_id, generate_valid_discrete_combinations


TOTAL_TEST_CASES = 100
BOUNDARY_TEST_CASES = 70
COVERAGE_TEST_CASES = TOTAL_TEST_CASES - BOUNDARY_TEST_CASES
BOUNDARY_POOL_MULTIPLIER = 30
BOUNDARY_NEIGHBORS = 5
MIN_COMBO_SAMPLES_FOR_BOUNDARY = 4
NORMALIZED_MIN_DISTANCE = 0.10
OUTPUT_SUBDIR = Path("final_test_doe")


def filter_excluded_reference_ranges(df, excluded_reference_ranges):
    if not excluded_reference_ranges:
        return df.reset_index(drop=True)
    mask = pd.Series(True, index=df.index)
    for col, info in excluded_reference_ranges.items():
        center = float(info["center"])
        half_width = float(info["half_width"])
        mask &= ~df[col].between(center - half_width, center + half_width, inclusive="both")
    return df.loc[mask].reset_index(drop=True)


def generate_lhs_samples(n_samples, continuous_cols, continuous_bounds, seed):
    sampler = qmc.LatinHypercube(d=len(continuous_cols), seed=seed)
    x_unit = sampler.random(n=n_samples)
    lows = np.array([continuous_bounds[col][0] for col in continuous_cols], dtype=float)
    highs = np.array([continuous_bounds[col][1] for col in continuous_cols], dtype=float)
    return pd.DataFrame(qmc.scale(x_unit, lows, highs), columns=continuous_cols)


def make_combo_pool(combo_row, n_samples, continuous_cols, continuous_bounds, discrete_cols, excluded_reference_ranges, seed):
    cont = generate_lhs_samples(n_samples, continuous_cols, continuous_bounds, seed)
    for col in discrete_cols:
        cont[col] = combo_row[col]
    cont["discrete_combo_id"] = combo_row["discrete_combo_id"]
    return filter_excluded_reference_ranges(cont, excluded_reference_ranges)


def normalized_continuous_matrix(df, continuous_cols, continuous_bounds):
    lows = np.array([continuous_bounds[col][0] for col in continuous_cols], dtype=float)
    highs = np.array([continuous_bounds[col][1] for col in continuous_cols], dtype=float)
    denom = np.where(highs > lows, highs - lows, 1.0)
    values = df[continuous_cols].to_numpy(dtype=float)
    return (values - lows) / denom


def boundary_score_candidates(candidate_df, reference_df, continuous_cols, continuous_bounds, passfail_col, tp_label, k_neighbors):
    if len(reference_df) < max(MIN_COMBO_SAMPLES_FOR_BOUNDARY, 2):
        return np.zeros(len(candidate_df), dtype=float)

    ref_x = normalized_continuous_matrix(reference_df, continuous_cols, continuous_bounds)
    cand_x = normalized_continuous_matrix(candidate_df, continuous_cols, continuous_bounds)
    ref_y = reference_df[passfail_col].astype(int).to_numpy()
    k = max(1, min(k_neighbors, len(reference_df)))

    scores = []
    for row in cand_x:
        dists = np.linalg.norm(ref_x - row, axis=1)
        nn_idx = np.argsort(dists)[:k]
        tp_rate = float(np.mean(ref_y[nn_idx] == tp_label))
        boundary_score = 1.0 - 2.0 * abs(tp_rate - 0.5)
        scores.append(np.clip(boundary_score, 0.0, 1.0))
    return np.asarray(scores, dtype=float)


def combo_boundary_weight(reference_df, passfail_col, tp_label):
    if len(reference_df) == 0:
        return 0.0
    tp_rate = float(np.mean(reference_df[passfail_col].astype(int).to_numpy() == tp_label))
    return float(np.clip(1.0 - 2.0 * abs(tp_rate - 0.5), 0.0, 1.0))


def allocate_counts(total, combo_ids, combo_weights):
    base = total // len(combo_ids)
    rem = total % len(combo_ids)
    out = {combo_id: base for combo_id in combo_ids}
    if rem <= 0:
        return out
    order = sorted(combo_ids, key=lambda combo_id: (-combo_weights.get(combo_id, 0.0), combo_id))
    for combo_id in order[:rem]:
        out[combo_id] += 1
    return out


def remove_duplicates(candidate_df, existing_df, continuous_cols, discrete_cols):
    existing_keys = set()
    for _, row in existing_df[continuous_cols + discrete_cols].iterrows():
        key = tuple(np.round(row[continuous_cols].to_numpy(dtype=float), 10).tolist()) + tuple(row[discrete_cols].astype(str).tolist())
        existing_keys.add(key)

    kept_rows = []
    seen = set(existing_keys)
    for _, row in candidate_df.iterrows():
        key = tuple(np.round(row[continuous_cols].to_numpy(dtype=float), 10).tolist()) + tuple(row[discrete_cols].astype(str).tolist())
        if key in seen:
            continue
        seen.add(key)
        kept_rows.append(row)
    if not kept_rows:
        return candidate_df.iloc[0:0].copy()
    return pd.DataFrame(kept_rows).reset_index(drop=True)


def greedy_pick_with_distance(pool_df, score_col, n_pick, continuous_cols, continuous_bounds, min_distance):
    if len(pool_df) == 0 or n_pick <= 0:
        return pool_df.iloc[0:0].copy()

    norm_x = normalized_continuous_matrix(pool_df, continuous_cols, continuous_bounds)
    ordered_idx = pool_df.sort_values(score_col, ascending=False).index.to_list()
    chosen = []
    chosen_norm = []

    for idx in ordered_idx:
        row_norm = norm_x[pool_df.index.get_loc(idx)]
        if chosen_norm:
            min_dist = float(np.min(np.linalg.norm(np.vstack(chosen_norm) - row_norm, axis=1)))
            if min_dist < min_distance:
                continue
        chosen.append(idx)
        chosen_norm.append(row_norm)
        if len(chosen) >= n_pick:
            break

    if len(chosen) < n_pick:
        for idx in ordered_idx:
            if idx in chosen:
                continue
            chosen.append(idx)
            if len(chosen) >= n_pick:
                break

    return pool_df.loc[chosen].reset_index(drop=True)


def build_test_dataset(base_df, valid_combos):
    combo_weights = {}
    combo_frames = {}
    for combo_id in valid_combos["discrete_combo_id"]:
        combo_ref = base_df.loc[base_df["discrete_combo_id"] == combo_id].copy()
        combo_frames[combo_id] = combo_ref
        combo_weights[combo_id] = combo_boundary_weight(combo_ref, config.TPNoTP_COL, config.TP_LABEL)

    combo_ids = valid_combos["discrete_combo_id"].tolist()
    boundary_alloc = allocate_counts(BOUNDARY_TEST_CASES, combo_ids, combo_weights)
    coverage_alloc = allocate_counts(COVERAGE_TEST_CASES, combo_ids, combo_weights)

    selected_parts = []
    seed_cursor = int(config.RANDOM_SEED)

    for _, combo_row in valid_combos.iterrows():
        combo_id = combo_row["discrete_combo_id"]
        combo_ref = combo_frames[combo_id]

        boundary_need = int(boundary_alloc[combo_id])
        boundary_pool = make_combo_pool(
            combo_row,
            max(boundary_need * BOUNDARY_POOL_MULTIPLIER, boundary_need),
            config.CONTINUOUS_COLS,
            config.CONTINUOUS_BOUNDS,
            config.DISCRETE_COLS,
            config.EXCLUDED_REFERENCE_RANGES,
            seed_cursor,
        )
        seed_cursor += 1009
        boundary_pool = remove_duplicates(boundary_pool, base_df, config.CONTINUOUS_COLS, config.DISCRETE_COLS)
        boundary_pool["test_bucket"] = "boundary_operational"
        boundary_pool["boundary_proxy_score"] = boundary_score_candidates(
            boundary_pool,
            combo_ref,
            config.CONTINUOUS_COLS,
            config.CONTINUOUS_BOUNDS,
            config.TPNoTP_COL,
            config.TP_LABEL,
            BOUNDARY_NEIGHBORS,
        )
        boundary_selected = greedy_pick_with_distance(
            boundary_pool,
            "boundary_proxy_score",
            boundary_need,
            config.CONTINUOUS_COLS,
            config.CONTINUOUS_BOUNDS,
            NORMALIZED_MIN_DISTANCE,
        )
        selected_parts.append(boundary_selected)

        coverage_need = int(coverage_alloc[combo_id])
        coverage_pool = make_combo_pool(
            combo_row,
            max(coverage_need * 4, coverage_need),
            config.CONTINUOUS_COLS,
            config.CONTINUOUS_BOUNDS,
            config.DISCRETE_COLS,
            config.EXCLUDED_REFERENCE_RANGES,
            seed_cursor,
        )
        seed_cursor += 1009
        existing_for_coverage = pd.concat([base_df] + selected_parts, ignore_index=True) if selected_parts else base_df
        coverage_pool = remove_duplicates(coverage_pool, existing_for_coverage, config.CONTINUOUS_COLS, config.DISCRETE_COLS)
        coverage_pool["test_bucket"] = "coverage_sanity"
        coverage_pool["boundary_proxy_score"] = boundary_score_candidates(
            coverage_pool,
            combo_ref,
            config.CONTINUOUS_COLS,
            config.CONTINUOUS_BOUNDS,
            config.TPNoTP_COL,
            config.TP_LABEL,
            BOUNDARY_NEIGHBORS,
        )
        coverage_pool["coverage_random_score"] = np.random.default_rng(seed_cursor).random(len(coverage_pool))
        coverage_selected = greedy_pick_with_distance(
            coverage_pool,
            "coverage_random_score",
            coverage_need,
            config.CONTINUOUS_COLS,
            config.CONTINUOUS_BOUNDS,
            NORMALIZED_MIN_DISTANCE,
        )
        selected_parts.append(coverage_selected)

    final_df = pd.concat(selected_parts, ignore_index=True)
    final_df.insert(0, "final_test_rank", np.arange(1, len(final_df) + 1))
    return final_df, boundary_alloc, coverage_alloc, combo_weights


def save_outputs(final_df, boundary_alloc, coverage_alloc, combo_weights):
    output_dir = config.OUTPUT_DIR / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_csv = output_dir / "final_test_dataset_100.csv"
    summary_csv = output_dir / "final_test_combo_summary.csv"
    report_json = output_dir / "final_test_generation_report.json"

    final_df.to_csv(dataset_csv, index=False, encoding="utf-8-sig")

    combo_summary = (
        final_df.groupby(["discrete_combo_id", *config.DISCRETE_COLS, "test_bucket"]).size().unstack(fill_value=0).reset_index()
    )
    combo_summary["boundary_alloc_target"] = combo_summary["discrete_combo_id"].map(boundary_alloc)
    combo_summary["coverage_alloc_target"] = combo_summary["discrete_combo_id"].map(coverage_alloc)
    combo_summary["boundary_weight"] = combo_summary["discrete_combo_id"].map(combo_weights)
    combo_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    report = {
        "total_cases": int(len(final_df)),
        "boundary_cases": int((final_df["test_bucket"] == "boundary_operational").sum()),
        "coverage_cases": int((final_df["test_bucket"] == "coverage_sanity").sum()),
        "continuous_bounds": config.CONTINUOUS_BOUNDS,
        "excluded_reference_ranges": config.EXCLUDED_REFERENCE_RANGES,
        "discrete_levels": config.DISCRETE_LEVELS,
        "boundary_pool_multiplier": BOUNDARY_POOL_MULTIPLIER,
        "boundary_neighbors": BOUNDARY_NEIGHBORS,
        "normalized_min_distance": NORMALIZED_MIN_DISTANCE,
        "combo_boundary_alloc": boundary_alloc,
        "combo_coverage_alloc": coverage_alloc,
    }
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return dataset_csv, summary_csv, report_json


def main():
    base_df = load_labeled_data(config.INPUT_CSV)
    valid_combos = generate_valid_discrete_combinations(config.DISCRETE_LEVELS, config.DISCRETE_COLS, config.S_PREFIX)
    base_df = attach_discrete_combo_id(base_df, valid_combos, config.DISCRETE_COLS)

    final_df, boundary_alloc, coverage_alloc, combo_weights = build_test_dataset(base_df, valid_combos)
    dataset_csv, summary_csv, report_json = save_outputs(final_df, boundary_alloc, coverage_alloc, combo_weights)

    print(f"[INFO] Final test DOE saved: {dataset_csv}")
    print(f"[INFO] Combo summary saved: {summary_csv}")
    print(f"[INFO] Generation report saved: {report_json}")
    print(f"[INFO] Final test size: {len(final_df)}")
    print(final_df[["final_test_rank", "test_bucket", "discrete_combo_id", *config.CONTINUOUS_COLS]].head(10))


if __name__ == "__main__":
    main()