import json
import logging
import pandas as pd
import config
from data_loader import load_labeled_data, validate_required_columns, validate_passfail_labels
from discrete_space import generate_valid_discrete_combinations, attach_discrete_combo_id
from candidate_generator import generate_candidate_pool
from preprocessing import build_preprocessor, make_xy, make_extra_targets
from diagnostics import combo_diagnostics
from optuna_tuning import maybe_tune_models
from model_selector import select_and_fit_model
from evaluation import fold_metrics_to_df
from acquisition import compute_acquisition_scores
from batch_selector import select_batch

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

def main():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_labeled_data(config.INPUT_CSV)
    validate_required_columns(df, config.CONTINUOUS_COLS, config.DISCRETE_COLS, config.TPNoTP_COL, config.TMAX_COL, config.OTHER_REGRESSION_COLS, config.TIME_FEATURE_COLS)
    validate_passfail_labels(df, config.TPNoTP_COL, config.PASS_LABEL, config.FAIL_LABEL)
    valid_combos = generate_valid_discrete_combinations(config.DISCRETE_LEVELS, config.DISCRETE_COLS, config.S_PREFIX)
    print(f"[INFO] Valid discrete combinations: {len(valid_combos)}")
    if len(valid_combos) != 28: print("[WARN] Valid combo count is not 28. Check DISCRETE_LEVELS and constraint logic.")
    df = attach_discrete_combo_id(df, valid_combos, config.DISCRETE_COLS)
    diag = combo_diagnostics(df, valid_combos, config.DISCRETE_COLS, config.TPNoTP_COL, config.TMAX_COL, config.PASS_LABEL, config.FAIL_LABEL)
    diag.to_csv(config.OUTPUT_DIAGNOSTICS_CSV, index=False, encoding="utf-8-sig")
    x_raw, y_class, y_tmax = make_xy(df, config.CONTINUOUS_COLS, config.DISCRETE_COLS, config.TPNoTP_COL, config.TMAX_COL)
    y_extra, extra_cols = make_extra_targets(df, config.OTHER_REGRESSION_COLS, config.TIME_FEATURE_COLS)
    pre = build_preprocessor(config.CONTINUOUS_COLS, config.DISCRETE_COLS)
    x_train = pre.fit_transform(x_raw)
    tuned = maybe_tune_models(df, x_train, y_class, y_tmax, y_extra, config)
    with open(config.OUTPUT_OPTUNA_REPORT_JSON, "w", encoding="utf-8") as f: json.dump(tuned["report"], f, indent=2, ensure_ascii=False)
    print("[INFO] Optuna report:"); print(json.dumps(tuned["report"], indent=2, ensure_ascii=False))
    selected_model, selection_report, fold_results = select_and_fit_model(df, x_train, y_class, y_tmax, y_extra, config, tuned)
    with open(config.OUTPUT_MODEL_SELECTION_JSON, "w", encoding="utf-8") as f: json.dump(selection_report, f, indent=2, ensure_ascii=False)
    fold_df = fold_metrics_to_df(fold_results)
    if len(fold_df): fold_df.to_csv(config.OUTPUT_CV_FOLD_METRICS_CSV, index=False, encoding="utf-8-sig")
    print("[INFO] Model selection report:"); print(json.dumps(selection_report, indent=2, ensure_ascii=False))
    print(f"[INFO] Selected model kind: {getattr(selected_model, 'kind', 'gp')}")
    pool = generate_candidate_pool(valid_combos, config.CONTINUOUS_COLS, config.CONTINUOUS_BOUNDS, config.DISCRETE_COLS, config.CANDIDATES_PER_COMBO, config.EXCLUDED_REFERENCE_RANGES, config.RANDOM_SEED)
    print(f"[INFO] Candidate pool size after exclusion filter: {len(pool)}")
    x_candidate = pre.transform(pool[config.CONTINUOUS_COLS + config.DISCRETE_COLS].copy())
    scored = compute_acquisition_scores(pool, df, x_candidate, x_train, selected_model, config)
    scored.sort_values("acq_boundary", ascending=False).head(2000).to_csv(config.OUTPUT_SCORED_POOL_CSV, index=False, encoding="utf-8-sig")
    selected = select_batch(scored, x_candidate, df, config.BATCH_SIZE, config.BUCKET_RATIO, config.MAX_SAMPLES_PER_COMBO, config.MIN_BATCH_DISTANCE, config.RANDOM_SEED)
    front = ["sampling_rank", "selected_bucket", "selected_model_kind"] + config.CONTINUOUS_COLS + config.DISCRETE_COLS + ["discrete_combo_id"]
    score_cols = ["p_tp", "p_notp", "boundary_score", "clf_uncertainty_raw", "clf_uncertainty_scaled", "tmax_pred_given_notp", "tmax_std_given_notp", "notp_window_score", "local_sparsity", "combo_priority", "acq_boundary", "acq_notp_high_tmax", "acq_uncertainty_sparse"]
    selected = selected[front + score_cols]
    selected.to_csv(config.OUTPUT_CANDIDATES_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved next sampling candidates: {config.OUTPUT_CANDIDATES_CSV}")
    print(selected[["sampling_rank", "selected_bucket", "selected_model_kind", "discrete_combo_id", "p_tp", "p_notp", "tmax_pred_given_notp"]].head(20))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)