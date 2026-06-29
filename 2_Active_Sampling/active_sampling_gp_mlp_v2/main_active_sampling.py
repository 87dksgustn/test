import json
import config
from data_loader import load_labeled_data, validate_required_columns, validate_passfail_labels
from discrete_space import generate_valid_discrete_combinations, attach_discrete_combo_id
from candidate_generator import generate_candidate_pool
from preprocessing import build_preprocessor, make_xy, make_extra_targets
from models_gp import fit_gp_models
from acquisition import compute_acquisition_scores
from batch_selector import select_batch
from diagnostics import combo_diagnostics
from evaluation import evaluate_gpc_cv
from model_selector import select_and_fit_model


def main():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_labeled_data(config.INPUT_CSV)
    validate_required_columns(df, config.CONTINUOUS_COLS, config.DISCRETE_COLS, config.PASSFAIL_COL, config.TMAX_COL, config.OTHER_REGRESSION_COLS, config.TIME_FEATURE_COLS)
    validate_passfail_labels(df, config.PASSFAIL_COL, config.PASS_LABEL, config.FAIL_LABEL)

    valid_combos = generate_valid_discrete_combinations(config.DISCRETE_LEVELS, config.DISCRETE_COLS, config.S_PREFIX)
    print(f"[INFO] Valid discrete combinations: {len(valid_combos)}")
    if len(valid_combos) != 28:
        print("[WARN] Valid combo count is not 28. Check DISCRETE_LEVELS and constraint logic.")

    df = attach_discrete_combo_id(df, valid_combos, config.DISCRETE_COLS)

    diag = combo_diagnostics(df, valid_combos, config.DISCRETE_COLS, config.PASSFAIL_COL, config.TMAX_COL, config.PASS_LABEL, config.FAIL_LABEL)
    diag.to_csv(config.OUTPUT_DIAGNOSTICS_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved combo diagnostics: {config.OUTPUT_DIAGNOSTICS_CSV}")

    x_raw, y_class, y_tmax = make_xy(df, config.CONTINUOUS_COLS, config.DISCRETE_COLS, config.PASSFAIL_COL, config.TMAX_COL)
    y_extra, extra_cols = make_extra_targets(df, config.OTHER_REGRESSION_COLS, config.TIME_FEATURE_COLS)

    preprocessor = build_preprocessor(config.CONTINUOUS_COLS, config.DISCRETE_COLS)
    x_train = preprocessor.fit_transform(x_raw)

    gp_cv = evaluate_gpc_cv(x_train, y_class, fail_label=config.FAIL_LABEL, n_splits=config.CV_SPLITS)
    print("[INFO] GP PASS/FAIL CV result:")
    print(json.dumps(gp_cv, indent=2, ensure_ascii=False))

    gp_models = fit_gp_models(x_train, y_class, y_tmax, pass_label=config.PASS_LABEL, random_state=config.RANDOM_SEED)

    selected_model, report = select_and_fit_model(df, x_train, y_class, y_tmax, y_extra, extra_cols, gp_models, gp_cv, config)
    with open(config.OUTPUT_MODEL_SELECTION_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("[INFO] Model selection report:")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[INFO] Selected model kind: {getattr(selected_model, 'kind', 'gp')}")

    candidate_pool = generate_candidate_pool(valid_combos, config.CONTINUOUS_COLS, config.CONTINUOUS_BOUNDS, config.DISCRETE_COLS, config.CANDIDATES_PER_COMBO, config.EXCLUDED_REFERENCE_RANGES, config.RANDOM_SEED)
    print(f"[INFO] Candidate pool size after exclusion filter: {len(candidate_pool)}")

    x_cand_raw = candidate_pool[config.CONTINUOUS_COLS + config.DISCRETE_COLS].copy()
    x_cand = preprocessor.transform(x_cand_raw)

    scored = compute_acquisition_scores(candidate_pool, df, x_cand, x_train, selected_model, config)
    scored.sort_values("acq_boundary", ascending=False).head(2000).to_csv(config.OUTPUT_SCORED_POOL_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved scored candidate preview: {config.OUTPUT_SCORED_POOL_CSV}")

    selected = select_batch(scored, x_cand, df, config.BATCH_SIZE, config.BUCKET_RATIO, config.MAX_SAMPLES_PER_COMBO, config.MIN_BATCH_DISTANCE, config.RANDOM_SEED)
    front = ["sampling_rank", "selected_bucket", "selected_model_kind"] + config.CONTINUOUS_COLS + config.DISCRETE_COLS + ["discrete_combo_id"]
    scores = ["p_fail", "p_pass", "boundary_score", "clf_uncertainty_raw", "clf_uncertainty_scaled", "tmax_pred_given_pass", "tmax_std_given_pass", "pass_window_score", "local_sparsity", "combo_priority", "acq_boundary", "acq_pass_high_tmax", "acq_uncertainty_sparse"]
    selected = selected[front + scores]
    selected.to_csv(config.OUTPUT_CANDIDATES_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved next sampling candidates: {config.OUTPUT_CANDIDATES_CSV}")
    print(f"[INFO] Selected count: {len(selected)}")
    print(selected[["sampling_rank", "selected_bucket", "selected_model_kind", "discrete_combo_id", "p_fail", "p_pass", "tmax_pred_given_pass"]].head(20))


if __name__ == "__main__":
    main()
