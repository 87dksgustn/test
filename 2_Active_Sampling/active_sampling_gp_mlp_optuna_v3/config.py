from pathlib import Path

# ============================================================
# User-editable configuration
# ============================================================

INPUT_CSV = "Itr_1_dataset.csv"

CONTINUOUS_COLS = ["A_Cell_D", "C_Barrier_Thx", "E_Barrier_Outer_Thx", "F_ThermalResin_Thx"]
DISCRETE_COLS = ["B_Barrier_Type", "D_Barrier_Outer_Type"]

PASSFAIL_COL = "TP_NoTP"
TPNoTP_COL = PASSFAIL_COL
TMAX_COL = "MaxT_Adj"        # Valid mainly for NoTP cases

# Optional extra outputs. They can be trained by MLP multi-head,
# but are not used as a separate sampling bucket by default.
OTHER_REGRESSION_COLS = []
TIME_FEATURE_COLS = []

TP_LABEL = 1
NOTP_LABEL = 0
# Internal pass/fail pipeline semantics are remapped as NoTP/TP.
PASS_LABEL = NOTP_LABEL
FAIL_LABEL = TP_LABEL

CONTINUOUS_BOUNDS = {
    "A_Cell_D": (8.0, 16.0),
    "C_Barrier_Thx": (0.25, 2.5),
    "E_Barrier_Outer_Thx": (1.1, 3.0),
    "F_ThermalResin_Thx": (0.5, 2.5),
}

# Applied only to newly generated candidate points, not existing CFD data.
EXCLUDED_REFERENCE_RANGES = {
    "A_Cell_D": {"center": 12.4, "half_width": 0.01},
    "C_Barrier_Thx": {"center": 0.85, "half_width": 0.01},
    "E_Barrier_Outer_Thx": {"center": 2.0, "half_width": 0.01},
    "F_ThermalResin_Thx": {"center": 1.0, "half_width": 0.01},
}

DISCRETE_LEVELS = {
    "B_Barrier_Type": ["Si1", "Si2", "Si3", "A1", "A2"],
    "D_Barrier_Outer_Type": ["PU", "Si1", "Si2", "Si3"],
    # "I_Cell_Barrier": ["1CP", "2CP"],
}
S_PREFIX = "S"

CURRENT_LEVEL_TARGET_TOTAL = 636
INITIAL_TOTAL = 140
ADDITIONAL_TOTAL = CURRENT_LEVEL_TARGET_TOTAL - INITIAL_TOTAL

BATCH_SIZE = 23
CANDIDATES_PER_COMBO = 3000

MIN_SAMPLES_PER_COMBO = 8
MAX_SAMPLES_PER_COMBO = 40

BUCKET_RATIO = {
    "boundary": 0.75,
    "notp_high_tmax": 0.15,
    "uncertainty_sparse": 0.07,
    "random_check": 0.03,
}

BOUNDARY_WEIGHTS_GP = {
    "boundary": 0.70,
    "clf_uncertainty": 0.00,
    "local_sparsity": 0.15,
    "combo_priority": 0.15,
}

# GP classifier uncertainty mode:
# - "none": keep legacy behavior (clf_uncertainty=0 for GP path)
# - "ensemble_std": train bootstrap GP classifiers and use std of p_tp
GP_CLF_UNCERTAINTY_MODE = "ensemble_std"
GP_CLF_ENSEMBLE_SIZE = 5
GP_CLF_ENSEMBLE_SAMPLE_RATIO = 0.8
GP_CLF_ENSEMBLE_STRATIFIED = True

# Adaptive hybrid weighting for boundary bucket.
# Data-driven mode: boundary classifier uncertainty weight is determined
# only by current coverage readiness (no Try-index schedule).
ENABLE_ADAPTIVE_BOUNDARY_HYBRID = True
ADAPTIVE_BOUNDARY_CLF_UNC_MIN = 0.00
ADAPTIVE_BOUNDARY_CLF_UNC_MAX = 0.15
ADAPTIVE_BOUNDARY_COVERAGE_FULL_MIN_NORM_RANGE = 0.90
ADAPTIVE_BOUNDARY_COVERAGE_FULL_MEAN_NORM_RANGE = 0.93

BOUNDARY_WEIGHTS_MLP = {
    "boundary": 0.55,
    "clf_uncertainty": 0.15,
    "local_sparsity": 0.15,
    "combo_priority": 0.15,
}

NOTP_HIGH_TMAX_WEIGHTS = {
    "tmax": 0.70,  # Ver3: notp_window 제거분 이관 (0.40 → 0.70)
    "notp_window": 0.00,  # Ver3: 0.25 집중 무의미, 제거
    "tmax_uncertainty": 0.10,
    "local_sparsity": 0.10,
    "combo_priority": 0.10,
}

UNCERTAINTY_SPARSE_WEIGHTS = {
    "clf_uncertainty": 0.35,
    "tmax_uncertainty": 0.35,
    "local_sparsity": 0.15,
    "combo_priority": 0.15,
}

NOTP_WINDOW_LOW = 0.60
NOTP_WINDOW_HIGH = 0.90
NOTP_WINDOW_CENTER = 0.75
MIN_BATCH_DISTANCE = 0.12
# Apply stricter spacing for specific buckets during selection.
BUCKET_DISTANCE_MULTIPLIER = {
    "notp_high_tmax": 3.0,
}
BUCKET_LOCAL_DISTANCE_RULES = {
    "boundary": {
        "cols": ["A_Cell_D", "C_Barrier_Thx", "E_Barrier_Outer_Thx", "F_ThermalResin_Thx"],
        "min_dist": 0.10,
    },
    "notp_high_tmax": {
        "cols": ["A_Cell_D", "C_Barrier_Thx", "E_Barrier_Outer_Thx", "F_ThermalResin_Thx"],
        "min_dist": 0.22,
    }
}

# Optional hard p_tp bounds by bucket.
# Candidates outside each bucket range are skipped during greedy selection.
BUCKET_PTP_BOUNDS = {
    "boundary": {"min": 0.45, "max": 0.55},
    "notp_high_tmax": {"min": 0.10, "max": 0.50},
}

# Dynamic quota for notp_high_tmax by Cell_D bins.
# At each run, quota is recomputed from current labeled data as:
#   TP_NoTP==TP_LABEL count per Cell_D bin / total TP_NoTP==TP_LABEL count
NOTP_HIGHTMAX_USE_TP1_CELLD_RATIO = True
NOTP_HIGHTMAX_CELLD_COL = "A_Cell_D"
NOTP_HIGHTMAX_CELLD_BINS = [8.0, 10.0, 12.0, 14.0, 16.000001]
NOTP_HIGHTMAX_CELLD_BIN_LABELS = ["8-10", "10-12", "12-14", "14-16"]

# Bucket Cell_D quota mode:
# - "hybrid_baseline": baseline-centered fixed zone ratios (recommended for operational baseline around 12.4)
# - "tp_ratio_only": TP_NoTP==1 ratio-only quota by Cell_D bins
# - "off": disable bucket bin quotas
BUCKET_CELLD_QUOTA_MODE = "tp_ratio_only"

# Common TP-ratio quota settings (used in tp_ratio_only mode)
TP_RATIO_CELLD_COL = "A_Cell_D"
TP_RATIO_CELLD_BINS = [8.0, 10.0, 12.0, 14.0, 16.000001]
TP_RATIO_CELLD_BIN_LABELS = ["8-10", "10-12", "12-14", "14-16"]

# Hybrid baseline-centered quota settings
BASELINE_CELLD = 12.4
HYBRID_CELLD_COL = "A_Cell_D"
HYBRID_CELLD_BINS = [8.0, 11.8, 13.0, 16.000001]
HYBRID_CELLD_BIN_LABELS = ["low(8-11.8)", "near(11.8-13.0)", "high(13-16)"]
HYBRID_BOUNDARY_ZONE_RATIO = {
    "low(8-11.8)": 4,
    "near(11.8-13.0)": 8,
    "high(13-16)": 5,
}
HYBRID_NOTP_HIGHTMAX_ZONE_RATIO = {
    "low(8-11.8)": 2,
    "near(11.8-13.0)": 4,
    "high(13-16)": 2,
}

# ============================================================
# Model selection
# ============================================================
# "gp"   : force GP
# "mlp"  : force MLP if eligible, otherwise fallback to GP
# "auto" : compare GP and MLP if MLP is eligible
MODEL_MODE = "auto"

MLP_MIN_TOTAL_SAMPLES = 350
MLP_MIN_CLASS_RATIO = 0.30
MLP_MIN_PASS_SAMPLES = int(MLP_MIN_TOTAL_SAMPLES * MLP_MIN_CLASS_RATIO)
MLP_MIN_FAIL_SAMPLES = int(MLP_MIN_TOTAL_SAMPLES * MLP_MIN_CLASS_RATIO)
MLP_MIN_SAMPLES_PER_COMBO = 8

MODEL_SELECTION_WEIGHTS = {
    "tp_recall": 0.70,
    "tp_f1": 0.30,
}
MLP_SELECTION_MARGIN = 0.01

# Model comparison summary score (for GP vs MLP report/plot only).
# composite_score = MODEL_COMPARE_CV_WEIGHT * cv_stable_score + MODEL_COMPARE_HOLDOUT_WEIGHT * holdout_weighted_score
MODEL_COMPARE_CV_WEIGHT = 0.30
MODEL_COMPARE_HOLDOUT_WEIGHT = 0.70
MODEL_COMPARE_HOLDOUT_TEST_SIZE = 0.20

# ============================================================
# CV stabilization
# ============================================================
CV_SPLITS = 5
# stable_score = weighted_mean - CV_STD_PENALTY * weighted_std
CV_STD_PENALTY = 0.50

# ============================================================
# Optuna auto-tuning conditions
# ============================================================
ENABLE_OPTUNA_AUTO = True

# If Optuna is not installed, code automatically skips tuning.
# pip install optuna

GP_OPTUNA_MIN_TOTAL_SAMPLES = 100
GP_OPTUNA_MIN_PASS_SAMPLES = 20
GP_OPTUNA_MIN_FAIL_SAMPLES = 20
GP_OPTUNA_N_TRIALS = 30
GP_OPTUNA_TIMEOUT_SEC = None

TMAX_OPTUNA_MIN_PASS_SAMPLES = 50
TMAX_OPTUNA_N_TRIALS = 25
TMAX_OPTUNA_TIMEOUT_SEC = None

MLP_OPTUNA_MIN_TOTAL_SAMPLES = 350
MLP_OPTUNA_MIN_CLASS_RATIO = 0.30
MLP_OPTUNA_MIN_PASS_SAMPLES = int(MLP_OPTUNA_MIN_TOTAL_SAMPLES * MLP_OPTUNA_MIN_CLASS_RATIO)
MLP_OPTUNA_MIN_FAIL_SAMPLES = int(MLP_OPTUNA_MIN_TOTAL_SAMPLES * MLP_OPTUNA_MIN_CLASS_RATIO)
MLP_OPTUNA_N_TRIALS = 20
MLP_OPTUNA_TIMEOUT_SEC = None

# Hard-gate + classification-first objective for MLP Optuna.
# Trials failing classification gates receive a very low objective score.
MLP_OPTUNA_GATE_MIN_TP_RECALL = 0.20
MLP_OPTUNA_GATE_MIN_TP_F1 = 0.15
# Optional regression gates; set to None to disable.
MLP_OPTUNA_GATE_MAX_TMAX_RMSE = None
MLP_OPTUNA_GATE_MIN_TMAX_R2 = None
# Small tie-break weight among gate-passing trials (classification remains dominant).
MLP_OPTUNA_TMAX_TIEBREAK_WEIGHT = 0.01

# Prevent Optuna from slowing early iterations too much.
# If False, MLP CV uses fixed config even when Optuna is installed.
ENABLE_MLP_OPTUNA = True
ENABLE_GP_OPTUNA = True
ENABLE_TMAX_OPTUNA = True

# ============================================================
# MLP ensemble settings
# ============================================================
MLP_ENSEMBLE_SIZE = 5
MLP_HIDDEN_DIMS = [128, 64]
MLP_DROPOUT = 0.10
MLP_LEARNING_RATE = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_MAX_EPOCHS = 500
MLP_PATIENCE = 40
MLP_VALID_FRACTION = 0.15
MLP_BATCH_SIZE = 64
MLP_CV_MAX_EPOCHS = 250
MLP_ENSEMBLE_BOOTSTRAP = False
MLP_BOOTSTRAP_STRATIFIED = True
MLP_BOOTSTRAP_SAMPLE_RATIO = 1.0

MLP_CLASSIFICATION_LOSS_WEIGHT = 1.0
MLP_TMAX_LOSS_WEIGHT = 0.5
MLP_OTHER_REGRESSION_LOSS_WEIGHT = 0.1
MLP_USE_CLASS_WEIGHT = True

RANDOM_SEED = 42

OUTPUT_DIR = Path("outputs")
OUTPUT_CANDIDATES_CSV = OUTPUT_DIR / "next_sampling_candidates.csv"
OUTPUT_SCORED_POOL_CSV = OUTPUT_DIR / "scored_candidate_pool_preview.csv"
OUTPUT_DIAGNOSTICS_CSV = OUTPUT_DIR / "combo_diagnostics.csv"
OUTPUT_MODEL_SELECTION_JSON = OUTPUT_DIR / "model_selection_report.json"
OUTPUT_OPTUNA_REPORT_JSON = OUTPUT_DIR / "optuna_report.json"
OUTPUT_CV_FOLD_METRICS_CSV = OUTPUT_DIR / "cv_fold_metrics.csv"
