from pathlib import Path

INPUT_CSV = "initial_dataset.csv"
CONTINUOUS_COLS = ["x1", "x2", "x3", "x4"]
DISCRETE_COLS = ["disc1", "disc2", "disc3"]
PASSFAIL_COL = "pass_fail"  # PASS=0, FAIL=1
TMAX_COL = "tmax"
OTHER_REGRESSION_COLS = []
TIME_FEATURE_COLS = []
PASS_LABEL = 0
FAIL_LABEL = 1

CONTINUOUS_BOUNDS = {
    "x1": (8.0, 16.0),
    "x2": (0.5, 3.0),
    "x3": (1.1, 3.0),
    "x4": (0.5, 3.0),
}
EXCLUDED_REFERENCE_RANGES = {
    "x1": {"center": 12.4, "half_width": 0.01},
    "x2": {"center": 0.85, "half_width": 0.01},
    "x3": {"center": 2.0, "half_width": 0.01},
    "x4": {"center": 1.0, "half_width": 0.01},
}

DISCRETE_LEVELS = {
    "disc1": ["A1", "A2", "S1", "S2", "S3"],
    "disc2": ["S1", "S2", "S3", "P"],
    "disc3": ["L1", "L2"],
}
S_PREFIX = "S"

CURRENT_LEVEL_TARGET_TOTAL = 636
INITIAL_TOTAL = 168
ADDITIONAL_TOTAL = CURRENT_LEVEL_TARGET_TOTAL - INITIAL_TOTAL

BATCH_SIZE = 28
CANDIDATES_PER_COMBO = 3000
MIN_SAMPLES_PER_COMBO = 8
MAX_SAMPLES_PER_COMBO = 40
MIN_BATCH_DISTANCE = 0.02
BUCKET_RATIO = {
    "boundary": 0.60,
    "pass_high_tmax": 0.30,
    "uncertainty_sparse": 0.07,
    "random_check": 0.03,
}

# GP에서는 clf_uncertainty를 0으로 두고 boundary 근접도를 직접 사용.
BOUNDARY_WEIGHTS_GP = {"boundary": 0.70, "clf_uncertainty": 0.00, "local_sparsity": 0.15, "combo_priority": 0.15}
# MLP ensemble에서는 clf_uncertainty = ensemble p_fail 표준편차.
BOUNDARY_WEIGHTS_MLP = {"boundary": 0.55, "clf_uncertainty": 0.15, "local_sparsity": 0.15, "combo_priority": 0.15}
PASS_HIGH_TMAX_WEIGHTS = {"tmax": 0.40, "pass_window": 0.30, "tmax_uncertainty": 0.10, "local_sparsity": 0.10, "combo_priority": 0.10}
UNCERTAINTY_SPARSE_WEIGHTS = {"clf_uncertainty": 0.35, "tmax_uncertainty": 0.15, "local_sparsity": 0.35, "combo_priority": 0.15}
PASS_WINDOW_LOW = 0.60
PASS_WINDOW_HIGH = 0.90
PASS_WINDOW_CENTER = 0.75

# Model mode: "gp", "mlp", "auto"
MODEL_MODE = "auto"
MLP_MIN_TOTAL_SAMPLES = 350
MLP_MIN_PASS_SAMPLES = 40
MLP_MIN_FAIL_SAMPLES = 40
MLP_MIN_SAMPLES_PER_COMBO = 8
MODEL_SELECTION_WEIGHTS = {"fail_recall": 0.70, "fail_f1": 0.30}
MLP_SELECTION_MARGIN = 0.01

MLP_ENSEMBLE_SIZE = 5
MLP_HIDDEN_DIMS = [128, 64]
MLP_DROPOUT = 0.10
MLP_LEARNING_RATE = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_MAX_EPOCHS = 500
MLP_CV_MAX_EPOCHS = 250
MLP_PATIENCE = 40
MLP_VALID_FRACTION = 0.15
MLP_BATCH_SIZE = 64
MLP_USE_CLASS_WEIGHT = True
MLP_CLASSIFICATION_LOSS_WEIGHT = 1.0
MLP_TMAX_LOSS_WEIGHT = 0.5
MLP_OTHER_REGRESSION_LOSS_WEIGHT = 0.1
CV_SPLITS = 5
RANDOM_SEED = 42

OUTPUT_DIR = Path("outputs")
OUTPUT_CANDIDATES_CSV = OUTPUT_DIR / "next_sampling_candidates.csv"
OUTPUT_SCORED_POOL_CSV = OUTPUT_DIR / "scored_candidate_pool_preview.csv"
OUTPUT_DIAGNOSTICS_CSV = OUTPUT_DIR / "combo_diagnostics.csv"
OUTPUT_MODEL_SELECTION_JSON = OUTPUT_DIR / "model_selection_report.json"
