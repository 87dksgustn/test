import os
import math
import json
import logging
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


# ============================================================
# User configuration
# ============================================================
DATA_PATH = "1_Cond_screening.csv"
OUTPUT_DIR = "active_sampling_results"

# Column names
CATEGORICAL_COL = "A_Type_CompPad"
CONTINUOUS_COLS = [
    "B_Extd_CompPad",
    "C_Extd_Frame_Btm",
    "D_Extd_SideBeam",
    "E_ThermalResin_Thx",
    "F_dt_Cell",
    "G_MaxT",
]
TARGET_COLS = ["Time_TR_Adj_1", "Time_TR_Adj_2"]

# Study settings
N_SPLITS = 5
INITIAL_DOE_SIZE = 50
BATCH_SIZE = 10
MAX_TOTAL_TRAIN_SIZE = 400   # stopping이 있으므로 최대 상한선(cap)
TOP_PERCENT = 0.10
TOP_K = 10
ALPHAS = [0.3, 0.5, 0.7]
RANDOM_STATE = 42

# Model settings
MLP_ENSEMBLE_SIZE = 5
MLP_HIDDEN_LAYER_SIZES = (64, 64)
MLP_MAX_ITER = 2000

# Target-reached stopping settings
USE_TARGET_STOPPING = True
STOP_MIN_TRAIN_SIZE = 100
STOP_TARGET_REGRET_OBJ = 0.05   # lower is better
STOP_TARGET_R2_MEAN = 0.95      # higher is better


# ============================================================
# Configuration Validation
# ============================================================
def validate_config() -> None:
    """Validate configuration parameters at startup."""
    if INITIAL_DOE_SIZE >= MAX_TOTAL_TRAIN_SIZE:
        raise ValueError("INITIAL_DOE_SIZE must be < MAX_TOTAL_TRAIN_SIZE")
    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be > 0")
    if not 0 < TOP_PERCENT <= 1:
        raise ValueError("TOP_PERCENT must be in (0, 1]")
    if TOP_K <= 0:
        raise ValueError("TOP_K must be > 0")
    for alpha in ALPHAS:
        if not 0 <= alpha <= 1:
            raise ValueError(f"Alpha {alpha} must be in [0, 1]")
    if N_SPLITS < 2:
        raise ValueError("N_SPLITS must be >= 2")
    if USE_TARGET_STOPPING and STOP_MIN_TRAIN_SIZE < INITIAL_DOE_SIZE:
        raise ValueError("STOP_MIN_TRAIN_SIZE should be >= INITIAL_DOE_SIZE")
    logger.info("Configuration validated successfully.")


# ============================================================
# Utility functions
# ============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    corr, _ = spearmanr(y_true, y_pred)
    if np.isnan(corr):
        return 0.0
    return float(corr)


def minmax_normalize(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    a_min = np.min(arr)
    a_max = np.max(arr)
    if abs(a_max - a_min) < eps:
        return np.zeros_like(arr)
    return (arr - a_min) / (a_max - a_min)


def combined_objective_from_y(y: np.ndarray) -> np.ndarray:
    """
    Both y1 and y2 are maximized with equal 5:5 weight.
    To combine different scales fairly, normalize each target within the current reference set.
    """
    y = np.asarray(y, dtype=float)
    y1_norm = minmax_normalize(y[:, 0])
    y2_norm = minmax_normalize(y[:, 1])
    return 0.5 * y1_norm + 0.5 * y2_norm


def top_k_indices_desc(values: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(values))
    return np.argsort(values)[::-1][:k]


def build_preprocessor(cat_col: str, cont_cols: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, cont_cols),
            ("cat", categorical_transformer, [cat_col]),
        ]
    )
    return preprocessor


# ============================================================
# Models
# ============================================================
class GPRMultiOutputModel:
    def __init__(self, preprocessor: ColumnTransformer, random_state: int = 42):
        self.preprocessor = clone(preprocessor)
        self.random_state = random_state
        self.models = []
        self.y_scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        Xp = self.preprocessor.fit_transform(X)
        ys = self.y_scaler.fit_transform(y)
        self.models = []
        for i in range(ys.shape[1]):
            kernel = (
                C(1.0, (1e-3, 1e3))
                * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
                + WhiteKernel(noise_level=1e-5)
            )
            gpr = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=False,
                n_restarts_optimizer=2,
                random_state=self.random_state,
            )
            gpr.fit(Xp, ys[:, i])
            self.models.append(gpr)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict().")
        Xp = self.preprocessor.transform(X)
        pred_scaled_list = []
        std_scaled_list = []
        for model in self.models:
            mu, std = model.predict(Xp, return_std=True)
            pred_scaled_list.append(mu)
            std_scaled_list.append(std)
        pred_scaled = np.column_stack(pred_scaled_list)
        std_scaled = np.column_stack(std_scaled_list)

        pred = self.y_scaler.inverse_transform(pred_scaled)
        std = std_scaled * self.y_scaler.scale_
        return pred, std


class MLPEnsembleModel:
    def __init__(
        self,
        preprocessor: ColumnTransformer,
        ensemble_size: int = 5,
        hidden_layer_sizes: Tuple[int, ...] = (64, 64),
        max_iter: int = 2000,
        random_state: int = 42,
    ):
        self.preprocessor = clone(preprocessor)
        self.ensemble_size = ensemble_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.models = []
        self.y_scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        Xp = self.preprocessor.fit_transform(X)
        ys = self.y_scaler.fit_transform(y)
        self.models = []
        n = len(Xp)
        rng = np.random.RandomState(self.random_state)

        for i in range(self.ensemble_size):
            idx = rng.choice(n, size=n, replace=True)
            Xb = Xp[idx]
            yb = ys[idx]
            base = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=self.max_iter,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=self.random_state + i,
            )
            model = MultiOutputRegressor(base)
            model.fit(Xb, yb)
            self.models.append(model)

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict().")
        Xp = self.preprocessor.transform(X)
        preds_scaled = []
        for model in self.models:
            preds_scaled.append(model.predict(Xp))
        preds_scaled = np.stack(preds_scaled, axis=0)  # [n_models, n_samples, n_targets]

        mean_scaled = preds_scaled.mean(axis=0)
        std_scaled = preds_scaled.std(axis=0)

        pred = self.y_scaler.inverse_transform(mean_scaled)
        std = std_scaled * self.y_scaler.scale_
        return pred, std


# ============================================================
# Initial DOE selection: categorical stratified + continuous space-filling
# ============================================================
def select_initial_doe(
    X_pool: pd.DataFrame,
    y_pool: np.ndarray,
    cat_col: str,
    cont_cols: List[str],
    initial_size: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    1) Allocate initial DOE count by categorical level proportion
    2) Within each category, select space-filling representatives using KMeans centers
    Returns selected_idx, remaining_idx relative to X_pool index positions [0..len-1]
    """
    rng = np.random.RandomState(random_state)
    df = X_pool.reset_index(drop=True).copy()
    df["__rowid__"] = np.arange(len(df))

    levels = df[cat_col].value_counts().sort_index()
    n_levels = len(levels)
    if initial_size < n_levels:
        raise ValueError("INITIAL_DOE_SIZE must be >= number of categorical levels.")

    raw_alloc = levels / levels.sum() * initial_size
    alloc = np.floor(raw_alloc).astype(int)

    for lvl in alloc.index:
        if alloc[lvl] == 0:
            alloc[lvl] = 1

    while alloc.sum() > initial_size:
        lvl = alloc.idxmax()
        if alloc[lvl] > 1:
            alloc[lvl] -= 1
        else:
            break

    while alloc.sum() < initial_size:
        frac = raw_alloc - np.floor(raw_alloc)
        lvl = frac.idxmax()
        alloc[lvl] += 1
        raw_alloc[lvl] = -1

    selected = []
    for lvl, n_pick in alloc.items():
        subset = df[df[cat_col] == lvl].copy()
        sub_idx = subset["__rowid__"].values
        Xc = subset[cont_cols].values

        if len(subset) <= n_pick:
            selected.extend(sub_idx.tolist())
            continue

        scaler = StandardScaler()
        Xc_scaled = scaler.fit_transform(Xc)

        kmeans = KMeans(n_clusters=n_pick, random_state=random_state, n_init=10)
        kmeans.fit(Xc_scaled)
        centers = kmeans.cluster_centers_

        chosen_local = []
        used = set()
        for center in centers:
            d2 = np.sum((Xc_scaled - center) ** 2, axis=1)
            order = np.argsort(d2)
            pick = None
            for oi in order:
                rid = sub_idx[oi]
                if rid not in used:
                    pick = rid
                    used.add(rid)
                    break
            if pick is not None:
                chosen_local.append(pick)

        if len(chosen_local) < n_pick:
            remaining = [rid for rid in sub_idx if rid not in used]
            rng.shuffle(remaining)
            chosen_local.extend(remaining[: n_pick - len(chosen_local)])

        selected.extend(chosen_local)

    selected = np.array(sorted(set(selected)), dtype=int)

    if len(selected) < initial_size:
        all_ids = set(np.arange(len(df)))
        remaining = list(all_ids - set(selected.tolist()))
        rng.shuffle(remaining)
        selected = np.concatenate(
            [selected, np.array(remaining[: initial_size - len(selected)], dtype=int)]
        )

    selected = selected[:initial_size]
    remaining = np.array(
        [i for i in range(len(df)) if i not in set(selected.tolist())],
        dtype=int
    )
    return selected, remaining


# ============================================================
# Acquisition functions
# ============================================================
def compute_uncertainty_score(std_pred: np.ndarray) -> np.ndarray:
    """
    std_pred shape: [n_samples, 2]
    Combine target uncertainties equally.
    """
    return 0.5 * minmax_normalize(std_pred[:, 0]) + 0.5 * minmax_normalize(std_pred[:, 1])


def compute_exploitation_score(pred_y: np.ndarray) -> np.ndarray:
    """
    Both outputs are maximized equally.
    """
    obj = combined_objective_from_y(pred_y)
    return minmax_normalize(obj)


def select_batch_random(pool_indices: np.ndarray, batch_size: int, rng: np.random.RandomState) -> np.ndarray:
    if len(pool_indices) <= batch_size:
        return pool_indices.copy()
    return rng.choice(pool_indices, size=batch_size, replace=False)


def select_batch_uncertainty_exploitation(
    pred_y_pool: np.ndarray,
    std_y_pool: np.ndarray,
    pool_indices: np.ndarray,
    batch_size: int,
    alpha: float,
) -> np.ndarray:
    uncertainty = compute_uncertainty_score(std_y_pool)
    exploitation = compute_exploitation_score(pred_y_pool)
    score = alpha * uncertainty + (1.0 - alpha) * exploitation
    order = np.argsort(score)[::-1]
    chosen_pos = order[: min(batch_size, len(order))]
    return pool_indices[chosen_pos]


# ============================================================
# Metrics
# ============================================================
def evaluate_prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    out = {}
    for i, tname in enumerate(TARGET_COLS):
        out[f"r2_{tname}"] = float(r2_score(y_true[:, i], y_pred[:, i]))
        out[f"rmse_{tname}"] = rmse(y_true[:, i], y_pred[:, i])
        out[f"mae_{tname}"] = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
    return out


def evaluate_optimization_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Regret definition here (maximize):
      regret = true_best - true_value_at_predicted_best
      lower is better.
    """
    out = {}
    n = len(y_true)
    top_n = max(1, int(math.ceil(n * TOP_PERCENT)))

    for i, tname in enumerate(TARGET_COLS):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        true_best_idx = int(np.argmax(yt))
        pred_best_idx = int(np.argmax(yp))
        regret_val = float(yt[true_best_idx] - yt[pred_best_idx])

        true_top = set(top_k_indices_desc(yt, top_n).tolist())
        pred_top = set(top_k_indices_desc(yp, top_n).tolist())
        hit = 1.0 if pred_best_idx in true_top else 0.0

        true_topk = set(top_k_indices_desc(yt, TOP_K).tolist())
        pred_topk = set(top_k_indices_desc(yp, TOP_K).tolist())
        overlap = len(true_topk.intersection(pred_topk)) / max(1, TOP_K)

        out[f"regret_{tname}"] = regret_val
        out[f"top_percent_hit_{tname}"] = hit
        out[f"rank_corr_{tname}"] = safe_spearman(yt, yp)
        out[f"topk_overlap_{tname}"] = float(overlap)

    obj_true = combined_objective_from_y(y_true)
    obj_pred = combined_objective_from_y(y_pred)

    true_best_idx = int(np.argmax(obj_true))
    pred_best_idx = int(np.argmax(obj_pred))
    out["regret_obj"] = float(obj_true[true_best_idx] - obj_true[pred_best_idx])

    true_top = set(top_k_indices_desc(obj_true, top_n).tolist())
    pred_top = set(top_k_indices_desc(obj_pred, top_n).tolist())
    out["top_percent_hit_obj"] = 1.0 if pred_best_idx in true_top else 0.0

    true_topk = set(top_k_indices_desc(obj_true, TOP_K).tolist())
    pred_topk = set(top_k_indices_desc(obj_pred, TOP_K).tolist())
    out["rank_corr_obj"] = safe_spearman(obj_true, obj_pred)
    out["topk_overlap_obj"] = len(true_topk.intersection(pred_topk)) / max(1, TOP_K)

    return out


# ============================================================
# Main study loop
# ============================================================
@dataclass
class StudyConfig:
    model_name: str
    strategy_name: str
    alpha: Optional[float]
    split_id: int
    iteration: int
    train_size: int
    stopped_early: bool = False
    stop_reason: str = ""
    stop_iteration: Optional[int] = None
    stop_train_size: Optional[int] = None


class ActiveSamplingStudy:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.X = self.df[[CATEGORICAL_COL] + CONTINUOUS_COLS].copy()
        self.y = self.df[TARGET_COLS].values.astype(float)
        self.preprocessor = build_preprocessor(CATEGORICAL_COL, CONTINUOUS_COLS)
        ensure_dir(OUTPUT_DIR)
        ensure_dir(os.path.join(OUTPUT_DIR, "plots"))
        ensure_dir(os.path.join(OUTPUT_DIR, "stop"))
        ensure_dir(os.path.join(OUTPUT_DIR, "stop", "plots"))

    def _compute_r2_mean(self, pred_metrics: Dict[str, float]) -> float:
        return float(np.mean([pred_metrics[f"r2_{t}"] for t in TARGET_COLS]))

    def _check_target_stop(
        self,
        regret_obj: float,
        r2_mean: float,
        train_size: int,
    ) -> Tuple[bool, str]:
        if not USE_TARGET_STOPPING:
            return False, ""
        if train_size < STOP_MIN_TRAIN_SIZE:
            return False, ""

        regret_ok = regret_obj <= STOP_TARGET_REGRET_OBJ
        r2_ok = r2_mean >= STOP_TARGET_R2_MEAN

        if regret_ok and r2_ok:
            return True, "target_reached"
        return False, ""

    def _make_model(self, model_name: str, seed: int):
        if model_name == "GPR":
            return GPRMultiOutputModel(preprocessor=self.preprocessor, random_state=seed)
        elif model_name == "MLP_ENSEMBLE":
            return MLPEnsembleModel(
                preprocessor=self.preprocessor,
                ensemble_size=MLP_ENSEMBLE_SIZE,
                hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
                max_iter=MLP_MAX_ITER,
                random_state=seed,
            )
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

    def _run_strategy(
        self,
        model_name: str,
        strategy_name: str,
        alpha: Optional[float],
        split_id: int,
        X_pool_full: pd.DataFrame,
        y_pool_full: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        initial_idx: np.ndarray,
        remaining_idx: np.ndarray,
    ) -> List[Dict]:
        """Run a single strategy for a given model and split."""
        results = []
        rng = np.random.RandomState(RANDOM_STATE + split_id)
        train_idx = initial_idx.copy()
        pool_remain_idx = remaining_idx.copy()
        stopped_early = False
        stop_reason = ""
        stop_iteration = None
        stop_train_size = None

        n_iterations = int((MAX_TOTAL_TRAIN_SIZE - INITIAL_DOE_SIZE) / BATCH_SIZE)

        for iteration in range(n_iterations + 1):
            X_train = X_pool_full.iloc[train_idx].reset_index(drop=True)
            y_train = y_pool_full[train_idx]

            model = self._make_model(
                model_name=model_name,
                seed=RANDOM_STATE + split_id + iteration
            )
            model.fit(X_train, y_train)

            y_pred_test, y_std_test = model.predict(X_test)
            pred_metrics = evaluate_prediction_metrics(y_test, y_pred_test)
            opt_metrics = evaluate_optimization_metrics(y_test, y_pred_test)
            r2_mean = self._compute_r2_mean(pred_metrics)

            should_stop, current_stop_reason = self._check_target_stop(
                regret_obj=opt_metrics["regret_obj"],
                r2_mean=r2_mean,
                train_size=len(train_idx),
            )

            if should_stop and not stopped_early:
                stopped_early = True
                stop_reason = current_stop_reason
                stop_iteration = iteration
                stop_train_size = len(train_idx)

            row = asdict(
                StudyConfig(
                    model_name=model_name,
                    strategy_name=strategy_name,
                    alpha=alpha,
                    split_id=split_id,
                    iteration=iteration,
                    train_size=len(train_idx),
                    stopped_early=stopped_early,
                    stop_reason=stop_reason,
                    stop_iteration=stop_iteration,
                    stop_train_size=stop_train_size,
                )
            )
            row["r2_mean"] = r2_mean
            row.update(pred_metrics)
            row.update(opt_metrics)
            results.append(row)

            if should_stop:
                logger.info(
                    f"[Stop] split={split_id}, model={model_name}, "
                    f"strategy={strategy_name}, train_size={len(train_idx)}, "
                    f"reason={stop_reason}"
                )
                break

            if iteration == n_iterations or len(pool_remain_idx) == 0:
                continue

            X_remain = X_pool_full.iloc[pool_remain_idx].reset_index(drop=True)
            y_pred_remain, y_std_remain = model.predict(X_remain)

            if strategy_name == "random":
                selected_global = select_batch_random(pool_remain_idx, BATCH_SIZE, rng)
            else:
                selected_global = select_batch_uncertainty_exploitation(
                    pred_y_pool=y_pred_remain,
                    std_y_pool=y_std_remain,
                    pool_indices=pool_remain_idx,
                    batch_size=BATCH_SIZE,
                    alpha=float(alpha),
                )

            train_idx = np.concatenate([train_idx, selected_global])
            selected_set = set(selected_global.tolist())
            pool_remain_idx = np.array(
                [i for i in pool_remain_idx if i not in selected_set],
                dtype=int
            )

        return results

    def _run_single_split(
        self,
        split_id: int,
        pool_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> List[Dict]:
        """Run all models and strategies for a single K-Fold split."""
        results = []

        X_pool_full = self.X.iloc[pool_idx].reset_index(drop=True)
        y_pool_full = self.y[pool_idx]
        X_test = self.X.iloc[test_idx].reset_index(drop=True)
        y_test = self.y[test_idx]

        initial_idx, remaining_idx = select_initial_doe(
            X_pool=X_pool_full,
            y_pool=y_pool_full,
            cat_col=CATEGORICAL_COL,
            cont_cols=CONTINUOUS_COLS,
            initial_size=INITIAL_DOE_SIZE,
            random_state=RANDOM_STATE + split_id,
        )

        strategies = [("random", None)] + [(f"ue_alpha_{a}", a) for a in ALPHAS]
        model_names = ["GPR", "MLP_ENSEMBLE"]

        for model_name in model_names:
            for strategy_name, alpha in strategies:
                strategy_results = self._run_strategy(
                    model_name=model_name,
                    strategy_name=strategy_name,
                    alpha=alpha,
                    split_id=split_id,
                    X_pool_full=X_pool_full,
                    y_pool_full=y_pool_full,
                    X_test=X_test,
                    y_test=y_test,
                    initial_idx=initial_idx,
                    remaining_idx=remaining_idx,
                )
                results.extend(strategy_results)

        logger.info(f"[Done] Split {split_id}/{N_SPLITS}")
        return results

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute the full active sampling study across all K-Fold splits."""
        results = []
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

        for split_id, (pool_idx, test_idx) in enumerate(kf.split(self.X), start=1):
            split_results = self._run_single_split(split_id, pool_idx, test_idx)
            results.extend(split_results)

        df_results = pd.DataFrame(results)
        df_results.to_csv(os.path.join(OUTPUT_DIR, "all_results_raw.csv"), index=False)

        metric_cols = [
            c for c in df_results.columns
            if c not in [
                "model_name", "strategy_name", "alpha", "split_id", "iteration", "train_size",
                "stopped_early", "stop_reason", "stop_iteration", "stop_train_size"
            ]
        ]

        df_summary = (
            df_results
            .groupby(["model_name", "strategy_name", "alpha", "iteration", "train_size"])[metric_cols]
            .agg(["mean", "std"])
            .reset_index()
        )
        df_summary.columns = ["_".join(col).strip("_") for col in df_summary.columns.values]
        df_summary.to_csv(os.path.join(OUTPUT_DIR, "summary_mean_std.csv"), index=False)

        self._plot_metrics(df_results)
        self.export_stop_summary(df_results)

        return df_results, df_summary

    def _plot_metrics(self, df_results: pd.DataFrame) -> None:
        metrics_to_plot = []
        for t in TARGET_COLS:
            metrics_to_plot.extend([
                f"r2_{t}", f"rmse_{t}", f"mae_{t}",
                f"regret_{t}", f"top_percent_hit_{t}", f"rank_corr_{t}", f"topk_overlap_{t}",
            ])
        metrics_to_plot.extend([
            "r2_mean",
            "regret_obj", "top_percent_hit_obj", "rank_corr_obj", "topk_overlap_obj",
        ])

        for model_name in sorted(df_results["model_name"].unique()):
            df_model = df_results[df_results["model_name"] == model_name].copy()
            strategy_order = ["random"] + [f"ue_alpha_{a}" for a in ALPHAS]

            for metric in metrics_to_plot:
                if metric not in df_model.columns:
                    logger.warning(f"{metric} not found, skip")
                    continue

                plt.figure(figsize=(8, 5))
                has_data = False

                for strategy in strategy_order:
                    d = df_model[df_model["strategy_name"] == strategy]
                    if d.empty:
                        continue

                    g = d.groupby("train_size")[metric].agg(["mean", "std"]).reset_index()
                    if g.empty:
                        continue

                    has_data = True
                    plt.plot(g["train_size"], g["mean"], marker="o", label=strategy)
                    plt.fill_between(
                        g["train_size"],
                        g["mean"] - g["std"].fillna(0),
                        g["mean"] + g["std"].fillna(0),
                        alpha=0.15,
                    )

                if not has_data:
                    plt.close()
                    continue

                plt.xlabel("Train size")
                plt.ylabel(metric)
                plt.title(f"{model_name} - {metric}")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    os.path.join(OUTPUT_DIR, "plots", f"{model_name}_{metric}.png"),
                    dpi=200
                )
                plt.close()

    def export_stop_summary(self, df_results: pd.DataFrame):
        df_last = (
            df_results
            .sort_values(["model_name", "strategy_name", "split_id", "iteration"])
            .groupby(["model_name", "strategy_name", "alpha", "split_id"])
            .tail(1)
            .copy()
        )

        df_last["final_stop_reason"] = np.where(
            df_last["stopped_early"],
            df_last["stop_reason"],
            "max_iter"
        )
        df_last["final_train_size"] = df_last["train_size"]

        df_last.to_csv(
            os.path.join(OUTPUT_DIR, "stop", "stop_summary_raw.csv"),
            index=False
        )

        summary = (
            df_last
            .groupby(["model_name", "strategy_name", "alpha"])[
                ["final_train_size", "regret_obj", "r2_mean"]
            ]
            .agg(["mean", "std"])
            .reset_index()
        )
        summary.columns = ["_".join(col).strip("_") for col in summary.columns]
        summary.to_csv(
            os.path.join(OUTPUT_DIR, "stop", "stop_summary_mean_std.csv"),
            index=False
        )

        self.plot_stop_summary(df_last)

    def plot_stop_summary(self, df_last: pd.DataFrame):
        strategy_order = ["random"] + [f"ue_alpha_{a}" for a in ALPHAS]

        for model_name in df_last["model_name"].unique():
            df_m = df_last[df_last["model_name"] == model_name].copy()

            # 1. Stop train size
            plt.figure(figsize=(8, 5))
            labels = []
            means = []
            stds = []

            for s in strategy_order:
                d = df_m[df_m["strategy_name"] == s]
                if len(d) == 0:
                    continue
                labels.append(s)
                means.append(d["final_train_size"].mean())
                stds.append(d["final_train_size"].std(ddof=0))

            x = np.arange(len(labels))
            plt.bar(x, means, yerr=stds, capsize=5)
            plt.xticks(x, labels, rotation=20)
            plt.ylabel("Final train size")
            plt.title(f"{model_name} - Stop Train Size")
            plt.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUTPUT_DIR, "stop", "plots", f"{model_name}_stop_train_size.png"),
                dpi=200
            )
            plt.close()

            # 2. Final regret
            plt.figure(figsize=(8, 5))
            labels = []
            means = []
            stds = []

            for s in strategy_order:
                d = df_m[df_m["strategy_name"] == s]
                if len(d) == 0:
                    continue
                labels.append(s)
                means.append(d["regret_obj"].mean())
                stds.append(d["regret_obj"].std(ddof=0))

            x = np.arange(len(labels))
            plt.bar(x, means, yerr=stds, capsize=5)
            plt.xticks(x, labels, rotation=20)
            plt.ylabel("Final regret_obj")
            plt.title(f"{model_name} - Final Regret at Stop")
            plt.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUTPUT_DIR, "stop", "plots", f"{model_name}_stop_regret.png"),
                dpi=200
            )
            plt.close()

            # 3. Stop rate
            plt.figure(figsize=(8, 5))
            labels = []
            means = []

            for s in strategy_order:
                d = df_m[df_m["strategy_name"] == s]
                if len(d) == 0:
                    continue
                labels.append(s)
                stop_flag = (d["final_stop_reason"] != "max_iter").astype(float)
                means.append(stop_flag.mean())

            x = np.arange(len(labels))
            plt.bar(x, means)
            plt.xticks(x, labels, rotation=20)
            plt.ylabel("Stop rate")
            plt.title(f"{model_name} - Stop Rate")
            plt.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUTPUT_DIR, "stop", "plots", f"{model_name}_stop_rate.png"),
                dpi=200
            )
            plt.close()


# ============================================================
# Entry point
# ============================================================
def validate_dataframe(df: pd.DataFrame) -> None:
    required_cols = [CATEGORICAL_COL] + CONTINUOUS_COLS + TARGET_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")
    if len(df) < INITIAL_DOE_SIZE:
        raise ValueError(f"Dataset must have at least {INITIAL_DOE_SIZE} rows, got {len(df)}.")
    logger.info(f"DataFrame validated: {len(df)} rows, {len(df.columns)} columns.")


if __name__ == "__main__":
    validate_config()
    ensure_dir(OUTPUT_DIR)
    df = pd.read_csv(DATA_PATH)
    validate_dataframe(df)

    study = ActiveSamplingStudy(df)
    raw_results, summary = study.run()

    logger.info("Study finished.")
    logger.info(f"Raw result file   : {os.path.join(OUTPUT_DIR, 'all_results_raw.csv')}")
    logger.info(f"Summary file      : {os.path.join(OUTPUT_DIR, 'summary_mean_std.csv')}")
    logger.info(f"Plots directory   : {os.path.join(OUTPUT_DIR, 'plots')}")
    logger.info(f"Stop raw csv      : {os.path.join(OUTPUT_DIR, 'stop', 'stop_summary_raw.csv')}")
    logger.info(f"Stop summary csv  : {os.path.join(OUTPUT_DIR, 'stop', 'stop_summary_mean_std.csv')}")
    logger.info(f"Stop plots dir    : {os.path.join(OUTPUT_DIR, 'stop', 'plots')}")

    config = {
        "DATA_PATH": DATA_PATH,
        "OUTPUT_DIR": OUTPUT_DIR,
        "CATEGORICAL_COL": CATEGORICAL_COL,
        "CONTINUOUS_COLS": CONTINUOUS_COLS,
        "TARGET_COLS": TARGET_COLS,
        "N_SPLITS": N_SPLITS,
        "INITIAL_DOE_SIZE": INITIAL_DOE_SIZE,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_TOTAL_TRAIN_SIZE": MAX_TOTAL_TRAIN_SIZE,
        "TOP_PERCENT": TOP_PERCENT,
        "TOP_K": TOP_K,
        "ALPHAS": ALPHAS,
        "RANDOM_STATE": RANDOM_STATE,
        "MLP_ENSEMBLE_SIZE": MLP_ENSEMBLE_SIZE,
        "MLP_HIDDEN_LAYER_SIZES": MLP_HIDDEN_LAYER_SIZES,
        "MLP_MAX_ITER": MLP_MAX_ITER,
        "USE_TARGET_STOPPING": USE_TARGET_STOPPING,
        "STOP_MIN_TRAIN_SIZE": STOP_MIN_TRAIN_SIZE,
        "STOP_TARGET_REGRET_OBJ": STOP_TARGET_REGRET_OBJ,
        "STOP_TARGET_R2_MEAN": STOP_TARGET_R2_MEAN,
    }

    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)