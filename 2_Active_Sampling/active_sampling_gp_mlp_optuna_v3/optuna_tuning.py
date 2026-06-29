import copy
import numpy as np

try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

from evaluation import evaluate_gpc_cv
from metrics_utils import weighted_score
from models_mlp import TORCH_AVAILABLE, evaluate_mlp_cv

# Tmax tuning is provided as a placeholder for condition-aware execution.
# It is intentionally conservative because Tmax is secondary and PASS-only data can be small.

def optuna_eligibility(df, config):
    n_total = len(df)
    n_pass = int((df[config.PASSFAIL_COL] == config.PASS_LABEL).sum())
    n_fail = int((df[config.PASSFAIL_COL] == config.FAIL_LABEL).sum())
    return {
        "optuna_available": bool(OPTUNA_AVAILABLE),
        "gp_optuna_eligible": bool(OPTUNA_AVAILABLE and config.ENABLE_OPTUNA_AUTO and config.ENABLE_GP_OPTUNA and n_total >= config.GP_OPTUNA_MIN_TOTAL_SAMPLES and n_pass >= config.GP_OPTUNA_MIN_PASS_SAMPLES and n_fail >= config.GP_OPTUNA_MIN_FAIL_SAMPLES),
        "tmax_optuna_eligible": bool(OPTUNA_AVAILABLE and config.ENABLE_OPTUNA_AUTO and config.ENABLE_TMAX_OPTUNA and n_pass >= config.TMAX_OPTUNA_MIN_PASS_SAMPLES),
        "mlp_optuna_eligible": bool(OPTUNA_AVAILABLE and TORCH_AVAILABLE and config.ENABLE_OPTUNA_AUTO and config.ENABLE_MLP_OPTUNA and n_total >= config.MLP_OPTUNA_MIN_TOTAL_SAMPLES and n_pass >= config.MLP_OPTUNA_MIN_PASS_SAMPLES and n_fail >= config.MLP_OPTUNA_MIN_FAIL_SAMPLES),
        "n_total": n_total,
        "n_pass": n_pass,
        "n_fail": n_fail,
    }

def tune_gpc_with_optuna(x_train, y_class, config, n_trials=None):
    if not OPTUNA_AVAILABLE:
        return None, {"skipped": True, "reason": "Optuna is not installed."}
    n_trials = n_trials or config.GP_OPTUNA_N_TRIALS
    def objective(trial):
        params = {
            "kernel": trial.suggest_categorical("kernel", ["RBF", "Matern32", "Matern52"]),
            "constant": trial.suggest_float("constant", 0.1, 10.0, log=True),
            "length_scale": trial.suggest_float("length_scale", 0.05, 10.0, log=True),
            "n_restarts_optimizer": 1,
        }
        res = evaluate_gpc_cv(x_train, y_class, fail_label=config.FAIL_LABEL, n_splits=config.CV_SPLITS, weights=config.MODEL_SELECTION_WEIGHTS, std_penalty=config.CV_STD_PENALTY, params=params, random_state=config.RANDOM_SEED)
        return res["summary"].get("stable_score", -1e9)
    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=config.GP_OPTUNA_TIMEOUT_SEC, show_progress_bar=False)
    return study.best_params, {"best_value": study.best_value, "n_trials": len(study.trials), "best_params": study.best_params}

def tune_mlp_with_optuna(x_train, y_class, y_tmax, y_extra, config, n_trials=None):
    if not OPTUNA_AVAILABLE:
        return None, {"skipped": True, "reason": "Optuna is not installed."}
    if not TORCH_AVAILABLE:
        return None, {"skipped": True, "reason": "PyTorch is not installed."}
    n_trials = n_trials or config.MLP_OPTUNA_N_TRIALS
    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 2, 3)
        width = trial.suggest_categorical("width", [64, 128, 256])
        hidden_dims = [width for _ in range(n_layers)]
        params = {
            "hidden_dims": hidden_dims,
            "dropout": trial.suggest_float("dropout", 0.0, 0.30),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "tmax_loss_weight": trial.suggest_float("tmax_loss_weight", 0.2, 1.0),
            "ensemble_size": config.MLP_ENSEMBLE_SIZE,
        }
        res = evaluate_mlp_cv(x_train, y_class, y_tmax, y_extra, config, fail_label=config.FAIL_LABEL, n_splits=config.CV_SPLITS, weights=config.MODEL_SELECTION_WEIGHTS, std_penalty=config.CV_STD_PENALTY, params=params)
        return res["summary"].get("stable_score", -1e9)
    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=config.MLP_OPTUNA_TIMEOUT_SEC, show_progress_bar=False)
    return study.best_params, {"best_value": study.best_value, "n_trials": len(study.trials), "best_params": study.best_params}

def maybe_tune_models(df, x_train, y_class, y_tmax, y_extra, config):
    report = {"eligibility": optuna_eligibility(df, config), "gp": None, "mlp": None, "tmax": None}
    gp_params = None; mlp_params = None; tmax_params = None
    if report["eligibility"]["gp_optuna_eligible"]:
        gp_params, report["gp"] = tune_gpc_with_optuna(x_train, y_class, config)
    else:
        report["gp"] = {"skipped": True, "reason": "GP Optuna eligibility conditions not met."}
    if report["eligibility"]["mlp_optuna_eligible"]:
        mlp_params, report["mlp"] = tune_mlp_with_optuna(x_train, y_class, y_tmax, y_extra, config)
    else:
        report["mlp"] = {"skipped": True, "reason": "MLP Optuna eligibility conditions not met."}
    if report["eligibility"]["tmax_optuna_eligible"]:
        # Conservative default for now. Use fixed robust noise params.
        tmax_params = {"kernel": "RBF", "constant": 1.0, "length_scale": 1.0, "noise_level": 1e-4, "alpha": 1e-8, "normalize_y": True, "n_restarts_optimizer": 3}
        report["tmax"] = {"used_default_robust_params": True, "params": tmax_params}
    else:
        report["tmax"] = {"skipped": True, "reason": "Tmax Optuna eligibility conditions not met."}
    return {"gp_params": gp_params, "mlp_params": mlp_params, "tmax_params": tmax_params, "report": report}
