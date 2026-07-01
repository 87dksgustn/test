import copy
import numpy as np
from sklearn.model_selection import KFold

try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

from evaluation import evaluate_gpc_cv
from metrics_utils import weighted_score
from models_gp import fit_gpr_tmax_given_pass
from models_mlp import TORCH_AVAILABLE, evaluate_mlp_cv

# Tmax tuning is provided as a placeholder for condition-aware execution.
# It is intentionally conservative because Tmax is secondary and NoTP-only data can be small.

def optuna_eligibility(df, config):
    n_total = len(df)
    n_notp = int((df[config.PASSFAIL_COL] == config.PASS_LABEL).sum())
    n_tp = int((df[config.PASSFAIL_COL] == config.FAIL_LABEL).sum())
    return {
        "optuna_available": bool(OPTUNA_AVAILABLE),
        "gp_optuna_eligible": bool(OPTUNA_AVAILABLE and config.ENABLE_OPTUNA_AUTO and config.ENABLE_GP_OPTUNA and n_total >= config.GP_OPTUNA_MIN_TOTAL_SAMPLES and n_notp >= config.GP_OPTUNA_MIN_PASS_SAMPLES and n_tp >= config.GP_OPTUNA_MIN_FAIL_SAMPLES),
        "tmax_optuna_eligible": bool(OPTUNA_AVAILABLE and config.ENABLE_OPTUNA_AUTO and config.ENABLE_TMAX_OPTUNA and n_notp >= config.TMAX_OPTUNA_MIN_PASS_SAMPLES),
        "mlp_optuna_eligible": bool(OPTUNA_AVAILABLE and TORCH_AVAILABLE and config.ENABLE_OPTUNA_AUTO and config.ENABLE_MLP_OPTUNA and n_total >= config.MLP_OPTUNA_MIN_TOTAL_SAMPLES and n_notp >= config.MLP_OPTUNA_MIN_PASS_SAMPLES and n_tp >= config.MLP_OPTUNA_MIN_FAIL_SAMPLES),
        "n_total": n_total,
        "n_notp": n_notp,
        "n_tp": n_tp,
    }

def tune_gpc_with_optuna(x_train, y_class, y_tmax, config, n_trials=None):
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
        res = evaluate_gpc_cv(x_train, y_class, y_tmax=y_tmax, pass_label=config.PASS_LABEL, tp_label=config.FAIL_LABEL, n_splits=config.CV_SPLITS, weights=config.MODEL_SELECTION_WEIGHTS, std_penalty=config.CV_STD_PENALTY, params=params, random_state=config.RANDOM_SEED)
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
    gate_stats = {"trials_total": 0, "trials_passed_gate": 0, "trials_failed_gate": 0}

    def objective(trial):
        gate_stats["trials_total"] += 1
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
        res = evaluate_mlp_cv(x_train, y_class, y_tmax, y_extra, config, tp_label=config.FAIL_LABEL, n_splits=config.CV_SPLITS, weights=config.MODEL_SELECTION_WEIGHTS, std_penalty=config.CV_STD_PENALTY, params=params)
        summary = res["summary"]

        tp_recall = float(summary.get("tp_recall", np.nan))
        tp_f1 = float(summary.get("tp_f1", np.nan))
        tmax_rmse = float(summary.get("tmax_rmse", np.nan))
        tmax_r2 = float(summary.get("tmax_r2", np.nan))
        stable = float(summary.get("stable_score", -1e9))

        pass_gate = True
        if not np.isfinite(tp_recall) or tp_recall < float(config.MLP_OPTUNA_GATE_MIN_TP_RECALL):
            pass_gate = False
        if not np.isfinite(tp_f1) or tp_f1 < float(config.MLP_OPTUNA_GATE_MIN_TP_F1):
            pass_gate = False

        max_rmse = config.MLP_OPTUNA_GATE_MAX_TMAX_RMSE
        if max_rmse is not None:
            if (not np.isfinite(tmax_rmse)) or (tmax_rmse > float(max_rmse)):
                pass_gate = False

        min_r2 = config.MLP_OPTUNA_GATE_MIN_TMAX_R2
        if min_r2 is not None:
            if (not np.isfinite(tmax_r2)) or (tmax_r2 < float(min_r2)):
                pass_gate = False

        if not pass_gate:
            gate_stats["trials_failed_gate"] += 1
            return -1e12

        gate_stats["trials_passed_gate"] += 1

        # Classification-first optimization; Tmax is only a small tie-breaker.
        score = stable
        if np.isfinite(tmax_rmse):
            score += -float(config.MLP_OPTUNA_TMAX_TIEBREAK_WEIGHT) * tmax_rmse
        return score

    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=config.MLP_OPTUNA_TIMEOUT_SEC, show_progress_bar=False)
    report = {
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "best_params": study.best_params,
        "gate": {
            "min_tp_recall": float(config.MLP_OPTUNA_GATE_MIN_TP_RECALL),
            "min_tp_f1": float(config.MLP_OPTUNA_GATE_MIN_TP_F1),
            "max_tmax_rmse": config.MLP_OPTUNA_GATE_MAX_TMAX_RMSE,
            "min_tmax_r2": config.MLP_OPTUNA_GATE_MIN_TMAX_R2,
            "tmax_tiebreak_weight": float(config.MLP_OPTUNA_TMAX_TIEBREAK_WEIGHT),
            **gate_stats,
        },
    }
    return study.best_params, report

def tune_tmax_gpr_with_optuna(x_train, y_class, y_tmax, config, n_trials=None):
    if not OPTUNA_AVAILABLE:
        return None, {"skipped": True, "reason": "Optuna is not installed."}

    mask = (y_class == config.PASS_LABEL)
    x_pass = x_train[mask]
    y_pass = y_tmax[mask]
    if len(y_pass) < max(8, config.CV_SPLITS):
        return None, {"skipped": True, "reason": "Not enough NoTP samples for Tmax GPR tuning."}

    n_trials = n_trials or config.TMAX_OPTUNA_N_TRIALS
    n_splits = max(2, min(config.CV_SPLITS, len(y_pass)))

    def objective(trial):
        params = {
            "kernel": trial.suggest_categorical("kernel", ["RBF", "Matern32", "Matern52"]),
            "constant": trial.suggest_float("constant", 0.1, 10.0, log=True),
            "length_scale": trial.suggest_float("length_scale", 0.05, 10.0, log=True),
            "noise_level": trial.suggest_float("noise_level", 1e-7, 1e-2, log=True),
            "alpha": trial.suggest_float("alpha", 1e-10, 1e-4, log=True),
            "normalize_y": trial.suggest_categorical("normalize_y", [True, False]),
            "n_restarts_optimizer": 1,
        }

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)
        rmses = []
        for tr, va in kf.split(x_pass):
            xtr = x_pass[tr]
            ytr = y_pass[tr]
            xva = x_pass[va]
            yva = y_pass[va]

            # Fit helper expects full arrays + class labels.
            dummy_cls = np.full(len(ytr), config.PASS_LABEL, dtype=int)
            reg, has = fit_gpr_tmax_given_pass(
                xtr, ytr, dummy_cls,
                pass_label=config.PASS_LABEL,
                min_pass_samples=4,
                random_state=config.RANDOM_SEED,
                params=params,
            )
            if not has:
                rmses.append(1e9)
                continue

            pred = reg.predict(xva)
            rmse = float(np.sqrt(np.mean((yva - pred) ** 2)))
            rmses.append(rmse)

        return float(np.mean(rmses))

    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=config.TMAX_OPTUNA_TIMEOUT_SEC, show_progress_bar=False)
    return study.best_params, {
        "best_value_rmse": study.best_value,
        "n_trials": len(study.trials),
        "best_params": study.best_params,
        "objective": "minimize_noTP_tmax_rmse",
    }

def maybe_tune_models(df, x_train, y_class, y_tmax, y_extra, config):
    report = {"eligibility": optuna_eligibility(df, config), "gp": None, "mlp": None, "tmax": None}
    gp_params = None; mlp_params = None; tmax_params = None
    if report["eligibility"]["gp_optuna_eligible"]:
        gp_params, report["gp"] = tune_gpc_with_optuna(x_train, y_class, y_tmax, config)
    else:
        report["gp"] = {"skipped": True, "reason": "GP Optuna eligibility conditions not met."}
    if report["eligibility"]["mlp_optuna_eligible"]:
        mlp_params, report["mlp"] = tune_mlp_with_optuna(x_train, y_class, y_tmax, y_extra, config)
    else:
        report["mlp"] = {"skipped": True, "reason": "MLP Optuna eligibility conditions not met."}
    if report["eligibility"]["tmax_optuna_eligible"]:
        tmax_params, report["tmax"] = tune_tmax_gpr_with_optuna(x_train, y_class, y_tmax, config)
    else:
        report["tmax"] = {"skipped": True, "reason": "Tmax Optuna eligibility conditions not met."}
    return {"gp_params": gp_params, "mlp_params": mlp_params, "tmax_params": tmax_params, "report": report}
