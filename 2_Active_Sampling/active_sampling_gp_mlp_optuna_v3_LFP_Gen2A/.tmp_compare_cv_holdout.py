import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import config
from data_loader import load_labeled_data
from preprocessing import build_preprocessor, make_xy
from models_gp import fit_gpc_passfail
from optuna_tuning import maybe_tune_models
from discrete_space import generate_valid_discrete_combinations, attach_discrete_combo_id
from pathlib import Path

def add_interaction_terms(df, interaction_terms):
    if not interaction_terms:
        return df
    df = df.copy()
    for col1, col2, new_col in interaction_terms:
        if col1 in df.columns and col2 in df.columns:
            df[new_col] = df[col1].astype(float) * df[col2].astype(float)
    return df

df = load_labeled_data(config.INPUT_CSV)
df = add_interaction_terms(df, getattr(config, 'INTERACTION_TERMS', []))
valid_combos = generate_valid_discrete_combinations(config.DISCRETE_LEVELS, config.DISCRETE_COLS, config.S_PREFIX)
df = attach_discrete_combo_id(df, valid_combos, config.DISCRETE_COLS)

x_raw, y_class, _ = make_xy(df, config.CONTINUOUS_COLS, config.DISCRETE_COLS, config.TPNoTP_COL, config.TMAX_COL)
pre = build_preprocessor(config.CONTINUOUS_COLS, config.DISCRETE_COLS)
x_train = pre.fit_transform(x_raw)

tuned = maybe_tune_models(df, x_train, y_class, None, None, config)
gp_params = tuned.get('gp_params')

seed = int(getattr(config, 'RANDOM_SEED', 42))
use_ard = bool(getattr(config, 'GP_MODEL_SELECTION_USE_ARD', False))
class_counts = pd.Series(y_class).value_counts()
splits = min(5, int(class_counts.min())) if len(class_counts) > 1 else 0
if splits < 2:
    raise SystemExit('Not enough class counts for CV split.')

cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
oof_pred = np.full(len(y_class), -1, dtype=int)
for fold, (tr, va) in enumerate(cv.split(x_train, y_class)):
    clf = fit_gpc_passfail(x_train[tr], y_class[tr], random_state=seed + fold, params=gp_params, use_ard=use_ard)
    oof_pred[va] = clf.predict(x_train[va])

mis_mask = (oof_pred >= 0) & (oof_pred != y_class)
cv_mis = df.loc[mis_mask].copy()
cv_mis['oof_predicted_label'] = oof_pred[mis_mask]

holdout_path = Path('outputs/Itr_5/Performance/holdout_misclassified_samples.csv')
hold_mis = pd.read_csv(holdout_path)

base = config.BASE_CONTINUOUS_COLS
for c in base:
    cv_mis[c] = cv_mis[c].astype(float)
    hold_mis[c] = hold_mis[c].astype(float)

cv_combo = set(cv_mis['discrete_combo_id'].astype(str).unique().tolist()) if len(cv_mis) else set()
h_combo = set(hold_mis['discrete_combo_id'].astype(str).unique().tolist()) if len(hold_mis) else set()

bounds = config.CONTINUOUS_BOUNDS
lo = np.array([bounds[c][0] for c in base], dtype=float)
sp = np.array([max(bounds[c][1]-bounds[c][0],1e-12) for c in base], dtype=float)

cv_x = ((cv_mis[base].to_numpy(dtype=float) - lo) / sp) if len(cv_mis) else np.zeros((0, len(base)))
h_x = ((hold_mis[base].to_numpy(dtype=float) - lo) / sp) if len(hold_mis) else np.zeros((0, len(base)))

all_nn = []
same_combo_nn = []
for i in range(len(hold_mis)):
    d_all = np.linalg.norm(cv_x - h_x[i], axis=1) if len(cv_x) else np.array([np.inf])
    all_nn.append(float(np.min(d_all)))
    mask = (cv_mis['discrete_combo_id'].astype(str).to_numpy() == str(hold_mis.iloc[i]['discrete_combo_id'])) if len(cv_mis) else np.array([], dtype=bool)
    if mask.any():
        d_sc = np.linalg.norm(cv_x[mask] - h_x[i], axis=1)
        same_combo_nn.append(float(np.min(d_sc)))
    else:
        same_combo_nn.append(float('inf'))

print('cv_mis_count=', len(cv_mis))
print('holdout_mis_count=', len(hold_mis))
print('cv_combo=', sorted(cv_combo))
print('holdout_combo=', sorted(h_combo))
print('combo_intersection=', sorted(cv_combo & h_combo))
print('combo_jaccard=', (len(cv_combo & h_combo) / max(1, len(cv_combo | h_combo))))
print('all_nn_mean=', np.mean(all_nn) if all_nn else np.nan)
print('all_nn_max=', np.max(all_nn) if all_nn else np.nan)
print('same_combo_nn_mean=', np.mean(same_combo_nn) if same_combo_nn else np.nan)
print('same_combo_nn_max=', np.max(same_combo_nn) if same_combo_nn else np.nan)

for i in range(len(hold_mis)):
    if len(cv_mis)==0:
        break
    d = np.linalg.norm(cv_x - h_x[i], axis=1)
    j = int(np.argmin(d))
    print('holdout_row=', hold_mis.iloc[i].get('holdout_row_num','NA'), 'combo=', hold_mis.iloc[i]['discrete_combo_id'], 'err=', hold_mis.iloc[i]['error_type'], 'nearest_cv_eval=', cv_mis.iloc[j].get('Eval#','NA'), 'nn=', round(float(d[j]),4))
