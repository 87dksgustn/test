"""Compare Isotropic vs ARD kernel performance."""
import warnings
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning

import config
from data_loader import load_labeled_data
from discrete_space import generate_valid_discrete_combinations, attach_discrete_combo_id
from preprocessing import build_preprocessor, make_xy

def add_interaction_terms(df, interaction_terms):
    if not interaction_terms:
        return df
    df = df.copy()
    for col1, col2, new_col in interaction_terms:
        if col1 in df.columns and col2 in df.columns:
            df[new_col] = df[col1].astype(float) * df[col2].astype(float)
    return df

# Load data
df = load_labeled_data(config.INPUT_CSV)
df = add_interaction_terms(df, getattr(config, "INTERACTION_TERMS", []))
valid_combos = generate_valid_discrete_combinations(config.DISCRETE_LEVELS, config.DISCRETE_COLS, config.S_PREFIX)
df = attach_discrete_combo_id(df, valid_combos, config.DISCRETE_COLS)

x_raw, y_class, y_tmax = make_xy(df, config.CONTINUOUS_COLS, config.DISCRETE_COLS, config.TPNoTP_COL, config.TMAX_COL)
pre = build_preprocessor(config.CONTINUOUS_COLS, config.DISCRETE_COLS)
X = pre.fit_transform(x_raw)
y = y_class

n_features = X.shape[1]
print(f"Data: {len(y)} samples, {n_features} features")
print(f"Class balance: TP={sum(y==1)}, NoTP={sum(y==0)}")

# Define kernels
def make_isotropic_kernel(kernel_type="Matern52"):
    length_scale = 1.0  # scalar (isotropic)
    if kernel_type == "Matern52":
        base = Matern(length_scale=length_scale, nu=2.5, length_scale_bounds=(1e-2, 1e2))
    else:
        base = RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2))
    return C(1.0, (1e-3, 1e3)) * base

def make_ard_kernel(n_features, kernel_type="Matern52"):
    length_scale = np.ones(n_features)  # vector (ARD)
    if kernel_type == "Matern52":
        base = Matern(length_scale=length_scale, nu=2.5, length_scale_bounds=(1e-2, 1e2))
    else:
        base = RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2))
    return C(1.0, (1e-3, 1e3)) * base

# Cross-validation comparison
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {"Isotropic": [], "ARD": []}

print("\n=== 5-Fold CV Comparison ===")
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    for name, kernel_fn in [("Isotropic", make_isotropic_kernel), ("ARD", lambda: make_ard_kernel(n_features))]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            clf = GaussianProcessClassifier(
                kernel=kernel_fn(),
                random_state=42,
                n_restarts_optimizer=3,
                max_iter_predict=100,
            )
            clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]
        
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_prob)
        
        results[name].append({"fold": fold+1, "acc": acc, "f1": f1, "auc": auc})
    
    print(f"Fold {fold+1}: Iso acc={results['Isotropic'][-1]['acc']:.3f}, ARD acc={results['ARD'][-1]['acc']:.3f}")

# Summary
print("\n=== Summary ===")
for name in ["Isotropic", "ARD"]:
    accs = [r["acc"] for r in results[name]]
    f1s = [r["f1"] for r in results[name]]
    aucs = [r["auc"] for r in results[name]]
    print(f"{name}:")
    print(f"  Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  F1:       {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

# Check if ARD is better or equal
iso_acc = np.mean([r["acc"] for r in results["Isotropic"]])
ard_acc = np.mean([r["acc"] for r in results["ARD"]])
iso_f1 = np.mean([r["f1"] for r in results["Isotropic"]])
ard_f1 = np.mean([r["f1"] for r in results["ARD"]])

print("\n=== Recommendation ===")
if ard_acc >= iso_acc - 0.01 and ard_f1 >= iso_f1 - 0.01:
    print("✅ ARD kernel is safe to use (no significant performance drop)")
    print("   ARD provides variable importance learning capability")
else:
    print("⚠️ ARD kernel shows performance drop, keep Isotropic")

# Compare learned length scales with ARD
print("\n=== ARD Length Scales (full data fit) ===")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    clf_ard = GaussianProcessClassifier(
        kernel=make_ard_kernel(n_features),
        random_state=42,
        n_restarts_optimizer=5,
        max_iter_predict=100,
    )
    clf_ard.fit(X, y)

learned_kernel = clf_ard.kernel_
print(f"Learned kernel: {learned_kernel}")

# Extract length scales
try:
    ls = learned_kernel.k2.length_scale
    if hasattr(ls, '__len__'):
        print(f"\nLength scales per feature (lower = more important):")
        # Get feature names
        cont_features = config.CONTINUOUS_COLS
        cat_features = []
        for col in config.DISCRETE_COLS:
            cats = config.DISCRETE_LEVELS.get(col, [])
            cat_features.extend([f"{col}_{c}" for c in cats])
        all_features = cont_features + cat_features
        
        for i, (feat, l) in enumerate(zip(all_features[:len(ls)], ls)):
            importance = 1.0 / l  # inverse of length scale
            print(f"  {feat}: length_scale={l:.3f}, importance={importance:.3f}")
except Exception as e:
    print(f"Could not extract length scales: {e}")
