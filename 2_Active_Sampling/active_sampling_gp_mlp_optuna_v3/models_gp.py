import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

class GPModels:
    kind = "gp"
    def __init__(self, clf, reg_tmax, has_tmax_model, gp_params=None, tmax_params=None, clf_ensemble=None):
        self.clf = clf
        self.reg_tmax = reg_tmax
        self.has_tmax_model = has_tmax_model
        self.gp_params = gp_params or {}
        self.tmax_params = tmax_params or {}
        self.clf_ensemble = clf_ensemble or []

def build_clf_kernel(params=None, n_features=None):
    """Build GP classifier kernel.
    
    Args:
        params: Kernel hyperparameters dict
        n_features: Number of input features. If provided, uses ARD (per-feature length scales).
                   If None, uses isotropic (single length scale for all features).
    """
    params = params or {}
    constant = float(params.get("constant", 1.0))
    kernel_type = params.get("kernel", "Matern52")  # Default to Matern52 for better smoothness
    
    # ARD mode: per-feature length scales (enables variable importance learning)
    if n_features is not None and n_features > 0:
        length_scale = np.ones(n_features)  # Will be optimized per feature
    else:
        length_scale = float(params.get("length_scale", 1.0))  # Isotropic fallback
    
    if kernel_type == "Matern32":
        base = Matern(length_scale=length_scale, nu=1.5, length_scale_bounds=(1e-2, 1e2))
    elif kernel_type == "Matern52":
        base = Matern(length_scale=length_scale, nu=2.5, length_scale_bounds=(1e-2, 1e2))
    else:
        base = RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2))
    return C(constant, (1e-3, 1e3)) * base

def build_tmax_kernel(params=None, n_features=None):
    """Build GP regressor kernel for Tmax prediction.
    
    Args:
        params: Kernel hyperparameters dict
        n_features: Number of input features. If provided, uses ARD.
    """
    params = params or {}
    constant = float(params.get("constant", 1.0))
    noise = float(params.get("noise_level", 1e-5))
    kernel_type = params.get("kernel", "Matern52")
    
    # ARD mode
    if n_features is not None and n_features > 0:
        length_scale = np.ones(n_features)
    else:
        length_scale = float(params.get("length_scale", 1.0))
    
    if kernel_type == "Matern32":
        base = Matern(length_scale=length_scale, nu=1.5, length_scale_bounds=(1e-2, 1e2))
    elif kernel_type == "Matern52":
        base = Matern(length_scale=length_scale, nu=2.5, length_scale_bounds=(1e-2, 1e2))
    else:
        base = RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2))
    return C(constant, (1e-3, 1e3)) * base + WhiteKernel(noise_level=noise, noise_level_bounds=(1e-8, 1e1))

def fit_gpc_passfail(x_train, y_class, random_state=42, params=None, use_ard=True):
    """Fit GP classifier for TP/NoTP classification.
    
    Args:
        x_train: Training features (n_samples, n_features)
        y_class: Training labels
        random_state: Random seed
        params: Kernel hyperparameters
        use_ard: If True, use ARD kernel (per-feature length scales)
    """
    params = params or {}
    n_features = x_train.shape[1] if use_ard else None
    clf = GaussianProcessClassifier(
        kernel=build_clf_kernel(params, n_features=n_features),
        random_state=random_state,
        n_restarts_optimizer=int(params.get("n_restarts_optimizer", 5)),  # Increased for ARD
        max_iter_predict=100,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        clf.fit(x_train, y_class)
    return clf

def _stratified_bootstrap_indices(y, n_samples, rng):
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) <= 1:
        return rng.integers(0, len(y), size=n_samples)
    target_counts = []
    for c in counts:
        target_counts.append(max(1, int(round(n_samples * (c / len(y))))))
    diff = int(n_samples - sum(target_counts))
    if diff != 0:
        order = np.argsort(-counts)
        step = 1 if diff > 0 else -1
        k = 0
        while diff != 0 and k < 10000:
            idx = int(order[k % len(order)])
            if step > 0 or target_counts[idx] > 1:
                target_counts[idx] += step
                diff -= step
            k += 1
    sampled = []
    for cls, n_cls in zip(classes, target_counts):
        cls_idx = np.where(y == cls)[0]
        sampled.append(rng.choice(cls_idx, size=int(n_cls), replace=True))
    out = np.concatenate(sampled)
    rng.shuffle(out)
    return out

def fit_gpc_ensemble_passfail(x_train, y_class, n_members=5, sample_ratio=0.8, stratified=True, random_state=42, params=None, use_ard=True):
    """Fit ensemble of GP classifiers for uncertainty estimation."""
    n_members = max(1, int(n_members))
    n_total = int(len(y_class))
    n_boot = max(2, min(n_total, int(round(float(sample_ratio) * n_total))))
    ensemble = []
    for i in range(n_members):
        rng = np.random.default_rng(int(random_state) + i * 9973)
        if stratified:
            idx = _stratified_bootstrap_indices(y_class, n_boot, rng)
        else:
            idx = rng.integers(0, n_total, size=n_boot)
        clf_i = fit_gpc_passfail(x_train[idx], y_class[idx], random_state=int(random_state) + i + 1, params=params, use_ard=use_ard)
        ensemble.append(clf_i)
    return ensemble

def fit_gpr_tmax_given_pass(x_train, y_tmax, y_class, pass_label=0, min_pass_samples=8, random_state=42, params=None, use_ard=True):
    """Fit GP regressor for Tmax prediction on NoTP samples."""
    params = params or {}
    mask = y_class == pass_label
    if int(mask.sum()) < min_pass_samples:
        return None, False
    n_features = x_train.shape[1] if use_ard else None
    reg = GaussianProcessRegressor(
        kernel=build_tmax_kernel(params, n_features=n_features),
        alpha=float(params.get("alpha", 1e-8)),
        normalize_y=bool(params.get("normalize_y", True)),
        n_restarts_optimizer=int(params.get("n_restarts_optimizer", 5)),  # Increased for ARD
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        reg.fit(x_train[mask], y_tmax[mask])
    return reg, True

def fit_gp_models(
    x_train,
    y_class,
    y_tmax,
    pass_label=0,
    random_state=42,
    gp_params=None,
    tmax_params=None,
    clf_uncertainty_mode="none",
    clf_ensemble_size=5,
    clf_ensemble_sample_ratio=0.8,
    clf_ensemble_stratified=True,
    use_ard=True,
):
    """Fit GP models for classification and Tmax regression.
    
    Args:
        use_ard: If True, use ARD kernel (Automatic Relevance Determination).
                 This enables per-feature length scale learning for better
                 variable importance detection and interaction modeling.
    """
    clf = fit_gpc_passfail(x_train, y_class, random_state=random_state, params=gp_params, use_ard=use_ard)
    reg, has = fit_gpr_tmax_given_pass(x_train, y_tmax, y_class, pass_label=pass_label, random_state=random_state, params=tmax_params, use_ard=use_ard)
    clf_ensemble = []
    if str(clf_uncertainty_mode).strip().lower() == "ensemble_std":
        clf_ensemble = fit_gpc_ensemble_passfail(
            x_train,
            y_class,
            n_members=clf_ensemble_size,
            sample_ratio=clf_ensemble_sample_ratio,
            stratified=bool(clf_ensemble_stratified),
            random_state=random_state,
            params=gp_params,
            use_ard=use_ard,
        )
    return GPModels(
        clf=clf,
        reg_tmax=reg,
        has_tmax_model=has,
        gp_params=gp_params,
        tmax_params=tmax_params,
        clf_ensemble=clf_ensemble,
    )
