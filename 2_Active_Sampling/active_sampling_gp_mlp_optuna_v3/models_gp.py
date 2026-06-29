import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

class GPModels:
    kind = "gp"
    def __init__(self, clf, reg_tmax, has_tmax_model, gp_params=None, tmax_params=None):
        self.clf = clf
        self.reg_tmax = reg_tmax
        self.has_tmax_model = has_tmax_model
        self.gp_params = gp_params or {}
        self.tmax_params = tmax_params or {}

def build_clf_kernel(params=None):
    params = params or {}
    constant = float(params.get("constant", 1.0))
    length_scale = float(params.get("length_scale", 1.0))
    kernel_type = params.get("kernel", "RBF")
    if kernel_type == "Matern32":
        base = Matern(length_scale=length_scale, nu=1.5, length_scale_bounds=(1e-2, 1e2))
    elif kernel_type == "Matern52":
        base = Matern(length_scale=length_scale, nu=2.5, length_scale_bounds=(1e-2, 1e2))
    else:
        base = RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2))
    return C(constant, (1e-3, 1e3)) * base

def build_tmax_kernel(params=None):
    params = params or {}
    constant = float(params.get("constant", 1.0))
    length_scale = float(params.get("length_scale", 1.0))
    noise = float(params.get("noise_level", 1e-5))
    kernel_type = params.get("kernel", "RBF")
    if kernel_type == "Matern32":
        base = Matern(length_scale=length_scale, nu=1.5, length_scale_bounds=(1e-2, 1e2))
    elif kernel_type == "Matern52":
        base = Matern(length_scale=length_scale, nu=2.5, length_scale_bounds=(1e-2, 1e2))
    else:
        base = RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2))
    return C(constant, (1e-3, 1e3)) * base + WhiteKernel(noise_level=noise, noise_level_bounds=(1e-8, 1e1))

def fit_gpc_passfail(x_train, y_class, random_state=42, params=None):
    params = params or {}
    clf = GaussianProcessClassifier(
        kernel=build_clf_kernel(params),
        random_state=random_state,
        n_restarts_optimizer=int(params.get("n_restarts_optimizer", 3)),
        max_iter_predict=100,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        clf.fit(x_train, y_class)
    return clf

def fit_gpr_tmax_given_pass(x_train, y_tmax, y_class, pass_label=0, min_pass_samples=8, random_state=42, params=None):
    params = params or {}
    mask = y_class == pass_label
    if int(mask.sum()) < min_pass_samples:
        return None, False
    reg = GaussianProcessRegressor(
        kernel=build_tmax_kernel(params),
        alpha=float(params.get("alpha", 1e-8)),
        normalize_y=bool(params.get("normalize_y", True)),
        n_restarts_optimizer=int(params.get("n_restarts_optimizer", 3)),
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        reg.fit(x_train[mask], y_tmax[mask])
    return reg, True

def fit_gp_models(x_train, y_class, y_tmax, pass_label=0, random_state=42, gp_params=None, tmax_params=None):
    clf = fit_gpc_passfail(x_train, y_class, random_state=random_state, params=gp_params)
    reg, has = fit_gpr_tmax_given_pass(x_train, y_tmax, y_class, pass_label=pass_label, random_state=random_state, params=tmax_params)
    return GPModels(clf=clf, reg_tmax=reg, has_tmax_model=has, gp_params=gp_params, tmax_params=tmax_params)
