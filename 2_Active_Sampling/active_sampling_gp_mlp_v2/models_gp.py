import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.exceptions import ConvergenceWarning


class GPModels:
    def __init__(self, clf, reg_tmax, has_tmax_model: bool):
        self.clf = clf
        self.reg_tmax = reg_tmax
        self.has_tmax_model = has_tmax_model


def fit_gpc_passfail(x_train, y_class, random_state: int = 42):
    """
    GPC for PASS/FAIL.
    Label convention:
      PASS = 0
      FAIL = 1
    """
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

    clf = GaussianProcessClassifier(
        kernel=kernel,
        random_state=random_state,
        n_restarts_optimizer=3,
        max_iter_predict=100,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        clf.fit(x_train, y_class)

    return clf


def fit_gpr_tmax_given_pass(
    x_train,
    y_tmax,
    y_class,
    pass_label: int = 0,
    min_pass_samples: int = 8,
    random_state: int = 42,
):
    """
    Tmax regression model is trained only on PASS cases.
    If there are too few PASS samples, returns (None, False).
    """
    pass_mask = y_class == pass_label
    n_pass = int(pass_mask.sum())

    if n_pass < min_pass_samples:
        return None, False

    x_pass = x_train[pass_mask]
    y_pass = y_tmax[pass_mask]

    kernel = (
        C(1.0, (1e-3, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e1))
    )

    reg = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=random_state,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        reg.fit(x_pass, y_pass)

    return reg, True


def fit_gp_models(
    x_train,
    y_class,
    y_tmax,
    pass_label: int = 0,
    random_state: int = 42,
):
    clf = fit_gpc_passfail(x_train, y_class, random_state=random_state)
    reg_tmax, has_tmax_model = fit_gpr_tmax_given_pass(
        x_train=x_train,
        y_tmax=y_tmax,
        y_class=y_class,
        pass_label=pass_label,
        random_state=random_state,
    )
    return GPModels(clf=clf, reg_tmax=reg_tmax, has_tmax_model=has_tmax_model)
