import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def evaluate_gpc_cv(x_transformed, y_class, fail_label=1, n_splits=5):
    unique, counts = np.unique(y_class, return_counts=True)
    if len(unique) < 2:
        return {"error": "Only one class is present."}
    splits = max(2, min(n_splits, int(counts.min())))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    clf = GaussianProcessClassifier(kernel=kernel, random_state=42, n_restarts_optimizer=1, max_iter_predict=100)
    y_pred = cross_val_predict(clf, x_transformed, y_class, cv=cv)
    cm = confusion_matrix(y_class, y_pred, labels=[0, 1])
    return {
        "cv_splits": splits,
        "accuracy": accuracy_score(y_class, y_pred),
        "fail_precision": precision_score(y_class, y_pred, pos_label=fail_label, zero_division=0),
        "fail_recall": recall_score(y_class, y_pred, pos_label=fail_label, zero_division=0),
        "fail_f1": f1_score(y_class, y_pred, pos_label=fail_label, zero_division=0),
        "confusion_matrix_labels_PASS0_FAIL1": cm.tolist(),
    }


def model_score(metric_dict, weights):
    if not metric_dict or "error" in metric_dict:
        return float("-inf")
    return sum(float(w) * float(metric_dict.get(k, 0.0)) for k, w in weights.items())
