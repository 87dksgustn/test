import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


@dataclass
class TargetScaler:
    mean: float
    std: float
    def transform(self, y):
        return (y - self.mean) / self.std
    def inverse_transform(self, y):
        return y * self.std + self.mean


if TORCH_AVAILABLE:
    class MultiHeadMLP(nn.Module):
        def __init__(self, input_dim, hidden_dims, dropout=0.1, n_extra_outputs=0):
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            self.trunk = nn.Sequential(*layers)
            self.class_head = nn.Linear(prev, 2)
            self.tmax_head = nn.Linear(prev, 1)
            self.extra_head = nn.Linear(prev, n_extra_outputs) if n_extra_outputs > 0 else None
        def forward(self, x):
            z = self.trunk(x)
            logits = self.class_head(z)
            tmax = self.tmax_head(z).squeeze(-1)
            extra = self.extra_head(z) if self.extra_head is not None else None
            return logits, tmax, extra
else:
    class MultiHeadMLP:  # placeholder
        pass


class MLPEnsemble:
    kind = "mlp"
    def __init__(self, models, tmax_scaler, extra_scaler, device, extra_cols=None):
        self.models = models
        self.tmax_scaler = tmax_scaler
        self.extra_scaler = extra_scaler
        self.device = device
        self.extra_cols = extra_cols or []
        self.has_tmax_model = True


def _check_torch():
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for MLP ensemble. Install with: pip install torch")


def _make_tmax_scaler(y_tmax, y_class, pass_label):
    y = np.asarray(y_tmax)[np.asarray(y_class) == pass_label].astype(float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return TargetScaler(0.0, 1.0)
    mean = float(np.mean(y)); std = float(np.std(y))
    if std < 1e-12: std = 1.0
    return TargetScaler(mean, std)


def _make_extra_scaler(y_extra):
    if y_extra is None or y_extra.shape[1] == 0:
        return None
    mean = np.nanmean(y_extra, axis=0)
    std = np.nanstd(y_extra, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return {"mean": mean, "std": std}


def _transform_extra(y_extra, scaler):
    if y_extra is None or scaler is None:
        return None
    return (y_extra - scaler["mean"]) / scaler["std"]


def _class_weights(y_class):
    counts = np.bincount(y_class.astype(int), minlength=2).astype(float)
    counts = np.where(counts <= 0, 1.0, counts)
    return (counts.sum() / (2.0 * counts)).astype(np.float32)


def _train_single_mlp(x_train, y_class, y_tmax, y_extra, config, seed, max_epochs=None):
    _check_torch()
    torch.manual_seed(seed); np.random.seed(seed)
    max_epochs = max_epochs or config.MLP_MAX_EPOCHS
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tmax_scaler = _make_tmax_scaler(y_tmax, y_class, config.PASS_LABEL)
    y_tmax_s = tmax_scaler.transform(np.asarray(y_tmax, dtype=float))
    extra_scaler = _make_extra_scaler(y_extra)
    y_extra_s = _transform_extra(y_extra, extra_scaler)
    n_extra = 0 if y_extra_s is None else y_extra_s.shape[1]

    idx = np.arange(len(y_class))
    stratify = y_class if len(np.unique(y_class)) == 2 else None
    tr_idx, va_idx = train_test_split(idx, test_size=config.MLP_VALID_FRACTION, random_state=seed, stratify=stratify)

    def ft(a): return torch.tensor(a, dtype=torch.float32)
    x_tr = ft(x_train[tr_idx]); y_tr = torch.tensor(y_class[tr_idx], dtype=torch.long)
    t_tr = ft(y_tmax_s[tr_idx]); pass_tr = torch.tensor(y_class[tr_idx] == config.PASS_LABEL, dtype=torch.bool)
    if y_extra_s is not None:
        e_np = y_extra_s[tr_idx]
        e_tr = ft(np.nan_to_num(e_np, nan=0.0)); em_tr = torch.tensor(np.isfinite(e_np), dtype=torch.bool)
        ds = TensorDataset(x_tr, y_tr, t_tr, pass_tr, e_tr, em_tr)
    else:
        ds = TensorDataset(x_tr, y_tr, t_tr, pass_tr)
    loader = DataLoader(ds, batch_size=config.MLP_BATCH_SIZE, shuffle=True)

    x_va = ft(x_train[va_idx]).to(device); y_va = torch.tensor(y_class[va_idx], dtype=torch.long).to(device)
    t_va = ft(y_tmax_s[va_idx]).to(device); pass_va = torch.tensor(y_class[va_idx] == config.PASS_LABEL, dtype=torch.bool).to(device)
    if y_extra_s is not None:
        e_np_va = y_extra_s[va_idx]
        e_va = ft(np.nan_to_num(e_np_va, nan=0.0)).to(device); em_va = torch.tensor(np.isfinite(e_np_va), dtype=torch.bool).to(device)
    else:
        e_va = em_va = None

    model = MultiHeadMLP(x_train.shape[1], config.MLP_HIDDEN_DIMS, config.MLP_DROPOUT, n_extra).to(device)
    weights = torch.tensor(_class_weights(y_class), dtype=torch.float32).to(device) if config.MLP_USE_CLASS_WEIGHT else None
    opt = torch.optim.AdamW(model.parameters(), lr=config.MLP_LEARNING_RATE, weight_decay=config.MLP_WEIGHT_DECAY)
    best_state = None; best_loss = float("inf"); wait = 0

    for _ in range(max_epochs):
        model.train()
        for batch in loader:
            batch = [b.to(device) for b in batch]
            if y_extra_s is not None:
                xb, yb, tb, pb, eb, emb = batch
            else:
                xb, yb, tb, pb = batch; eb = emb = None
            logits, tpred, epred = model(xb)
            loss = config.MLP_CLASSIFICATION_LOSS_WEIGHT * F.cross_entropy(logits, yb, weight=weights)
            if pb.any():
                loss = loss + config.MLP_TMAX_LOSS_WEIGHT * F.mse_loss(tpred[pb], tb[pb])
            if epred is not None and emb is not None and emb.any():
                loss = loss + config.MLP_OTHER_REGRESSION_LOSS_WEIGHT * torch.mean(((epred - eb)[emb]) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            logits, tpred, epred = model(x_va)
            vloss = F.cross_entropy(logits, y_va, weight=weights)
            if pass_va.any():
                vloss = vloss + config.MLP_TMAX_LOSS_WEIGHT * F.mse_loss(tpred[pass_va], t_va[pass_va])
            if epred is not None and em_va is not None and em_va.any():
                vloss = vloss + config.MLP_OTHER_REGRESSION_LOSS_WEIGHT * torch.mean(((epred - e_va)[em_va]) ** 2)
            v = float(vloss.cpu())
        if v < best_loss - 1e-6:
            best_loss = v; best_state = {k: val.detach().cpu().clone() for k, val in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= config.MLP_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, tmax_scaler, extra_scaler, device


def fit_mlp_ensemble(x_train, y_class, y_tmax, y_extra, extra_cols, config, seed=42):
    _check_torch()
    models=[]; ts=None; es=None; dev="cpu"
    for i in range(config.MLP_ENSEMBLE_SIZE):
        m, ts, es, dev = _train_single_mlp(x_train, y_class, y_tmax, y_extra, config, seed + i*997, config.MLP_MAX_EPOCHS)
        models.append(m)
    return MLPEnsemble(models, ts, es, dev, extra_cols)


def predict_mlp_ensemble(bundle, x):
    _check_torch()
    xt = torch.tensor(x, dtype=torch.float32).to(bundle.device)
    p_list=[]; t_list=[]
    for m in bundle.models:
        m.eval()
        with torch.no_grad():
            logits, t_s, _ = m(xt)
            p = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            t = bundle.tmax_scaler.inverse_transform(t_s.cpu().numpy())
        p_list.append(p); t_list.append(t)
    p_arr=np.vstack(p_list); t_arr=np.vstack(t_list)
    p_mean=p_arr.mean(axis=0)
    return {"p_fail":p_mean, "p_pass":1-p_mean, "p_fail_std":p_arr.std(axis=0), "tmax_pred":t_arr.mean(axis=0), "tmax_std":t_arr.std(axis=0)}


def evaluate_mlp_cv(x_transformed, y_class, y_tmax, y_extra, extra_cols, config, fail_label=1, n_splits=5):
    _check_torch()
    uniq, counts = np.unique(y_class, return_counts=True)
    if len(uniq) < 2:
        return {"error":"Only one class is present."}
    splits = max(2, min(n_splits, int(counts.min())))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=config.RANDOM_SEED)
    y_pred_all=np.zeros_like(y_class)
    for fold,(tr,va) in enumerate(cv.split(x_transformed, y_class)):
        m, ts, es, dev = _train_single_mlp(x_transformed[tr], y_class[tr], y_tmax[tr], None if y_extra is None else y_extra[tr], config, config.RANDOM_SEED+fold*101, config.MLP_CV_MAX_EPOCHS)
        bundle = MLPEnsemble([m], ts, es, dev, extra_cols)
        pred = predict_mlp_ensemble(bundle, x_transformed[va])
        y_pred_all[va] = (pred["p_fail"] >= 0.5).astype(int)
    cm = confusion_matrix(y_class, y_pred_all, labels=[0,1])
    return {"cv_splits":splits, "accuracy":accuracy_score(y_class,y_pred_all), "fail_precision":precision_score(y_class,y_pred_all,pos_label=fail_label,zero_division=0), "fail_recall":recall_score(y_class,y_pred_all,pos_label=fail_label,zero_division=0), "fail_f1":f1_score(y_class,y_pred_all,pos_label=fail_label,zero_division=0), "confusion_matrix_labels_PASS0_FAIL1":cm.tolist()}
