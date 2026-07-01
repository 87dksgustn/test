import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, StratifiedKFold
from metrics_utils import classification_metrics, regression_metrics, stable_metric_summary

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
    def transform(self, y): return (y - self.mean) / self.std
    def inverse_transform(self, y): return y * self.std + self.mean

if TORCH_AVAILABLE:
    class MultiHeadMLP(nn.Module):
        def __init__(self, input_dim, hidden_dims, dropout=0.1, n_extra_outputs=0):
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers += [nn.Linear(prev, int(h)), nn.ReLU(), nn.Dropout(float(dropout))]
                prev = int(h)
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
    class MultiHeadMLP: pass

class MLPEnsemble:
    kind = "mlp"
    def __init__(self, models, tmax_scaler, extra_scaler, device, params=None):
        self.models = models
        self.tmax_scaler = tmax_scaler
        self.extra_scaler = extra_scaler
        self.device = device
        self.params = params or {}
        self.has_tmax_model = True

def _check_torch():
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for MLP ensemble. Install with: pip install torch")

def _cfg(config, params, key):
    return params[key] if params and key in params else getattr(config, key)

def _tmax_scaler(y_tmax, y_class, pass_label):
    y = np.asarray(y_tmax[y_class == pass_label], dtype=float)
    y = y[np.isfinite(y)]
    if len(y) == 0: return TargetScaler(0.0, 1.0)
    mean = float(np.mean(y)); std = float(np.std(y))
    return TargetScaler(mean, std if std > 1e-12 else 1.0)

def _extra_scaler(y_extra):
    if y_extra is None or y_extra.shape[1] == 0: return None
    mean = np.nanmean(y_extra, axis=0); std = np.nanstd(y_extra, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return {"mean": mean, "std": std}

def _class_weights(y_class):
    counts = np.bincount(y_class.astype(int), minlength=2).astype(float)
    counts = np.where(counts <= 0, 1.0, counts)
    return (counts.sum() / (2 * counts)).astype(np.float32)

def _bootstrap_indices(y_class, seed, stratified=True, sample_ratio=1.0):
    rng = np.random.default_rng(seed)
    n = len(y_class)
    sample_n = max(2, int(round(float(sample_ratio) * n)))
    if not stratified:
        return rng.choice(n, size=sample_n, replace=True)

    y = np.asarray(y_class)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return rng.choice(n, size=sample_n, replace=True)

    parts = []
    allocated = 0
    for i, cls in enumerate(classes):
        cls_idx = np.where(y == cls)[0]
        if i == len(classes) - 1:
            k = sample_n - allocated
        else:
            k = int(round(sample_n * (len(cls_idx) / n)))
            k = max(1, k)
            allocated += k
        parts.append(rng.choice(cls_idx, size=k, replace=True))
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out

def train_single_mlp(x_train, y_class, y_tmax, y_extra, config, seed=42, params=None, max_epochs=None):
    _check_torch(); torch.manual_seed(seed); np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_epochs = max_epochs or int(_cfg(config, params, "MLP_MAX_EPOCHS"))
    hidden_dims = params.get("hidden_dims", config.MLP_HIDDEN_DIMS) if params else config.MLP_HIDDEN_DIMS
    dropout = float(params.get("dropout", config.MLP_DROPOUT)) if params else config.MLP_DROPOUT
    lr = float(params.get("learning_rate", config.MLP_LEARNING_RATE)) if params else config.MLP_LEARNING_RATE
    weight_decay = float(params.get("weight_decay", config.MLP_WEIGHT_DECAY)) if params else config.MLP_WEIGHT_DECAY
    tmax_loss_weight = float(params.get("tmax_loss_weight", config.MLP_TMAX_LOSS_WEIGHT)) if params else config.MLP_TMAX_LOSS_WEIGHT
    valid_fraction = config.MLP_VALID_FRACTION
    tsc = _tmax_scaler(y_tmax, y_class, config.PASS_LABEL)
    yt = tsc.transform(np.asarray(y_tmax, dtype=float))
    esc = _extra_scaler(y_extra)
    ye = None if y_extra is None or esc is None else (y_extra - esc["mean"]) / esc["std"]
    n_extra = 0 if ye is None else ye.shape[1]
    idx = np.arange(len(y_class))
    stratify = y_class if len(np.unique(y_class)) == 2 else None
    tr, va = train_test_split(idx, test_size=valid_fraction, random_state=seed, stratify=stratify)
    def ten(a, dtype=torch.float32): return torch.tensor(a, dtype=dtype)
    xtr = ten(x_train[tr]); ytr = torch.tensor(y_class[tr], dtype=torch.long); ttr = ten(yt[tr]); ptr = torch.tensor(y_class[tr] == config.PASS_LABEL, dtype=torch.bool)
    if ye is not None:
        etr_np = ye[tr]; etr = ten(np.nan_to_num(etr_np, nan=0.0)); emtr = torch.tensor(np.isfinite(etr_np), dtype=torch.bool)
        ds = TensorDataset(xtr, ytr, ttr, ptr, etr, emtr)
    else:
        ds = TensorDataset(xtr, ytr, ttr, ptr)
    loader = DataLoader(ds, batch_size=config.MLP_BATCH_SIZE, shuffle=True)
    xva = ten(x_train[va]).to(device); yva = torch.tensor(y_class[va], dtype=torch.long).to(device); tva = ten(yt[va]).to(device); pva = torch.tensor(y_class[va] == config.PASS_LABEL, dtype=torch.bool).to(device)
    model = MultiHeadMLP(x_train.shape[1], hidden_dims, dropout, n_extra).to(device)
    weights = torch.tensor(_class_weights(y_class), dtype=torch.float32).to(device) if config.MLP_USE_CLASS_WEIGHT else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None; best = float("inf"); wait = 0
    for epoch in range(max_epochs):
        model.train()
        for batch in loader:
            batch = [b.to(device) for b in batch]
            if ye is not None: xb, yb, tb, pb, eb, emb = batch
            else: xb, yb, tb, pb = batch; eb = emb = None
            logits, tp, ep = model(xb)
            loss = config.MLP_CLASSIFICATION_LOSS_WEIGHT * F.cross_entropy(logits, yb, weight=weights)
            if pb.any(): loss = loss + tmax_loss_weight * F.mse_loss(tp[pb], tb[pb])
            if ep is not None and emb is not None and emb.any(): loss = loss + config.MLP_OTHER_REGRESSION_LOSS_WEIGHT * torch.mean(((ep - eb)[emb]) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            logits, tp, _ = model(xva)
            vloss = F.cross_entropy(logits, yva, weight=weights)
            if pva.any(): vloss = vloss + tmax_loss_weight * F.mse_loss(tp[pva], tva[pva])
            val = float(vloss.detach().cpu())
        if val < best - 1e-6:
            best = val; best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= config.MLP_PATIENCE: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    return model, tsc, esc, device

def fit_mlp_ensemble(x_train, y_class, y_tmax, y_extra, config, seed=42, params=None):
    models = []; tsc = esc = None; device = "cpu"
    n = int(params.get("ensemble_size", config.MLP_ENSEMBLE_SIZE)) if params else config.MLP_ENSEMBLE_SIZE
    use_bootstrap = bool(params.get("bootstrap", config.MLP_ENSEMBLE_BOOTSTRAP)) if params else bool(config.MLP_ENSEMBLE_BOOTSTRAP)
    bootstrap_stratified = bool(params.get("bootstrap_stratified", config.MLP_BOOTSTRAP_STRATIFIED)) if params else bool(config.MLP_BOOTSTRAP_STRATIFIED)
    bootstrap_ratio = float(params.get("bootstrap_sample_ratio", config.MLP_BOOTSTRAP_SAMPLE_RATIO)) if params else float(config.MLP_BOOTSTRAP_SAMPLE_RATIO)
    for i in range(n):
        member_seed = seed + i*997
        if use_bootstrap:
            idx = _bootstrap_indices(y_class, member_seed, stratified=bootstrap_stratified, sample_ratio=bootstrap_ratio)
            xb = x_train[idx]
            yb = y_class[idx]
            tb = y_tmax[idx]
            eb = None if y_extra is None else y_extra[idx]
            model, tsc, esc, device = train_single_mlp(xb, yb, tb, eb, config, member_seed, params=params)
        else:
            model, tsc, esc, device = train_single_mlp(x_train, y_class, y_tmax, y_extra, config, member_seed, params=params)
        models.append(model)
    return MLPEnsemble(models, tsc, esc, device, params=params)

def predict_mlp_ensemble(bundle, x):
    _check_torch(); device = bundle.device
    xt = torch.tensor(x, dtype=torch.float32).to(device)
    ps = []; ts = []
    for model in bundle.models:
        model.eval()
        with torch.no_grad():
            logits, tscaled, _ = model(xt)
            prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
            ps.append(prob[:,1]); ts.append(bundle.tmax_scaler.inverse_transform(tscaled.detach().cpu().numpy()))
    p = np.vstack(ps); t = np.vstack(ts)
    return {"p_tp": p.mean(axis=0), "p_notp": 1-p.mean(axis=0), "p_tp_std": p.std(axis=0), "tmax_pred": t.mean(axis=0), "tmax_std": t.std(axis=0)}

def evaluate_mlp_cv(x_transformed, y_class, y_tmax, y_extra, config, tp_label=1, n_splits=5, weights=None, std_penalty=0.5, params=None):
    _check_torch()
    unique, counts = np.unique(y_class, return_counts=True)
    if len(unique) < 2: return {"summary": {"error": "Only one class is present."}, "fold_metrics": []}
    splits = max(2, min(n_splits, int(counts.min())))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=config.RANDOM_SEED)
    fold_metrics = []
    for fold, (tr, va) in enumerate(cv.split(x_transformed, y_class)):
        model, tsc, esc, device = train_single_mlp(x_transformed[tr], y_class[tr], y_tmax[tr], None if y_extra is None else y_extra[tr], config, seed=config.RANDOM_SEED + fold*101, params=params, max_epochs=config.MLP_CV_MAX_EPOCHS)
        bundle = MLPEnsemble([model], tsc, esc, device, params=params)
        pred = predict_mlp_ensemble(bundle, x_transformed[va])
        ypred = (pred["p_tp"] >= 0.5).astype(int)
        m = classification_metrics(y_class[va], ypred, tp_label=tp_label)
        pass_mask = (y_class[va] == config.PASS_LABEL)
        if int(pass_mask.sum()) > 0:
            m.update(regression_metrics(y_tmax[va][pass_mask], pred["tmax_pred"][pass_mask]))
        else:
            m.update({"tmax_mae": np.nan, "tmax_rmse": np.nan, "tmax_r2": np.nan})
        m["tmax_eval_n"] = int(pass_mask.sum())
        m["model"] = "mlp"; m["fold"] = fold
        fold_metrics.append(m)
    summary = stable_metric_summary(fold_metrics, weights or {"tp_recall":0.7,"tp_f1":0.3}, std_penalty)
    summary["cv_splits"] = splits
    return {"summary": summary, "fold_metrics": fold_metrics}
