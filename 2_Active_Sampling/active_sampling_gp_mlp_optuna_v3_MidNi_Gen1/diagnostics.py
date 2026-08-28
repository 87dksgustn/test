import pandas as pd

def combo_diagnostics(df, valid_combos, discrete_cols, passfail_col, tmax_col, pass_label, fail_label):
    rows = []
    for _, combo in valid_combos.iterrows():
        mask = pd.Series(True, index=df.index)
        for col in discrete_cols:
            mask &= df[col].astype(str).eq(str(combo[col]))
        part = df.loc[mask]
        n = len(part)
        n_notp = int((part[passfail_col] == pass_label).sum()) if n else 0
        n_tp = int((part[passfail_col] == fail_label).sum()) if n else 0
        notp_tmax = part.loc[part[passfail_col] == pass_label, tmax_col]
        rows.append({
            "discrete_combo_id": combo["discrete_combo_id"],
            **{col: combo[col] for col in discrete_cols},
            "n_total": n,
            "n_notp": n_notp,
            "n_tp": n_tp,
            "notp_ratio": n_notp / n if n else None,
            "tp_ratio": n_tp / n if n else None,
            "notp_tmax_min": notp_tmax.min() if len(notp_tmax) else None,
            "notp_tmax_mean": notp_tmax.mean() if len(notp_tmax) else None,
            "notp_tmax_max": notp_tmax.max() if len(notp_tmax) else None,
        })
    return pd.DataFrame(rows)
