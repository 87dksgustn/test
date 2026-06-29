import pandas as pd


def combo_diagnostics(
    df: pd.DataFrame,
    valid_combos: pd.DataFrame,
    discrete_cols: list[str],
    passfail_col: str,
    tmax_col: str,
    pass_label: int,
    fail_label: int,
) -> pd.DataFrame:
    rows = []

    for _, combo in valid_combos.iterrows():
        combo_id = combo["discrete_combo_id"]
        mask = pd.Series(True, index=df.index)
        for col in discrete_cols:
            mask &= df[col].astype(str).eq(str(combo[col]))

        part = df.loc[mask]
        n = len(part)
        n_pass = int((part[passfail_col] == pass_label).sum()) if n else 0
        n_fail = int((part[passfail_col] == fail_label).sum()) if n else 0

        pass_tmax = part.loc[part[passfail_col] == pass_label, tmax_col]
        rows.append(
            {
                "discrete_combo_id": combo_id,
                **{col: combo[col] for col in discrete_cols},
                "n_total": n,
                "n_pass": n_pass,
                "n_fail": n_fail,
                "pass_ratio": n_pass / n if n else None,
                "fail_ratio": n_fail / n if n else None,
                "pass_tmax_min": pass_tmax.min() if len(pass_tmax) else None,
                "pass_tmax_mean": pass_tmax.mean() if len(pass_tmax) else None,
                "pass_tmax_max": pass_tmax.max() if len(pass_tmax) else None,
            }
        )

    return pd.DataFrame(rows)
