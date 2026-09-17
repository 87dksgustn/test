from itertools import product
import pandas as pd

def is_s_level(value, s_prefix="S"):
    return str(value).startswith(s_prefix)

def is_valid_discrete_combo(combo, discrete_cols, s_prefix="S"):
    if len(discrete_cols) < 2:
        return True
    d1 = combo[discrete_cols[0]]
    d2 = combo[discrete_cols[1]]
    if is_s_level(d1, s_prefix) and is_s_level(d2, s_prefix):
        return str(d1) == str(d2)
    return True

def generate_valid_discrete_combinations(discrete_levels, discrete_cols, s_prefix="S"):
    rows = []
    for values in product(*[discrete_levels[c] for c in discrete_cols]):
        combo = dict(zip(discrete_cols, values))
        if is_valid_discrete_combo(combo, discrete_cols, s_prefix):
            rows.append(combo)
    df = pd.DataFrame(rows)
    df["discrete_combo_id"] = ["combo_" + str(i + 1).zfill(3) for i in range(len(df))]
    return df

def attach_discrete_combo_id(df, valid_combos, discrete_cols):
    out = df.merge(valid_combos[discrete_cols + ["discrete_combo_id"]], on=discrete_cols, how="left")
    missing = out["discrete_combo_id"].isna().sum()
    if missing:
        bad = out.loc[out["discrete_combo_id"].isna(), discrete_cols].drop_duplicates()
        raise ValueError(f"{missing} rows do not match valid discrete combinations.\n{bad.head(10)}")
    return out
