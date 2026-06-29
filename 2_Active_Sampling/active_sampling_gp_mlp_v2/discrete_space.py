from itertools import product
import pandas as pd


def is_s_level(value: object, s_prefix: str = "S") -> bool:
    return str(value).startswith(s_prefix)


def is_valid_discrete_combo(combo: dict, discrete_cols: list[str], s_prefix: str = "S") -> bool:
    """
    Constraint for disc1 and disc2:
    If both are S-series, they must be identical.
    This keeps:
      S1/S1, S2/S2, S3/S3
    and rejects:
      S1/S2, S2/S3, ...
    S/P is allowed because only both-S mismatch is rejected.
    """
    if len(discrete_cols) < 2:
        return True

    d1 = combo[discrete_cols[0]]
    d2 = combo[discrete_cols[1]]

    if is_s_level(d1, s_prefix) and is_s_level(d2, s_prefix):
        return str(d1) == str(d2)

    return True


def generate_valid_discrete_combinations(
    discrete_levels: dict[str, list],
    discrete_cols: list[str],
    s_prefix: str = "S",
) -> pd.DataFrame:
    rows = []
    levels = [discrete_levels[col] for col in discrete_cols]

    for values in product(*levels):
        combo = dict(zip(discrete_cols, values))
        if is_valid_discrete_combo(combo, discrete_cols, s_prefix):
            rows.append(combo)

    df = pd.DataFrame(rows)
    df["discrete_combo_id"] = [
        "combo_" + str(i + 1).zfill(3) for i in range(len(df))
    ]
    return df


def attach_discrete_combo_id(
    df: pd.DataFrame,
    valid_combos: pd.DataFrame,
    discrete_cols: list[str],
) -> pd.DataFrame:
    """
    Adds discrete_combo_id to df by merging with valid_combos.
    Raises error if any row does not match a valid combo.
    """
    out = df.merge(
        valid_combos[discrete_cols + ["discrete_combo_id"]],
        on=discrete_cols,
        how="left",
    )

    missing = out["discrete_combo_id"].isna().sum()
    if missing > 0:
        bad = out.loc[out["discrete_combo_id"].isna(), discrete_cols].drop_duplicates()
        raise ValueError(
            f"{missing} rows do not match valid discrete combinations.\n"
            f"Examples:\n{bad.head(10)}"
        )

    return out
