# vorp.py
import pandas as pd


def compute_vorp(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate value over a same-season, same-position replacement player."""
    df = df.copy()
    year_col = next(
        (column for column in ("year", "year_results", "year_adp") if column in df),
        None,
    )
    if year_col is None:
        raise KeyError("VORP scoring requires a year column.")

    groups = []
    for (_, pos), group in df.groupby([year_col, "pos"], sort=False):
        replacement = group["projection"].sort_values(ascending=False).reset_index(drop=True)
        replacement_rank = {
            "QB": 12,
            "RB": 24,
            "WR": 36,
            "TE": 12
        }.get(pos, 24)

        if len(replacement) < replacement_rank:
            repl_value = replacement.iloc[-1]
        else:
            repl_value = replacement.iloc[replacement_rank-1]

        group = group.copy()
        group["vorp"] = group["projection"] - repl_value
        groups.append(group)

    return pd.concat(groups, ignore_index=True)
