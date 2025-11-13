# vorp.py
import numpy as np
import pandas as pd

# VORP (Value Over Replacement Player) calculation module
def compute_vorp(df):
    df = df.copy()

    vorp = []

    for pos, group in df.groupby("pos"):
        replacement = group["projection"].sort_values(ascending=False).reset_index(drop=True)
        
        # choose replacement rank
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

        group["vorp"] = group["projection"] - repl_value
        vorp.append(group)

    return pd.concat(vorp, ignore_index=True)