# cost.py
import numpy as np
import pandas as pd

# Expected fantasy points based on ADP slot
# This curve is tuned to mimic how real draft value decreases by pick
def expected_points_from_adp(adp: float) -> float:
    if pd.isna(adp):
        return 0.0
    
    # The exponential decay curve:
    # - top picks expected to score ~250+
    # - by pick 100 expected score ~100
    # - by pick 150 expected score ~70
    return 260 * np.exp(-0.028 * (adp - 1))


def compute_cost(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Estimate draft round (helpful later in UI)
    df["round"] = (df["adp_avg"] // 12 + 1).astype(int)

    # Expected points AT the player's ADP slot
    df["expected_points_at_cost"] = df["adp_avg"].apply(expected_points_from_adp)

    # Value over draft cost
    df["value_over_cost"] = df["projection"] - df["expected_points_at_cost"]

    return df