import numpy as np
import pandas as pd

# Transparent calibration points for expected season totals by ADP slot.
_ADP_ANCHORS = np.array([1, 12, 36, 60, 100, 150, 216], dtype=float)
_POINT_ANCHORS = np.array([260, 230, 190, 155, 100, 70, 45], dtype=float)


def expected_points_from_adp(adp: float) -> float:
    if pd.isna(adp):
        return 0.0
    return float(np.interp(float(adp), _ADP_ANCHORS, _POINT_ANCHORS))


def compute_cost(df: pd.DataFrame, league_size: int = 12) -> pd.DataFrame:
    """Calculate expected points and value at each player's draft cost.

    When season and position are available, the expected total is calibrated
    from that season's result distribution for the same position. The first QB
    drafted is therefore compared with QB1 scoring, rather than with an RB or
    kicker taken at a similar overall pick. The anchor curve remains a fallback
    for standalone data that lacks season/position columns.
    """
    df = df.copy()
    if league_size <= 0:
        raise ValueError("league_size must be positive.")

    df["round"] = np.ceil(df["adp_avg"] / league_size).astype(int)

    year_col = next(
        (column for column in ("year", "year_results", "year_adp") if column in df),
        None,
    )
    if year_col is not None and "pos" in df.columns:
        df["expected_points_at_cost"] = np.nan
        for _, indexes in df.groupby([year_col, "pos"]).groups.items():
            group = df.loc[indexes]
            expected_totals = np.sort(group["projection"].to_numpy())[::-1]
            draft_order = group.sort_values("adp_avg").index
            df.loc[draft_order, "expected_points_at_cost"] = expected_totals
    else:
        df["expected_points_at_cost"] = df["adp_avg"].apply(
            expected_points_from_adp
        )

    # Value over draft cost
    df["value_over_cost"] = df["projection"] - df["expected_points_at_cost"]

    return df
