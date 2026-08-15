# scoring_engine.py
import pandas as pd
from fantasyoptimizer.utils.data_loader import build_history
from fantasyoptimizer.scoring.projection import compute_projection
from fantasyoptimizer.scoring.vorp import compute_vorp
from fantasyoptimizer.scoring.cost import compute_cost
from fantasyoptimizer.scoring.risk import compute_risk


def compute_scores(years: list[int]) -> pd.DataFrame:
    df = build_history(years)
    # Replacement levels and the dashboard currently support offensive slots.
    df = df[df["pos"].isin(["QB", "RB", "WR", "TE"])].copy()
    if df.empty:
        raise ValueError("No supported QB, RB, WR, or TE records were found.")

    df = compute_projection(df)
    df = compute_vorp(df)
    df = compute_cost(df)
    df = compute_risk(df)

    year_col = next(
        (column for column in ("year", "year_results", "year_adp") if column in df),
        None,
    )
    if year_col is None:
        raise KeyError("Scoring requires a year column.")

    # Each season back receives half the influence of the following season.
    newest_year = int(df[year_col].max())
    df["recency_weight"] = 0.5 ** (newest_year - df[year_col].astype(int))
    df["weighted_value_component"] = df["value_over_cost"] * df["recency_weight"]

    # Collapse to single score per player
    grouped = (
        df.groupby("player", as_index=False)
          .agg({
              "pos": "first",
              "team_adp": "last",      # most recent team from ADP data
              "adp_avg": "mean",       # average ADP across years
              "projection": "mean",    # average projection (or you could use last only)
              "vorp": "mean",          # average VORP
              "risk_penalty": "mean",  # average risk profile
              "round": "mean",         # average draft round
              "value_over_cost": "mean",
              "weighted_value_component": "sum",
              "recency_weight": "sum",
          })
    )

    grouped["weighted_value"] = (
        grouped.pop("weighted_value_component") / grouped.pop("recency_weight")
    )

    grouped["score"] = (
        grouped["weighted_value"]
        + grouped["vorp"]
        - grouped["risk_penalty"]
    )

    grouped = grouped.sort_values(by="score", ascending=False).reset_index(drop=True)
    return grouped
