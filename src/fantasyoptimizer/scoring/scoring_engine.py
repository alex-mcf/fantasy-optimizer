# scoring_engine.py
import pandas as pd
from fantasyoptimizer.utils.data_loader import build_history
from fantasyoptimizer.scoring.projection import compute_projection
from fantasyoptimizer.scoring.vorp import compute_vorp
from fantasyoptimizer.scoring.cost import compute_cost
from fantasyoptimizer.scoring.risk import compute_risk

def compute_scores(years: list[int]) -> pd.DataFrame:

    df = build_history(years)

    df = compute_projection(df)
    df = compute_vorp(df)
    df = compute_cost(df)
    df = compute_risk(df)
    
    # Weights for different years can be adjusted as needed
    weights = {
        2024: 0.6,
        2023: 0.3,
        2022: 0.1,
    }
    
    if "year_results" in df.columns:
        year_col = "year_results"
    elif "year_adp" in df.columns:
        year_col = "year_adp"
    else:
        year_col = "year"  
        
    df["weighted_value"] = df.apply(
        lambda row: row["value_over_cost"] * weights.get(int(row[year_col]), 0.0),
        axis=1,
    )
    
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
              "weighted_value": "sum", # multi-year undervalue signal
          })
    )

    grouped["score"] = (
        grouped["weighted_value"]
        + grouped["value_over_cost"]
        - grouped["vorp"] * 0.4 # optional to keep top players high
        - grouped["risk_penalty"] # penalize high risk
    )

    grouped = grouped.sort_values(by="score", ascending=False).reset_index(drop=True)
    return grouped