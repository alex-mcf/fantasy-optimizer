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

    df["score"] = (
        df["value_over_cost"]
        #- df["vorp"] * 0.5 # optional to keep top players high
        - df["risk_penalty"] * 0.3
    )

    df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
    return df