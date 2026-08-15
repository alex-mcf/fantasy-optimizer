import pandas as pd

from fantasyoptimizer.config.league_config import (
    DEFAULT_LEAGUE_CONFIG,
    LeagueConfig,
)
from fantasyoptimizer.scoring.cost import compute_cost
from fantasyoptimizer.scoring.projection import compute_projection
from fantasyoptimizer.scoring.risk import compute_risk
from fantasyoptimizer.scoring.vorp import compute_vorp
from fantasyoptimizer.utils.data_loader import build_history

SUPPORTED_POSITIONS = ("QB", "RB", "WR", "TE")


def _confidence_label(seasons: int) -> str:
    if seasons >= 3:
        return "High"
    if seasons == 2:
        return "Medium"
    return "Low"


def compute_player_season_scores(
    years: list[int], config: LeagueConfig = DEFAULT_LEAGUE_CONFIG
) -> pd.DataFrame:
    """Return one retrospective score row for every matched player-season."""
    df = build_history(years)
    df = df[df["pos"].isin(SUPPORTED_POSITIONS)].copy()
    if df.empty:
        raise ValueError("No supported QB, RB, WR, or TE records were found.")

    df = compute_projection(df)
    df = compute_vorp(df, config.replacement_ranks())
    df = compute_cost(df, league_size=config.league_size)
    df = compute_risk(df)

    df["seasons_played"] = df.groupby("player_key")["year"].transform("nunique")
    df["confidence"] = df["seasons_played"].map(_confidence_label)
    df["season_score"] = (
        df["value_over_cost"] + df["vorp"] - df["risk_penalty"]
    )
    return df.sort_values(
        ["year", "season_score"], ascending=[False, False], ignore_index=True
    )


def compute_scores(
    years: list[int], config: LeagueConfig = DEFAULT_LEAGUE_CONFIG
) -> pd.DataFrame:
    """Collapse player-seasons into a recency-weighted historical summary."""
    df = compute_player_season_scores(years, config)
    newest_year = int(df["year"].max())
    df = df.sort_values(["year", "rank_adp"], ignore_index=True)
    df["recency_weight"] = 0.5 ** (newest_year - df["year"].astype(int))

    weighted_columns = [
        "projection",
        "value_over_cost",
        "vorp",
        "risk_penalty",
        "season_score",
    ]
    for column in weighted_columns:
        df[f"weighted_{column}"] = df[column] * df["recency_weight"]

    grouped = df.groupby("player_key", as_index=False).agg(
        player=("player", "last"),
        pos=("pos", "last"),
        team_adp=("team_adp", "last"),
        adp_avg=("adp_avg", "last"),
        round=("round", "last"),
        latest_year=("year", "max"),
        seasons_played=("year", "nunique"),
        risk_known=("risk", lambda values: values.notna().any()),
        recency_weight=("recency_weight", "sum"),
        **{
            f"weighted_{column}": (f"weighted_{column}", "sum")
            for column in weighted_columns
        },
    )

    for column in weighted_columns:
        grouped[column] = grouped.pop(f"weighted_{column}") / grouped["recency_weight"]
    grouped = grouped.drop(columns="recency_weight")
    grouped["weighted_value"] = grouped["value_over_cost"]
    grouped["score"] = grouped.pop("season_score")
    grouped["confidence"] = grouped["seasons_played"].map(_confidence_label)

    return grouped.sort_values("score", ascending=False, ignore_index=True)
