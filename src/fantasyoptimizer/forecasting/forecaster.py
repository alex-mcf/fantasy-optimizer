from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fantasyoptimizer.config.league_config import DEFAULT_LEAGUE_CONFIG, LeagueConfig
from fantasyoptimizer.scoring.scoring_engine import SUPPORTED_POSITIONS
from fantasyoptimizer.utils.data_loader import (
    DEFAULT_DATA_DIR,
    available_adp_years,
    available_context_years,
    available_player_context_years,
    available_result_years,
    load_adp,
    load_results,
    load_player_team_context,
    load_team_position_context,
)

PLAYER_FEATURE_COLUMNS = [
    "lag1_ppg",
    "lag2_ppg",
    "lag3_ppg",
    "weighted_ppg",
    "lag1_games",
    "weighted_games",
    "lag1_total",
    "trend",
    "history_seasons",
    "lag1_opp",
    "lag2_opp",
    "lag3_opp",
    "weighted_opp",
    "lag1_ppo",
    "weighted_ppo",
    "pos_QB",
    "pos_RB",
    "pos_WR",
    "pos_TE",
]

CONTEXT_FEATURE_COLUMNS = [
    "team_pos_lag1_opportunities",
    "team_pos_lag2_opportunities",
    "team_pos_lag3_opportunities",
    "team_pos_weighted_opportunities",
    "team_pos_lag1_opportunity_percentile",
    "team_pos_lag1_contributors",
    "team_pos_other_lag1_points",
    "team_pos_other_lag2_points",
    "team_pos_other_lag3_points",
    "team_pos_other_weighted_points",
    "team_pos_other_lag1_opportunities",
    "team_pos_other_lag2_opportunities",
    "team_pos_other_lag3_opportunities",
    "team_pos_other_weighted_opportunities",
    "lag1_opportunity_share",
    "weighted_opportunity_share",
    "same_team_last_year",
    "changed_team",
]

FEATURE_COLUMNS = PLAYER_FEATURE_COLUMNS + CONTEXT_FEATURE_COLUMNS


@dataclass
class RidgeModel:
    """Small deterministic ridge regressor with no serialized model state."""

    alpha: float = 8.0
    feature_columns: tuple[str, ...] = tuple(FEATURE_COLUMNS)
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    coefficients_: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "RidgeModel":
        values = frame[list(self.feature_columns)].to_numpy(dtype=float)
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        standardized = (values - self.mean_) / self.scale_
        design = np.column_stack([np.ones(len(standardized)), standardized])
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0
        self.coefficients_ = np.linalg.solve(
            design.T @ design + penalty, design.T @ target.to_numpy(dtype=float)
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.coefficients_ is None:
            raise ValueError("The forecast model must be fitted before prediction.")
        values = frame[list(self.feature_columns)].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(values)), (values - self.mean_) / self.scale_])
        return design @ self.coefficients_


def _result_history(years: list[int], data_dir: Path | str) -> pd.DataFrame:
    frames = [load_results(year, data_dir) for year in years]
    if not frames:
        return pd.DataFrame()
    history = pd.concat(frames, ignore_index=True)
    return history[history["pos"].isin(SUPPORTED_POSITIONS)].copy()


def _team_context_history(years: list[int], data_dir: Path | str) -> pd.DataFrame:
    context_years = set(available_context_years(data_dir))
    frames = [
        load_team_position_context(year, data_dir)
        for year in years
        if year in context_years
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _player_team_context_history(
    years: list[int], data_dir: Path | str
) -> pd.DataFrame:
    context_years = set(available_player_context_years(data_dir))
    frames = [
        load_player_team_context(year, data_dir)
        for year in years
        if year in context_years
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _derive_team_context(history: pd.DataFrame) -> pd.DataFrame:
    """Fallback for tests or older archives without weekly-derived context."""
    required = {"year", "team", "pos", "pts_ttl", "opp", "player_key"}
    if history.empty or not required.issubset(history.columns):
        return pd.DataFrame()
    player_team = history.groupby(
        ["year", "team", "pos", "player_key"], as_index=False
    ).agg(
        player_points=("pts_ttl", "sum"),
        player_opportunities=("opp", "sum"),
    )
    context = player_team.groupby(["year", "team", "pos"], as_index=False).agg(
        total_points=("player_points", "sum"),
        leader_points=("player_points", "max"),
        top_two_points=("player_points", lambda values: values.nlargest(2).sum()),
        total_opportunities=("player_opportunities", "sum"),
        leader_opportunities=("player_opportunities", "max"),
        top_two_opportunities=(
            "player_opportunities", lambda values: values.nlargest(2).sum()
        ),
        contributors=("player_key", "nunique"),
    )
    context["position_rank"] = context.groupby(["year", "pos"])[
        "total_points"
    ].rank(method="min", ascending=False)
    context["opportunity_rank"] = context.groupby(["year", "pos"])[
        "total_opportunities"
    ].rank(method="min", ascending=False)
    team_counts = context.groupby(["year", "pos"])["team"].transform("nunique")
    context["rank_percentile"] = 1 - (
        (context["position_rank"] - 1) / (team_counts - 1).clip(lower=1)
    )
    context["opportunity_rank_percentile"] = 1 - (
        (context["opportunity_rank"] - 1) / (team_counts - 1).clip(lower=1)
    )
    return context


def _derive_player_team_context(history: pd.DataFrame) -> pd.DataFrame:
    required = {"year", "team", "pos", "player_key", "pts_ttl", "opp"}
    if history.empty or not required.issubset(history.columns):
        return pd.DataFrame()
    return history.groupby(
        ["year", "team", "pos", "player_key"], as_index=False
    ).agg(
        player_points=("pts_ttl", "sum"),
        player_opportunities=("opp", "sum"),
    )


def _text_or_empty(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def build_features(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    target_year: int,
    team_context: pd.DataFrame | None = None,
    player_team_context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create features using only seasons strictly before ``target_year``."""
    if not history.empty and int(history["year"].max()) >= target_year:
        history = history[history["year"] < target_year].copy()
    indexed = {
        key: group.set_index("year").sort_index()
        for key, group in history.groupby(["player_key", "pos"])
    }
    if team_context is None or team_context.empty:
        team_context = _derive_team_context(history)
    elif int(team_context["year"].max()) >= target_year:
        team_context = team_context[team_context["year"] < target_year].copy()
    if player_team_context is None or player_team_context.empty:
        player_team_context = _derive_player_team_context(history)
    elif int(player_team_context["year"].max()) >= target_year:
        player_team_context = player_team_context[
            player_team_context["year"] < target_year
        ].copy()
    context_indexed = (
        {
            key: group.groupby("year", as_index=True).agg(
                total_points=("total_points", "sum"),
                leader_points=("leader_points", "max"),
                top_two_points=("top_two_points", "sum"),
                total_opportunities=("total_opportunities", "sum"),
                leader_opportunities=("leader_opportunities", "max"),
                top_two_opportunities=("top_two_opportunities", "sum"),
                contributors=("contributors", "sum"),
                rank_percentile=("rank_percentile", "max"),
                opportunity_rank_percentile=(
                    "opportunity_rank_percentile",
                    "max",
                ),
            )
            for key, group in team_context.groupby(["team", "pos"])
        }
        if not team_context.empty
        else {}
    )
    contribution_indexed = (
        {
            key: group.groupby("year", as_index=True).agg(
                player_points=("player_points", "sum"),
                player_opportunities=("player_opportunities", "sum"),
            )
            for key, group in player_team_context.groupby(
                ["player_key", "team", "pos"]
            )
        }
        if not player_team_context.empty
        else {}
    )
    rows: list[dict[str, object]] = []
    for candidate in candidates.to_dict("records"):
        prior = indexed.get((candidate["player_key"], candidate["pos"]))
        candidate_team = _text_or_empty(candidate.get("team", ""))
        destination_context = context_indexed.get((candidate_team, candidate["pos"]))
        lags: list[pd.Series | None] = []
        context_lags: list[pd.Series | None] = []
        for lag in (1, 2, 3):
            year = target_year - lag
            lags.append(prior.loc[year] if prior is not None and year in prior.index else None)
            context_lags.append(
                destination_context.loc[year]
                if destination_context is not None and year in destination_context.index
                else None
            )
        weights = [1.0, 0.5, 0.25]
        available = [(row, weight) for row, weight in zip(lags, weights) if row is not None]
        weighted_ppg = (
            sum(float(row["pts_avg"]) * weight for row, weight in available)
            / sum(weight for _, weight in available)
            if available
            else 0.0
        )
        weighted_games = (
            sum(float(row["gp"]) * weight for row, weight in available)
            / sum(weight for _, weight in available)
            if available
            else 0.0
        )
        lag_ppg = [float(row["pts_avg"]) if row is not None else 0.0 for row in lags]
        lag_opportunities = [
            float(row.get("opp", 0)) if row is not None else 0.0 for row in lags
        ]
        lag_efficiency = [
            float(row.get("ppo", 0)) if row is not None else 0.0 for row in lags
        ]
        lag1_games = float(lags[0]["gp"]) if lags[0] is not None else 0.0
        lag1_total = float(lags[0]["pts_ttl"]) if lags[0] is not None else 0.0
        weighted_opportunities = (
            sum(float(row.get("opp", 0)) * weight for row, weight in available)
            / sum(weight for _, weight in available)
            if available
            else 0.0
        )
        weighted_efficiency = (
            sum(float(row.get("ppo", 0)) * weight for row, weight in available)
            / sum(weight for _, weight in available)
            if available
            else 0.0
        )
        context_totals = [
            float(row["total_points"]) if row is not None else 0.0
            for row in context_lags
        ]
        context_top_two = [
            float(row["top_two_points"]) if row is not None else 0.0
            for row in context_lags
        ]
        context_opportunities = [
            float(row["total_opportunities"]) if row is not None else 0.0
            for row in context_lags
        ]
        context_available = [
            (row, weight)
            for row, weight in zip(context_lags, weights)
            if row is not None
        ]
        context_weight_sum = sum(weight for _, weight in context_available)
        weighted_context_total = (
            sum(float(row["total_points"]) * weight for row, weight in context_available)
            / context_weight_sum
            if context_available
            else 0.0
        )
        weighted_context_top_two = (
            sum(float(row["top_two_points"]) * weight for row, weight in context_available)
            / context_weight_sum
            if context_available
            else 0.0
        )
        weighted_context_opportunities = (
            sum(
                float(row["total_opportunities"]) * weight
                for row, weight in context_available
            )
            / context_weight_sum
            if context_available
            else 0.0
        )
        destination_contributions: list[pd.Series | None] = []
        for lag in (1, 2, 3):
            year = target_year - lag
            contribution = contribution_indexed.get(
                (candidate["player_key"], candidate_team, candidate["pos"])
            )
            destination_contributions.append(
                contribution.loc[year]
                if contribution is not None and year in contribution.index
                else None
            )
        other_points = [
            max(
                0.0,
                context_totals[index]
                - (
                    float(contribution["player_points"])
                    if contribution is not None
                    else 0.0
                ),
            )
            for index, contribution in enumerate(destination_contributions)
        ]
        other_opportunities = [
            max(
                0.0,
                context_opportunities[index]
                - (
                    float(contribution["player_opportunities"])
                    if contribution is not None
                    else 0.0
                ),
            )
            for index, contribution in enumerate(destination_contributions)
        ]
        weighted_other_points = (
            sum(
                other_points[index] * weight
                for index, (_, weight) in enumerate(zip(context_lags, weights))
                if context_lags[index] is not None
            )
            / context_weight_sum
            if context_available
            else 0.0
        )
        weighted_other_opportunities = (
            sum(
                other_opportunities[index] * weight
                for index, (_, weight) in enumerate(zip(context_lags, weights))
                if context_lags[index] is not None
            )
            / context_weight_sum
            if context_available
            else 0.0
        )
        previous_team = (
            _text_or_empty(lags[0].get("team", "")) if lags[0] is not None else ""
        )
        same_team = int(bool(previous_team) and previous_team == candidate_team)
        changed_team = int(bool(previous_team) and bool(candidate_team) and not same_team)
        role_shares: list[float] = []
        for index, lag_row in enumerate(lags):
            if lag_row is None:
                role_shares.append(0.0)
                continue
            year = target_year - index - 1
            old_team = _text_or_empty(lag_row.get("team", ""))
            old_context = context_indexed.get((old_team, candidate["pos"]))
            old_context_row = (
                old_context.loc[year]
                if old_context is not None and year in old_context.index
                else None
            )
            old_room_opportunities = (
                float(old_context_row["total_opportunities"])
                if old_context_row is not None
                else 0.0
            )
            old_contribution = contribution_indexed.get(
                (candidate["player_key"], old_team, candidate["pos"])
            )
            old_contribution_row = (
                old_contribution.loc[year]
                if old_contribution is not None and year in old_contribution.index
                else None
            )
            player_team_opportunities = (
                float(old_contribution_row["player_opportunities"])
                if old_contribution_row is not None
                else lag_opportunities[index]
            )
            role_shares.append(
                player_team_opportunities / old_room_opportunities
                if old_room_opportunities > 0
                else 0.0
            )
        weighted_role_share = (
            sum(
                role_shares[index] * weight
                for index, (row, weight) in enumerate(zip(lags, weights))
                if row is not None
            )
            / sum(weight for _, weight in available)
            if available
            else 0.0
        )
        record = dict(candidate)
        record.update(
            {
                "lag1_ppg": lag_ppg[0],
                "lag2_ppg": lag_ppg[1],
                "lag3_ppg": lag_ppg[2],
                "weighted_ppg": weighted_ppg,
                "lag1_games": lag1_games,
                "weighted_games": weighted_games,
                "lag1_total": lag1_total,
                "trend": lag_ppg[0] - lag_ppg[1] if lags[1] is not None else 0.0,
                "history_seasons": len(available),
                "lag1_opp": lag_opportunities[0],
                "lag2_opp": lag_opportunities[1],
                "lag3_opp": lag_opportunities[2],
                "weighted_opp": weighted_opportunities,
                "lag1_ppo": lag_efficiency[0],
                "weighted_ppo": weighted_efficiency,
                "team_pos_lag1_total": context_totals[0],
                "team_pos_lag2_total": context_totals[1],
                "team_pos_lag3_total": context_totals[2],
                "team_pos_weighted_total": weighted_context_total,
                "team_pos_lag1_top_two": context_top_two[0],
                "team_pos_weighted_top_two": weighted_context_top_two,
                "team_pos_lag1_rank_percentile": (
                    float(context_lags[0]["rank_percentile"])
                    if context_lags[0] is not None
                    else 0.0
                ),
                "team_pos_lag1_contributors": (
                    float(context_lags[0]["contributors"])
                    if context_lags[0] is not None
                    else 0.0
                ),
                "team_pos_lag1_opportunities": context_opportunities[0],
                "team_pos_lag2_opportunities": context_opportunities[1],
                "team_pos_lag3_opportunities": context_opportunities[2],
                "team_pos_weighted_opportunities": weighted_context_opportunities,
                "team_pos_lag1_opportunity_percentile": (
                    float(context_lags[0]["opportunity_rank_percentile"])
                    if context_lags[0] is not None
                    else 0.0
                ),
                "team_pos_other_lag1_points": other_points[0],
                "team_pos_other_lag2_points": other_points[1],
                "team_pos_other_lag3_points": other_points[2],
                "team_pos_other_weighted_points": weighted_other_points,
                "team_pos_other_lag1_opportunities": other_opportunities[0],
                "team_pos_other_lag2_opportunities": other_opportunities[1],
                "team_pos_other_lag3_opportunities": other_opportunities[2],
                "team_pos_other_weighted_opportunities": (
                    weighted_other_opportunities
                ),
                "destination_other_points": other_points[0],
                "lag1_role_share": role_shares[0],
                "lag1_opportunity_share": role_shares[0],
                "weighted_opportunity_share": weighted_role_share,
                "same_team_last_year": same_team,
                "changed_team": changed_team,
            }
        )
        for position in SUPPORTED_POSITIONS:
            record[f"pos_{position}"] = int(candidate["pos"] == position)
        rows.append(record)
    return pd.DataFrame(rows)


def build_training_examples(
    through_year: int, data_dir: Path | str = DEFAULT_DATA_DIR
) -> pd.DataFrame:
    result_years = [year for year in available_result_years(data_dir) if year <= through_year]
    adp_years = set(available_adp_years(data_dir))
    history = _result_history(result_years, data_dir)
    team_context = _team_context_history(result_years, data_dir)
    player_team_context = _player_team_context_history(result_years, data_dir)
    examples: list[pd.DataFrame] = []
    for target_year in result_years:
        if target_year not in adp_years or target_year == min(result_years):
            continue
        candidates = load_adp(target_year, data_dir)
        candidates = candidates[candidates["pos"].isin(SUPPORTED_POSITIONS)].copy()
        features = build_features(
            candidates,
            history,
            target_year,
            team_context,
            player_team_context,
        )
        actual = load_results(target_year, data_dir)[
            ["player_key", "pos", "pts_ttl"]
        ].rename(columns={"pts_ttl": "actual_points"})
        example = features.merge(actual, on=["player_key", "pos"], how="inner")
        example["target_year"] = target_year
        examples.append(example)
    if not examples:
        raise ValueError("At least two complete ADP/results seasons are required.")
    return pd.concat(examples, ignore_index=True)


def _add_value_ranks(forecast: pd.DataFrame, config: LeagueConfig) -> pd.DataFrame:
    forecast = forecast.copy()
    replacements = config.replacement_ranks()
    forecast["replacement_points"] = 0.0
    forecast["position_rank"] = 0
    for position, indices in forecast.groupby("pos").groups.items():
        ordered = forecast.loc[indices].sort_values("forecast_points", ascending=False)
        replacement_index = min(replacements[position], len(ordered)) - 1
        replacement_points = float(ordered.iloc[replacement_index]["forecast_points"])
        forecast.loc[indices, "replacement_points"] = replacement_points
        forecast.loc[ordered.index, "position_rank"] = range(1, len(ordered) + 1)
    forecast["model_value"] = forecast["forecast_points"] - forecast["replacement_points"]
    forecast["team_context_rank"] = forecast.groupby("pos")[
        "team_pos_weighted_opportunities"
    ].rank(method="dense", ascending=False).astype(int)
    forecast = forecast.sort_values(
        ["model_value", "forecast_points"], ascending=False, ignore_index=True
    )
    forecast["model_rank"] = range(1, len(forecast) + 1)
    forecast["value_gap"] = forecast["adp_avg"] - forecast["model_rank"]
    return forecast


def _rank_by_replacement_value(
    frame: pd.DataFrame,
    points_column: str,
    config: LeagueConfig,
    value_column: str,
    rank_column: str,
) -> pd.DataFrame:
    """Rank a common player pool using league-adjusted value over replacement."""
    ranked = frame.copy()
    replacements = config.replacement_ranks()
    replacement_points = pd.Series(0.0, index=ranked.index)
    for position, indices in ranked.groupby("pos").groups.items():
        ordered = ranked.loc[indices].sort_values(points_column, ascending=False)
        replacement_index = min(replacements[position], len(ordered)) - 1
        replacement_points.loc[indices] = float(
            ordered.iloc[replacement_index][points_column]
        )
    ranked[value_column] = ranked[points_column] - replacement_points
    ranked = ranked.sort_values(
        [value_column, points_column], ascending=False, ignore_index=True
    )
    ranked[rank_column] = range(1, len(ranked) + 1)
    return ranked


def forecast_season(
    target_year: int,
    config: LeagueConfig = DEFAULT_LEAGUE_CONFIG,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    training_examples: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Forecast a season independently of its ADP, then compare rank with ADP."""
    prior_years = [year for year in available_result_years(data_dir) if year < target_year]
    if target_year not in available_adp_years(data_dir):
        raise FileNotFoundError(f"No {target_year} preseason ADP file is available.")
    training = (
        training_examples
        if training_examples is not None
        else build_training_examples(max(prior_years), data_dir)
    )
    model = RidgeModel().fit(training, training["actual_points"])
    player_only_model = RidgeModel(
        feature_columns=tuple(PLAYER_FEATURE_COLUMNS)
    ).fit(training, training["actual_points"])
    candidates = load_adp(target_year, data_dir)
    candidates = candidates[candidates["pos"].isin(SUPPORTED_POSITIONS)].copy()
    history = _result_history(prior_years, data_dir)
    team_context = _team_context_history(prior_years, data_dir)
    player_team_context = _player_team_context_history(prior_years, data_dir)
    features = build_features(
        candidates,
        history,
        target_year,
        team_context,
        player_team_context,
    )
    features["forecast_points"] = np.clip(model.predict(features), 0, 500).round(1)
    features["player_only_points"] = np.clip(
        player_only_model.predict(features), 0, 500
    ).round(1)
    features["context_adjustment"] = (
        features["forecast_points"] - features["player_only_points"]
    ).round(1)
    features["confidence"] = features["history_seasons"].map(
        {0: "Rookie / no NFL history", 1: "Low", 2: "Medium", 3: "High"}
    )
    return _add_value_ranks(features, config)


def build_team_position_outlook(forecast: pd.DataFrame) -> pd.DataFrame:
    """Summarize the model's destination role signal for each current team/position."""
    rows: list[dict[str, object]] = []
    for (team, position), group in forecast.groupby(["team", "pos"], dropna=False):
        ordered = group.sort_values("model_rank")
        favorite = ordered.iloc[0]
        rows.append(
            {
                "team": team,
                "pos": position,
                "context_rank": int(favorite["team_context_rank"]),
                "weighted_room_points": float(favorite["team_pos_weighted_total"]),
                "weighted_room_opportunities": float(
                    favorite["team_pos_weighted_opportunities"]
                ),
                "last_year_room_points": float(favorite["team_pos_lag1_total"]),
                "last_year_room_opportunities": float(
                    favorite["team_pos_lag1_opportunities"]
                ),
                "last_year_other_points": float(
                    favorite["team_pos_other_lag1_points"]
                ),
                "last_year_other_opportunities": float(
                    favorite["team_pos_other_lag1_opportunities"]
                ),
                "last_year_top_two": float(favorite["team_pos_lag1_top_two"]),
                "last_year_rank_percentile": float(
                    favorite["team_pos_lag1_rank_percentile"]
                ),
                "model_favorite": favorite["player"],
                "favorite_model_rank": int(favorite["model_rank"]),
                "favorite_forecast_points": float(favorite["forecast_points"]),
                "favorite_context_adjustment": float(
                    favorite["context_adjustment"]
                ),
                "favorite_role_share": float(favorite["lag1_opportunity_share"]),
                "candidates": ", ".join(ordered["player"].head(3)),
                "candidate_count": len(ordered),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["pos", "context_rank", "favorite_model_rank"], ignore_index=True
    )


def backtest_forecaster(
    through_year: int,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    training_examples: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run expanding-window backtests; each model sees only earlier targets."""
    examples = (
        training_examples
        if training_examples is not None
        else build_training_examples(through_year, data_dir)
    )
    rows: list[dict[str, object]] = []
    target_years = sorted(examples["target_year"].unique())
    for target_year in target_years:
        train = examples[examples["target_year"] < target_year]
        test = examples[examples["target_year"] == target_year].copy()
        if train["target_year"].nunique() < 2 or test.empty:
            continue
        model = RidgeModel().fit(train, train["actual_points"])
        player_only_model = RidgeModel(
            feature_columns=tuple(PLAYER_FEATURE_COLUMNS)
        ).fit(train, train["actual_points"])
        test["predicted"] = np.clip(model.predict(test), 0, 500)
        test["player_only_predicted"] = np.clip(
            player_only_model.predict(test), 0, 500
        )
        mae = float((test["actual_points"] - test["predicted"]).abs().mean())
        player_only_mae = float(
            (test["actual_points"] - test["player_only_predicted"]).abs().mean()
        )
        rows.append(
            {
                "year": int(target_year),
                "players": len(test),
                "mae": mae,
                "player_only_mae": player_only_mae,
                "context_mae_lift": player_only_mae - mae,
                "rank_correlation": float(
                    test[["actual_points", "predicted"]].corr(method="spearman").iloc[0, 1]
                ),
            }
        )
    return pd.DataFrame(rows)


def backtest_draft_value(
    through_year: int,
    config: LeagueConfig = DEFAULT_LEAGUE_CONFIG,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    training_examples: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test whether one-round model bargains subsequently beat their market rank.

    Both model and realized ranks use value over a league-specific replacement
    player. Market rank is normalized within the same modeled QB/RB/WR/TE pool,
    which keeps the comparison fair when an ADP source also contains kickers,
    defenses, or unmatched rows.
    """
    examples = (
        training_examples
        if training_examples is not None
        else build_training_examples(through_year, data_dir)
    )
    evaluated: list[pd.DataFrame] = []
    target_years = sorted(examples["target_year"].unique())
    for target_year in target_years:
        train = examples[examples["target_year"] < target_year]
        test = examples[examples["target_year"] == target_year].copy()
        if train["target_year"].nunique() < 2 or test.empty:
            continue

        model = RidgeModel().fit(train, train["actual_points"])
        test["predicted_points"] = np.clip(model.predict(test), 0, 500)
        test["market_rank"] = (
            test["adp_avg"].rank(method="first", ascending=True).astype(int)
        )

        predicted = _rank_by_replacement_value(
            test,
            "predicted_points",
            config,
            "predicted_value",
            "predicted_rank",
        )
        realized = _rank_by_replacement_value(
            test,
            "actual_points",
            config,
            "actual_value",
            "actual_rank",
        )
        identity = ["player_key", "pos"]
        evaluation = predicted[
            identity
            + [
                "player",
                "team",
                "target_year",
                "adp_avg",
                "market_rank",
                "predicted_points",
                "predicted_value",
                "predicted_rank",
            ]
        ].merge(
            realized[identity + ["actual_points", "actual_value", "actual_rank"]],
            on=identity,
            how="inner",
            validate="one_to_one",
        )
        evaluation["predicted_rank_surplus"] = (
            evaluation["market_rank"] - evaluation["predicted_rank"]
        )
        evaluation["actual_rank_surplus"] = (
            evaluation["market_rank"] - evaluation["actual_rank"]
        )
        round_size = config.league_size
        evaluation["market_round"] = np.ceil(
            evaluation["market_rank"] / round_size
        ).astype(int)
        evaluation["bargain_flag"] = (
            evaluation["predicted_rank_surplus"] >= round_size
        )
        evaluation["fade_flag"] = (
            evaluation["predicted_rank_surplus"] <= -round_size
        )
        evaluation["beat_market"] = evaluation["actual_rank_surplus"] > 0
        evaluation["finished_below_market"] = evaluation["actual_rank_surplus"] < 0
        evaluated.append(evaluation)

    if not evaluated:
        return pd.DataFrame(), pd.DataFrame()

    player_results = pd.concat(evaluated, ignore_index=True).sort_values(
        ["target_year", "predicted_rank"], ignore_index=True
    )

    def summarize(group: pd.DataFrame, year: int | str) -> dict[str, object]:
        bargains = group[group["bargain_flag"]]
        non_bargains = group[~group["bargain_flag"]]
        fades = group[group["fade_flag"]]
        control_rates = non_bargains.groupby(["target_year", "market_round"])[
            "beat_market"
        ].mean()
        bargain_control_rates = pd.MultiIndex.from_frame(
            bargains[["target_year", "market_round"]]
        ).map(control_rates)
        comparable = ~pd.isna(bargain_control_rates)
        comparable_bargains = bargains.loc[comparable]
        model_closer = (
            (bargains["actual_rank"] - bargains["predicted_rank"]).abs()
            < (bargains["actual_rank"] - bargains["market_rank"]).abs()
        )
        return {
            "year": year,
            "players": len(group),
            "bargains": len(bargains),
            "bargain_hit_rate": float(bargains["beat_market"].mean()),
            "non_bargain_hit_rate": float(non_bargains["beat_market"].mean()),
            "bargain_hit_lift": float(
                bargains["beat_market"].mean() - non_bargains["beat_market"].mean()
            ),
            "round_adjusted_expected_hit_rate": float(
                pd.Series(bargain_control_rates).mean()
            ),
            "round_adjusted_hit_lift": float(
                comparable_bargains["beat_market"].mean()
                - pd.Series(bargain_control_rates[comparable]).mean()
            ),
            "bargain_override_win_rate": float(model_closer.mean()),
            "bargain_avg_actual_surplus": float(
                bargains["actual_rank_surplus"].mean()
            ),
            "non_bargain_avg_actual_surplus": float(
                non_bargains["actual_rank_surplus"].mean()
            ),
            "fades": len(fades),
            "fade_hit_rate": float(fades["finished_below_market"].mean()),
            "market_rank_mae": float(
                (group["actual_rank"] - group["market_rank"]).abs().mean()
            ),
            "model_rank_mae": float(
                (group["actual_rank"] - group["predicted_rank"]).abs().mean()
            ),
            "rank_mae_improvement": float(
                (group["actual_rank"] - group["market_rank"]).abs().mean()
                - (group["actual_rank"] - group["predicted_rank"]).abs().mean()
            ),
            "market_correlation": float(
                group[["market_rank", "actual_rank"]]
                .corr(method="spearman")
                .iloc[0, 1]
            ),
            "signal_correlation": float(
                group[["predicted_rank_surplus", "actual_rank_surplus"]]
                .corr(method="spearman")
                .iloc[0, 1]
            ),
        }

    summary_rows = [
        summarize(group, int(year))
        for year, group in player_results.groupby("target_year", sort=True)
    ]
    summary_rows.append(summarize(player_results, "Overall"))
    return pd.DataFrame(summary_rows), player_results
