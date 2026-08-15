from __future__ import annotations

from dataclasses import dataclass, field
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
    load_player_metadata,
    load_preseason_roles,
    load_results,
    load_player_team_context,
    load_team_position_context,
)

PLAYER_FEATURE_COLUMNS = [
    "age",
    "age_squared",
    "age_known",
    "experience",
    "experience_known",
    "rookie",
    "sophomore",
    "drafted",
    "draft_round",
    "draft_pick",
    "height",
    "weight",
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

MARKET_FEATURE_COLUMNS = [
    "adp_avg",
    "market_log_adp",
    "market_sqrt_adp",
    "market_position_rank",
    "market_room_known",
    "market_room_rank",
    "market_room_adp_gap",
    "market_log_samples",
    "market_stddev",
    "market_range",
]


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


def add_market_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create point-in-time market features without using season results."""
    featured = frame.copy()
    featured["market_log_adp"] = np.log1p(featured["adp_avg"])
    featured["market_sqrt_adp"] = np.sqrt(featured["adp_avg"])
    groups = ["pos"]
    if "target_year" in featured:
        groups.insert(0, "target_year")
    featured["market_position_rank"] = featured.groupby(groups)["adp_avg"].rank(
        method="first", ascending=True
    )
    team = featured.get(
        "team", pd.Series("", index=featured.index, dtype="object")
    ).fillna("").astype(str).str.strip().str.upper()
    featured["market_room_known"] = (
        ~team.isin({"", "FA", "NAN", "NONE"})
    ).astype(int)
    featured["market_room_rank"] = 1.0
    featured["market_room_size"] = 1.0
    featured["market_room_leader"] = 0.0
    featured["market_room_adp_gap"] = 0.0
    known = featured["market_room_known"].eq(1)
    if known.any():
        room_groups = [*groups, "team"]
        room = featured.loc[known].copy()
        room["market_room_rank"] = room.groupby(room_groups)["adp_avg"].rank(
            method="first", ascending=True
        )
        room["market_room_size"] = room.groupby(room_groups)["adp_avg"].transform(
            "size"
        )
        room["market_room_leader"] = room["market_room_rank"].eq(1).astype(int)
        room["market_room_adp_gap"] = room["adp_avg"] - room.groupby(room_groups)[
            "adp_avg"
        ].transform("min")

        room_columns = [
            "market_room_rank",
            "market_room_size",
            "market_room_leader",
            "market_room_adp_gap",
        ]
        featured.loc[known, room_columns] = room[room_columns]
    samples = (
        featured["timesdrafted"]
        if "timesdrafted" in featured
        else pd.Series(0.0, index=featured.index)
    )
    deviation = (
        featured["stddev"]
        if "stddev" in featured
        else pd.Series(0.0, index=featured.index)
    )
    high = (
        featured["high"]
        if "high" in featured
        else featured["adp_avg"]
    )
    low = (
        featured["low"]
        if "low" in featured
        else featured["adp_avg"]
    )
    featured["market_log_samples"] = np.log1p(samples.fillna(0).clip(lower=0))
    featured["market_stddev"] = deviation.fillna(0).clip(lower=0)
    featured["market_range"] = (low - high).fillna(0).clip(lower=0)
    return featured


def add_official_role_context(
    frame: pd.DataFrame, roles: pd.DataFrame | None
) -> pd.DataFrame:
    """Attach current official role evidence without using it as a model feature."""
    featured = frame.copy()
    defaults: dict[str, object] = {
        "official_role_known": 0,
        "official_depth_rank": np.nan,
        "official_starter": 0,
        "official_roster_status": "Unknown",
        "official_roster_status_description": "",
        "official_depth_position": "",
        "official_role_snapshot": "",
        "official_role_timing": "unavailable",
        "depth_market_gap": np.nan,
        "role_agreement": "Official role unavailable",
    }
    for column, value in defaults.items():
        featured[column] = value
    if roles is None or roles.empty:
        return featured

    role_records = roles.sort_values("depth_rank").to_dict("records")
    by_id = {
        (_text_or_empty(row.get("gsis_id")), row["team"], row["pos"]): row
        for row in role_records
        if _text_or_empty(row.get("gsis_id"))
    }
    by_name = {
        (row["player_key"], row["team"], row["pos"]): row
        for row in role_records
    }
    for index, candidate in featured.iterrows():
        identity = (
            _text_or_empty(candidate.get("gsis_id")),
            candidate.get("team", ""),
            candidate.get("pos", ""),
        )
        role = by_id.get(identity)
        if role is None:
            role = by_name.get(
                (
                    candidate.get("player_key", ""),
                    candidate.get("team", ""),
                    candidate.get("pos", ""),
                )
            )
        if role is None:
            continue
        depth_rank = float(role["depth_rank"])
        market_rank = float(candidate.get("market_room_rank", 1))
        gap = market_rank - depth_rank
        if abs(gap) < 0.5:
            agreement = "Aligned"
        elif gap > 0:
            agreement = "Depth chart ahead of market"
        else:
            agreement = "Market ahead of depth chart"
        values = {
            "official_role_known": 1,
            "official_depth_rank": depth_rank,
            "official_starter": int(role.get("official_starter", 0)),
            "official_roster_status": _text_or_empty(
                role.get("roster_status", "Unknown")
            ) or "Unknown",
            "official_roster_status_description": _text_or_empty(
                role.get("roster_status_description", "")
            ),
            "official_depth_position": _text_or_empty(
                role.get("depth_position", "")
            ),
            "official_role_snapshot": _text_or_empty(role.get("snapshot_at", "")),
            "official_role_timing": _text_or_empty(
                role.get("timing_quality", "unavailable")
            ),
            "depth_market_gap": gap,
            "role_agreement": agreement,
        }
        for column, value in values.items():
            featured.at[index, column] = value
    return featured


@dataclass
class MarketResidualModel:
    """Use ADP as a baseline, then learn position-specific market mistakes."""

    alpha_market: float = 100.0
    alpha_residual: float = 300.0
    residual_feature_columns: tuple[str, ...] = tuple(FEATURE_COLUMNS)
    market_models_: dict[str, RidgeModel] = field(default_factory=dict)
    residual_models_: dict[str, RidgeModel] = field(default_factory=dict)
    error_scale_: dict[str, float] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "MarketResidualModel":
        prepared = add_market_features(frame)
        aligned_target = pd.Series(target.to_numpy(dtype=float), index=prepared.index)
        self.market_models_.clear()
        self.residual_models_.clear()
        self.error_scale_.clear()
        for position, group in prepared.groupby("pos"):
            position_target = aligned_target.loc[group.index]
            market_model = RidgeModel(
                alpha=self.alpha_market,
                feature_columns=tuple(MARKET_FEATURE_COLUMNS),
            ).fit(group, position_target)
            market_prediction = market_model.predict(group)
            residual_model = RidgeModel(
                alpha=self.alpha_residual,
                feature_columns=self.residual_feature_columns,
            ).fit(group, position_target - market_prediction)
            prediction = market_prediction + residual_model.predict(group)
            self.market_models_[position] = market_model
            self.residual_models_[position] = residual_model
            scale = float(np.std(position_target.to_numpy() - prediction))
            self.error_scale_[position] = max(scale, 1.0)
        return self

    def predict_components(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = add_market_features(frame)
        output = pd.DataFrame(index=prepared.index)
        output["market_prediction"] = 0.0
        output["market_adjustment"] = 0.0
        output["prediction"] = 0.0
        output["error_scale"] = 1.0
        for position, indices in prepared.groupby("pos").groups.items():
            if position not in self.market_models_:
                raise ValueError(f"No fitted market-residual model for {position}.")
            group = prepared.loc[indices]
            market = self.market_models_[position].predict(group)
            adjustment = self.residual_models_[position].predict(group)
            output.loc[indices, "market_prediction"] = market
            output.loc[indices, "market_adjustment"] = adjustment
            output.loc[indices, "prediction"] = market + adjustment
            output.loc[indices, "error_scale"] = self.error_scale_[position]
        return output

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_components(frame)["prediction"].to_numpy()


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


def _number_or(value: object, default: float) -> float:
    return default if value is None or pd.isna(value) else float(value)


def _enrich_player_metadata(
    candidates: pd.DataFrame,
    player_metadata: pd.DataFrame | None,
    target_year: int,
) -> pd.DataFrame:
    if player_metadata is None or player_metadata.empty:
        return candidates.copy()
    metadata = player_metadata.copy()
    metadata = metadata[
        metadata["rookie_season"].isna()
        | (metadata["rookie_season"] <= target_year)
    ].copy()
    metadata["rookie_distance"] = target_year - metadata["rookie_season"].fillna(-999)
    metadata = metadata.sort_values(
        ["player_key", "pos", "rookie_distance", "gsis_id"],
        ascending=[True, True, True, True],
    ).drop_duplicates(["player_key", "pos"], keep="first")
    keep = [
        "player_key",
        "pos",
        "birth_date",
        "height",
        "weight",
        "rookie_season",
        "draft_round",
        "draft_pick",
        "college_name",
        "gsis_id",
    ]
    return candidates.merge(
        metadata[keep], on=["player_key", "pos"], how="left", validate="many_to_one"
    )


def build_features(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    target_year: int,
    team_context: pd.DataFrame | None = None,
    player_team_context: pd.DataFrame | None = None,
    player_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create features using only seasons strictly before ``target_year``."""
    candidates = _enrich_player_metadata(candidates, player_metadata, target_year)
    if not history.empty and int(history["year"].max()) >= target_year:
        history = history[history["year"] < target_year].copy()
    indexed = (
        {
            key: group.set_index("year").sort_index()
            for key, group in history.groupby(["player_key", "pos"])
        }
        if not history.empty
        else {}
    )
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
        birth_date = candidate.get("birth_date")
        age_known = int(birth_date is not None and not pd.isna(birth_date))
        age = (
            (pd.Timestamp(target_year, 9, 1) - pd.Timestamp(birth_date)).days / 365.25
            if age_known
            else 0.0
        )
        rookie_season = candidate.get("rookie_season")
        experience_known = int(
            rookie_season is not None and not pd.isna(rookie_season)
        )
        experience = (
            max(0.0, float(target_year - rookie_season))
            if experience_known
            else 0.0
        )
        draft_pick = candidate.get("draft_pick")
        drafted = int(draft_pick is not None and not pd.isna(draft_pick))
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
                "age": age,
                "age_squared": age * age,
                "age_known": age_known,
                "experience": experience,
                "experience_known": experience_known,
                "rookie": int(experience_known and experience == 0),
                "sophomore": int(experience_known and experience == 1),
                "drafted": drafted,
                "draft_round": _number_or(candidate.get("draft_round"), 8.0),
                "draft_pick": float(draft_pick) if drafted else 300.0,
                "height": _number_or(candidate.get("height"), 0.0),
                "weight": _number_or(candidate.get("weight"), 0.0),
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
    player_metadata = load_player_metadata(data_dir)
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
            player_metadata,
        )
        actual = load_results(target_year, data_dir)[
            ["player_key", "pos", "pts_ttl", "pts_avg", "gp"]
        ].rename(
            columns={
                "pts_ttl": "actual_points",
                "pts_avg": "actual_ppg",
                "gp": "actual_games",
            }
        )
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
    forecast["market_rank"] = (
        forecast["adp_avg"].rank(method="first", ascending=True).astype(int)
    )
    forecast["fair_adp"] = forecast["model_rank"]
    forecast["actionable_adp"] = (
        0.75 * forecast["market_rank"] + 0.25 * forecast["fair_adp"]
    ).round().astype(int)
    forecast["raw_adp_gap"] = forecast["adp_avg"] - forecast["model_rank"]
    forecast["value_gap"] = forecast["market_rank"] - forecast["model_rank"]
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
    """Forecast a season by adjusting the market for player/context evidence."""
    prior_years = [year for year in available_result_years(data_dir) if year < target_year]
    if target_year not in available_adp_years(data_dir):
        raise FileNotFoundError(f"No {target_year} preseason ADP file is available.")
    training = (
        training_examples
        if training_examples is not None
        else build_training_examples(max(prior_years), data_dir)
    )
    model = MarketResidualModel().fit(training, training["actual_points"])
    player_only_model = MarketResidualModel(
        residual_feature_columns=tuple(PLAYER_FEATURE_COLUMNS)
    ).fit(training, training["actual_points"])
    ppg_model = MarketResidualModel().fit(training, training["actual_ppg"])
    games_model = MarketResidualModel().fit(training, training["actual_games"])
    candidates = load_adp(target_year, data_dir)
    candidates = candidates[candidates["pos"].isin(SUPPORTED_POSITIONS)].copy()
    history = _result_history(prior_years, data_dir)
    team_context = _team_context_history(prior_years, data_dir)
    player_team_context = _player_team_context_history(prior_years, data_dir)
    player_metadata = load_player_metadata(data_dir)
    features = build_features(
        candidates,
        history,
        target_year,
        team_context,
        player_team_context,
        player_metadata,
    )
    # Retain point-in-time market hierarchy fields for explanations in the UI.
    features = add_market_features(features)
    features = add_official_role_context(
        features, load_preseason_roles(target_year, data_dir)
    )
    point_components = model.predict_components(features)
    features["market_points"] = np.clip(
        point_components["market_prediction"], 0, 500
    ).round(1)
    features["forecast_points"] = np.clip(
        point_components["prediction"], 0, 500
    ).round(1)
    features["market_adjustment"] = (
        features["forecast_points"] - features["market_points"]
    ).round(1)
    features["player_only_points"] = np.clip(
        player_only_model.predict(features), 0, 500
    ).round(1)
    features["context_adjustment"] = (
        features["forecast_points"] - features["player_only_points"]
    ).round(1)
    features["forecast_ppg"] = np.clip(ppg_model.predict(features), 0, 40).round(1)
    features["forecast_games"] = np.clip(
        games_model.predict(features), 0, 18
    ).round(1)
    features["availability_projection_points"] = (
        features["forecast_ppg"] * features["forecast_games"]
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
        model = MarketResidualModel().fit(train, train["actual_points"])
        player_only_model = MarketResidualModel(
            residual_feature_columns=tuple(PLAYER_FEATURE_COLUMNS)
        ).fit(train, train["actual_points"])
        market_components = model.predict_components(test)
        test["predicted"] = np.clip(model.predict(test), 0, 500)
        test["market_predicted"] = np.clip(
            market_components["market_prediction"], 0, 500
        )
        test["player_only_predicted"] = np.clip(
            player_only_model.predict(test), 0, 500
        )
        mae = float((test["actual_points"] - test["predicted"]).abs().mean())
        player_only_mae = float(
            (test["actual_points"] - test["player_only_predicted"]).abs().mean()
        )
        market_mae = float(
            (test["actual_points"] - test["market_predicted"]).abs().mean()
        )
        rows.append(
            {
                "year": int(target_year),
                "players": len(test),
                "mae": mae,
                "market_mae": market_mae,
                "market_mae_lift": market_mae - mae,
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

        model = MarketResidualModel().fit(train, train["actual_points"])
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


def _edge_bucket(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-np.inf, -12, 0, 12, 24, np.inf],
        labels=["Below market", "Slight fade", "Near market", "1-round edge", "2+ round edge"],
        right=False,
    )


def _market_tier(market_rank: pd.Series, league_size: int) -> pd.Series:
    rounds = np.ceil(market_rank / league_size)
    return pd.cut(
        rounds,
        bins=[0, 4, 9, np.inf],
        labels=["Rounds 1-4", "Rounds 5-9", "Rounds 10+"],
        right=True,
    )


def calibrate_edge_probabilities(
    forecast: pd.DataFrame,
    historical_players: pd.DataFrame,
    config: LeagueConfig = DEFAULT_LEAGUE_CONFIG,
    prior_strength: float = 16.0,
) -> pd.DataFrame:
    """Estimate beat-ADP probability from comparable out-of-sample signals."""
    calibrated = forecast.copy()
    calibrated["edge_bucket"] = _edge_bucket(calibrated["value_gap"])
    calibrated["market_tier"] = _market_tier(
        calibrated["market_rank"], config.league_size
    )
    if historical_players.empty:
        calibrated["edge_probability"] = 0.5
        calibrated["calibration_sample"] = 0
        return calibrated

    history = historical_players.copy()
    history["edge_bucket"] = _edge_bucket(history["predicted_rank_surplus"])
    history["market_tier"] = _market_tier(
        history["market_rank"], config.league_size
    )
    global_rate = float(history["beat_market"].mean())
    rates = history.groupby(
        ["pos", "edge_bucket", "market_tier"], observed=True
    )["beat_market"].agg(["sum", "count"])
    probabilities: list[float] = []
    samples: list[int] = []
    for row in calibrated.to_dict("records"):
        key = (row["pos"], row["edge_bucket"], row["market_tier"])
        if key in rates.index:
            successes = float(rates.loc[key, "sum"])
            count = int(rates.loc[key, "count"])
        else:
            successes = 0.0
            count = 0
        probabilities.append(
            (successes + prior_strength * global_rate) / (count + prior_strength)
        )
        samples.append(count)
    calibrated["edge_probability"] = probabilities
    calibrated["calibration_sample"] = samples
    calibrated["edge_confidence"] = np.select(
        [
            (calibrated["calibration_sample"] >= 30)
            & (calibrated["history_seasons"] >= 2),
            (calibrated["calibration_sample"] >= 12)
            & (calibrated["history_seasons"] >= 1),
        ],
        ["High", "Medium"],
        default="Low",
    )
    return calibrated


def segment_draft_value(
    historical_players: pd.DataFrame,
    config: LeagueConfig = DEFAULT_LEAGUE_CONFIG,
) -> pd.DataFrame:
    """Expose where the bargain signal works instead of hiding pooled weakness."""
    if historical_players.empty:
        return pd.DataFrame()
    frame = historical_players.copy()
    frame["market_tier"] = _market_tier(frame["market_rank"], config.league_size)
    bargains = frame[frame["bargain_flag"]].copy()
    bargains["model_closer"] = (
        (bargains["actual_rank"] - bargains["predicted_rank"]).abs()
        < (bargains["actual_rank"] - bargains["market_rank"]).abs()
    )
    return (
        bargains.groupby(["pos", "market_tier"], observed=True, as_index=False)
        .agg(
            bargains=("player", "size"),
            hit_rate=("beat_market", "mean"),
            override_win_rate=("model_closer", "mean"),
            average_actual_surplus=("actual_rank_surplus", "mean"),
        )
        .sort_values(["pos", "market_tier"], ignore_index=True)
    )
