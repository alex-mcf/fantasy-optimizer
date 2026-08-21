#!/usr/bin/env python3
"""Create a FantasyOptimizer half-PPR results CSV from nflverse weekly stats."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import re
import sys
import unicodedata

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasyoptimizer.utils.atomic import write_csv, write_text  # noqa: E402

URL_TEMPLATE = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{year}.csv"
)
SNAP_URL_TEMPLATE = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "snap_counts/snap_counts_{year}.csv"
)
POSITIONS = ("QB", "RB", "WR", "TE")


def _download_csv(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    return pd.read_csv(BytesIO(response.content), low_memory=False)


def download_weekly_stats(year: int) -> pd.DataFrame:
    return _download_csv(URL_TEMPLATE.format(year=year))


def download_snap_counts(year: int) -> pd.DataFrame:
    return _download_csv(SNAP_URL_TEMPLATE.format(year=year))


def _player_key(name: str) -> str:
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\.?$", "", name.lower().strip())
    return re.sub(r"[^a-z0-9]", "", name)


def _scored_regular_season(stats: pd.DataFrame, year: int) -> pd.DataFrame:
    frame = stats.loc[
        (stats["season"] == year)
        & (stats["season_type"] == "REG")
        & stats["position"].isin(POSITIONS)
    ].copy()
    if frame.empty:
        raise ValueError(f"No regular-season player rows found for {year}.")
    # nflverse standard points use -2 per passing interception. FantasyPros'
    # historical leaders report uses -1, so one point per interception is added
    # back before applying the half-point reception bonus.
    frame["half_ppr"] = (
        frame["fantasy_points"].fillna(0)
        + frame["passing_interceptions"].fillna(0)
        + 0.5 * frame["receptions"].fillna(0)
    )
    frame["opportunities"] = frame["carries"].fillna(0) + frame["targets"].fillna(0)
    qb_rows = frame["position"] == "QB"
    frame.loc[qb_rows, "opportunities"] = (
        frame.loc[qb_rows, "attempts"].fillna(0)
        + frame.loc[qb_rows, "carries"].fillna(0)
    )
    return frame


def build_half_ppr_results(
    stats: pd.DataFrame, year: int, snap_counts: pd.DataFrame | None = None
) -> pd.DataFrame:
    required = {
        "player_id",
        "player_display_name",
        "position",
        "team",
        "season",
        "week",
        "season_type",
        "fantasy_points",
        "passing_interceptions",
        "receptions",
        "attempts",
        "carries",
        "targets",
    }
    missing = required.difference(stats.columns)
    if missing:
        raise ValueError(f"nflverse data is missing columns: {sorted(missing)}")

    frame = _scored_regular_season(stats, year)
    frame = frame.sort_values(["player_id", "week"])
    identity = frame.groupby("player_id", as_index=False).agg(
        Player=("player_display_name", "last"),
        Pos=("position", "last"),
        Team=("team", "last"),
        GP=("week", "nunique"),
        OPP=("opportunities", "sum"),
        TTL=("half_ppr", "sum"),
    )
    identity["_player_key"] = identity["Player"].map(_player_key)

    weekly = frame.pivot_table(
        index="player_id", columns="week", values="half_ppr", aggfunc="sum"
    )
    weekly = weekly.reindex(columns=range(1, 19))
    weekly.columns = [str(column) for column in weekly.columns]
    weekly = weekly.reset_index()

    results = identity.merge(weekly, on="player_id", validate="one_to_one")
    if snap_counts is not None:
        snaps = snap_counts.loc[
            (snap_counts["season"] == year)
            & (snap_counts["game_type"] == "REG")
            & snap_counts["position"].isin(POSITIONS)
            & (
                (snap_counts["offense_snaps"].fillna(0) > 0)
                | (snap_counts["st_snaps"].fillna(0) > 0)
            )
        ].copy()
        snaps["_player_key"] = snaps["player"].map(_player_key)
        participation = snaps.groupby(
            ["_player_key", "position"], as_index=False
        ).agg(
            snap_games=("game_id", "nunique"),
            active_weeks=("week", lambda weeks: tuple(sorted(set(weeks)))),
        )
        results = results.merge(
            participation,
            left_on=["_player_key", "Pos"],
            right_on=["_player_key", "position"],
            how="left",
            validate="many_to_one",
        )
        results["GP"] = results["snap_games"].fillna(results["GP"]).astype(int)
        for index, active_weeks in results["active_weeks"].items():
            if not isinstance(active_weeks, tuple):
                continue
            for week in active_weeks:
                column = str(week)
                if column in results and pd.isna(results.at[index, column]):
                    results.at[index, column] = 0.0
        results = results.drop(columns=["position", "snap_games", "active_weeks"])

    results["AVG"] = results["TTL"] / results["GP"]
    results["PPO"] = results["TTL"] / results["OPP"].replace(0, pd.NA)
    results["PPO"] = pd.to_numeric(results["PPO"], errors="coerce")
    results = results.sort_values("TTL", ascending=False, ignore_index=True)
    results.insert(0, "Rank", range(1, len(results) + 1))
    results = results.rename(columns={"player_id": "PlayerID"})
    results = results.drop(columns="_player_key")
    results["AVG"] = results["AVG"].round(1)
    results["TTL"] = results["TTL"].round(1)
    results["OPP"] = results["OPP"].round(1)
    results["PPO"] = results["PPO"].round(3)
    for week in map(str, range(1, 19)):
        results[week] = results[week].round(1)
    columns = [
        "Rank",
        "PlayerID",
        "Player",
        "Pos",
        "Team",
        "GP",
        "OPP",
        *map(str, range(1, 19)),
        "AVG",
        "PPO",
        "TTL",
    ]
    return results[columns]


def build_player_team_context(stats: pd.DataFrame, year: int) -> pd.DataFrame:
    """Preserve each player's production and opportunity for the correct team."""
    frame = _scored_regular_season(stats, year)
    frame = frame.dropna(subset=["team", "position", "player_id"])
    player_team = frame.groupby(
        ["team", "position", "player_id"], as_index=False
    ).agg(
        player=("player_display_name", "last"),
        player_points=("half_ppr", "sum"),
        player_opportunities=("opportunities", "sum"),
    )
    player_team.insert(0, "season", year)
    return player_team.sort_values(
        ["position", "team", "player_points"],
        ascending=[True, True, False],
        ignore_index=True,
    )


def build_team_position_context(stats: pd.DataFrame, year: int) -> pd.DataFrame:
    """Aggregate actual weekly production by franchise and fantasy position."""
    player_team = build_player_team_context(stats, year)
    context = player_team.groupby(["team", "position"], as_index=False).agg(
        total_points=("player_points", "sum"),
        leader_points=("player_points", "max"),
        top_two_points=("player_points", lambda values: values.nlargest(2).sum()),
        total_opportunities=("player_opportunities", "sum"),
        leader_opportunities=("player_opportunities", "max"),
        top_two_opportunities=(
            "player_opportunities",
            lambda values: values.nlargest(2).sum(),
        ),
        contributors=("player_id", "nunique"),
    )
    context["position_rank"] = context.groupby("position")["total_points"].rank(
        method="min", ascending=False
    )
    context["opportunity_rank"] = context.groupby("position")[
        "total_opportunities"
    ].rank(method="min", ascending=False)
    context.insert(0, "season", year)
    return context.sort_values(
        ["position", "position_rank", "team"], ignore_index=True
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("years", type=int, nargs="+")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw",
    )
    args = parser.parse_args()

    for year in sorted(set(args.years)):
        destination = args.data_dir / str(year) / f"Post_{year}_Results(HPPR).csv"
        stats_url = URL_TEMPLATE.format(year=year)
        snaps_url = SNAP_URL_TEMPLATE.format(year=year)
        stats = download_weekly_stats(year)
        snap_counts = download_snap_counts(year)
        results = build_half_ppr_results(stats, year, snap_counts=snap_counts)
        context = build_team_position_context(stats, year)
        player_team_context = build_player_team_context(stats, year)
        write_csv(results, destination, index=False, na_rep="-")
        context_destination = (
            args.data_dir / str(year) / f"Team_Position_{year}_Context(HPPR).csv"
        )
        write_csv(context, context_destination, index=False)
        player_context_destination = (
            args.data_dir / str(year) / f"Player_Team_{year}_Context(HPPR).csv"
        )
        write_csv(player_team_context, player_context_destination, index=False)
        metadata = {
            "season": year,
            "source": "nflverse",
            "stats_url": stats_url,
            "snap_counts_url": snaps_url,
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "scoring": {
                "reception": 0.5,
                "passing_interception": -1,
                "base": "nflverse fantasy_points",
            },
            "positions": list(POSITIONS),
            "players": len(results),
            "team_position_context_file": context_destination.name,
            "team_position_rows": len(context),
            "player_team_context_file": player_context_destination.name,
            "player_team_rows": len(player_team_context),
        }
        write_text(
            json.dumps(metadata, indent=2) + "\n",
            destination.with_suffix(".meta.json"),
        )
        print(f"Wrote {len(results)} players to {destination}")
        print(f"Wrote {len(context)} team-position rows to {context_destination}")
        print(
            f"Wrote {len(player_team_context)} player-team rows to "
            f"{player_context_destination}"
        )


if __name__ == "__main__":
    main()
