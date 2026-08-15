#!/usr/bin/env python3
"""Import point-in-time nflverse depth charts and weekly roster status."""

from __future__ import annotations

import argparse
from datetime import datetime, time, timezone
import io
import json
from pathlib import Path

import pandas as pd
import requests

DEPTH_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "depth_charts/depth_charts_{year}.csv.gz"
)
ROSTER_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "weekly_rosters/roster_weekly_{year}.csv"
)
POSITIONS = ("QB", "RB", "WR", "TE")


def download_csv(url: str, compressed: bool = False) -> pd.DataFrame:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return pd.read_csv(
        io.BytesIO(response.content),
        compression="gzip" if compressed else None,
        low_memory=False,
    )


def adp_cutoff(metadata: dict[str, object]) -> pd.Timestamp | None:
    """Return the inclusive end of the ADP sample's final calendar day."""
    source_meta = metadata.get("source_meta", {})
    if not isinstance(source_meta, dict) or not source_meta.get("end_date"):
        return None
    end_date = datetime.fromisoformat(str(source_meta["end_date"])).date()
    return pd.Timestamp(datetime.combine(end_date, time.max, tzinfo=timezone.utc))


def build_depth_export(
    depth: pd.DataFrame,
    season: int,
    cutoff: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Normalize both nflverse depth-chart schemas without hiding timing quality."""
    if "dt" in depth.columns:
        required = {
            "dt",
            "team",
            "player_name",
            "gsis_id",
            "pos_abb",
            "pos_slot",
            "pos_rank",
            "pos_name",
        }
        missing = required.difference(depth.columns)
        if missing:
            raise ValueError(f"Dated depth chart is missing: {sorted(missing)}")
        frame = depth.copy()
        frame["parsed_dt"] = pd.to_datetime(frame["dt"], utc=True, errors="coerce")
        if cutoff is not None:
            frame = frame[frame["parsed_dt"] <= cutoff].copy()
        if frame.empty:
            raise ValueError(f"No {season} depth chart exists at or before the cutoff.")
        snapshot_at = frame["parsed_dt"].max()
        frame = frame[frame["parsed_dt"] == snapshot_at].copy()
        frame = frame.rename(
            columns={
                "player_name": "player",
                "pos_abb": "pos",
                "pos_rank": "depth_rank",
                "pos_slot": "depth_slot",
                "pos_name": "depth_position",
            }
        )
        frame["source_schema"] = "dated_espn"
        frame["timing_quality"] = (
            "adp_aligned" if cutoff is not None else "latest_available"
        )
        frame["snapshot_at"] = snapshot_at.isoformat()
        rank = pd.to_numeric(frame["depth_rank"], errors="coerce")
        starter_limit = frame["pos"].map({"QB": 1, "RB": 1, "WR": 3, "TE": 1})
        frame["official_starter"] = rank.le(starter_limit).astype(int)
        metadata = {
            "source_schema": "dated_espn",
            "snapshot_at": snapshot_at.isoformat(),
            "timing_quality": frame["timing_quality"].iloc[0],
            "point_in_time_eligible": cutoff is not None,
        }
    else:
        required = {
            "club_code",
            "week",
            "game_type",
            "depth_team",
            "formation",
            "gsis_id",
            "position",
            "depth_position",
            "full_name",
        }
        missing = required.difference(depth.columns)
        if missing:
            raise ValueError(f"Legacy depth chart is missing: {sorted(missing)}")
        frame = depth.copy()
        frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
        frame = frame[
            frame["game_type"].eq("REG")
            & frame["formation"].eq("Offense")
            & frame["week"].eq(frame.loc[frame["game_type"].eq("REG"), "week"].min())
        ].copy()
        frame = frame.rename(
            columns={
                "club_code": "team",
                "full_name": "player",
                "position": "pos",
                "depth_team": "depth_rank",
                "depth_position": "depth_position",
            }
        )
        frame["depth_slot"] = frame["depth_position"]
        frame["source_schema"] = "weekly_legacy"
        frame["timing_quality"] = "opening_week_proxy"
        frame["snapshot_at"] = f"{season}-week-1"
        frame["official_starter"] = (
            pd.to_numeric(frame["depth_rank"], errors="coerce").eq(1).astype(int)
        )
        metadata = {
            "source_schema": "weekly_legacy",
            "snapshot_at": f"{season}-week-1",
            "timing_quality": "opening_week_proxy",
            "point_in_time_eligible": False,
        }

    frame["pos"] = frame["pos"].astype("string").str.strip().str.upper()
    frame = frame[frame["pos"].isin(POSITIONS)].copy()
    frame["depth_rank"] = pd.to_numeric(frame["depth_rank"], errors="coerce")
    frame["season"] = season
    columns = [
        "season",
        "snapshot_at",
        "source_schema",
        "timing_quality",
        "team",
        "player",
        "gsis_id",
        "pos",
        "depth_rank",
        "depth_slot",
        "depth_position",
        "official_starter",
    ]
    frame = frame[columns].dropna(subset=["team", "player", "pos", "depth_rank"])
    frame = frame.sort_values(["team", "pos", "depth_rank", "player"])
    frame = frame.drop_duplicates(["team", "pos", "gsis_id"], keep="first")
    return frame.reset_index(drop=True), metadata


def build_roster_export(roster: pd.DataFrame, season: int) -> pd.DataFrame:
    required = {
        "season",
        "team",
        "position",
        "full_name",
        "gsis_id",
        "week",
        "game_type",
        "status",
        "status_description_abbr",
    }
    missing = required.difference(roster.columns)
    if missing:
        raise ValueError(f"Weekly roster is missing: {sorted(missing)}")
    frame = roster.copy()
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    regular = frame["game_type"].eq("REG")
    first_week = frame.loc[regular, "week"].min()
    frame = frame[regular & frame["week"].eq(first_week)].copy()
    frame = frame.rename(
        columns={
            "position": "pos",
            "full_name": "roster_player",
            "status": "roster_status",
            "status_description_abbr": "roster_status_description",
        }
    )
    frame["pos"] = frame["pos"].astype("string").str.strip().str.upper()
    frame = frame[frame["pos"].isin(POSITIONS)].copy()
    columns = [
        "team",
        "pos",
        "gsis_id",
        "roster_player",
        "roster_status",
        "roster_status_description",
    ]
    return frame[columns].drop_duplicates(["team", "pos", "gsis_id"], keep="last")


def build_role_export(
    depth: pd.DataFrame,
    roster: pd.DataFrame,
    season: int,
    cutoff: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    depth_export, metadata = build_depth_export(depth, season, cutoff)
    roster_export = build_roster_export(roster, season)
    role = depth_export.merge(
        roster_export,
        on=["team", "pos", "gsis_id"],
        how="left",
        validate="one_to_one",
    )
    role["player"] = role["player"].fillna(role["roster_player"])
    return role.drop(columns="roster_player"), metadata


def write_role_export(
    role: pd.DataFrame, destination: Path, metadata: dict[str, object]
) -> Path:
    """Write the current role file and preserve an immutable fetched snapshot."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    role.to_csv(destination, index=False)
    metadata_text = json.dumps(metadata, indent=2) + "\n"
    destination.with_suffix(".meta.json").write_text(
        metadata_text, encoding="utf-8"
    )
    fetched = datetime.fromisoformat(str(metadata["fetched_at_utc"]))
    stamp = fetched.strftime("%Y%m%dT%H%M%SZ")
    snapshot = destination.parent / "snapshots" / f"Role_{stamp}.csv"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    role.to_csv(snapshot, index=False)
    snapshot.with_suffix(".meta.json").write_text(metadata_text, encoding="utf-8")
    return snapshot


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
        season_dir = args.data_dir / str(year)
        adp_metadata_path = season_dir / f"Pre_{year}_ADP(HPPR).meta.json"
        adp_metadata = (
            json.loads(adp_metadata_path.read_text(encoding="utf-8"))
            if adp_metadata_path.exists()
            else {}
        )
        cutoff = adp_cutoff(adp_metadata)
        depth_url = DEPTH_URL.format(year=year)
        roster_url = ROSTER_URL.format(year=year)
        depth = download_csv(depth_url, compressed=True)
        roster = download_csv(roster_url)
        role, role_metadata = build_role_export(depth, roster, year, cutoff)
        destination = season_dir / f"Preseason_Role_{year}.csv"
        metadata = {
            "season": year,
            "source": "nflverse depth charts and weekly rosters",
            "depth_source_url": depth_url,
            "roster_source_url": roster_url,
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "adp_cutoff": cutoff.isoformat() if cutoff is not None else None,
            "positions": list(POSITIONS),
            "players": len(role),
            **role_metadata,
        }
        snapshot = write_role_export(role, destination, metadata)
        print(
            f"Wrote {len(role)} role rows to {destination} "
            f"({role_metadata['timing_quality']})"
        )
        print(f"Preserved timestamped snapshot at {snapshot}")


if __name__ == "__main__":
    main()
