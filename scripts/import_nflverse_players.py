#!/usr/bin/env python3
"""Import nflverse player biography and draft metadata for forecast features."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasyoptimizer.utils.atomic import write_csv, write_text  # noqa: E402

PLAYERS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/players/players.csv"
)
POSITIONS = ("QB", "RB", "WR", "TE")


def download_players() -> pd.DataFrame:
    response = requests.get(PLAYERS_URL, timeout=90)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


def build_player_export(players: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "gsis_id",
        "display_name",
        "birth_date",
        "position_group",
        "position",
        "height",
        "weight",
        "college_name",
        "rookie_season",
        "last_season",
        "latest_team",
        "status",
        "years_of_experience",
        "draft_year",
        "draft_round",
        "draft_pick",
        "draft_team",
    ]
    missing = set(columns).difference(players.columns)
    if missing:
        raise ValueError(f"Player response is missing columns: {sorted(missing)}")
    frame = players.loc[players["position_group"].isin(POSITIONS), columns].copy()
    frame = frame.dropna(subset=["gsis_id", "display_name", "position_group"])
    frame = frame.sort_values(
        ["display_name", "position_group", "rookie_season", "gsis_id"]
    ).drop_duplicates("gsis_id", keep="last")
    return frame.reset_index(drop=True)


def main() -> None:
    destination = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "raw"
        / "players"
        / "players.csv"
    )
    players = build_player_export(download_players())
    write_csv(players, destination, index=False)
    metadata = {
        "source": "nflverse players v2",
        "source_url": PLAYERS_URL,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "positions": list(POSITIONS),
        "players": len(players),
        "fields": list(players.columns),
    }
    write_text(
        json.dumps(metadata, indent=2) + "\n", destination.with_suffix(".meta.json")
    )
    print(f"Wrote {len(players)} player metadata rows to {destination}")


if __name__ == "__main__":
    main()
