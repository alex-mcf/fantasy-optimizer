#!/usr/bin/env python3
"""Import consistent half-PPR mock-draft ADP from Fantasy Football Calculator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://fantasyfootballcalculator.com/api/v1/adp/half-ppr"
POSITIONS = ("QB", "RB", "WR", "TE")


def download_adp(year: int, teams: int = 12) -> tuple[pd.DataFrame, dict]:
    params = {"teams": teams, "year": year, "position": "all"}
    response = requests.get(API_URL, params=params, timeout=90)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "Success" or not payload.get("players"):
        raise ValueError(f"Fantasy Football Calculator returned no ADP for {year}.")
    return pd.DataFrame(payload["players"]), payload.get("meta", {})


def build_adp_export(players: pd.DataFrame) -> pd.DataFrame:
    required = {
        "player_id",
        "name",
        "position",
        "team",
        "adp",
        "times_drafted",
        "high",
        "low",
        "stdev",
    }
    missing = required.difference(players.columns)
    if missing:
        raise ValueError(f"ADP response is missing columns: {sorted(missing)}")
    valid_name = ~players["name"].astype("string").str.contains(
        r"^deleted(?: deleted)?$", case=False, na=False, regex=True
    )
    frame = players.loc[players["position"].isin(POSITIONS) & valid_name].copy()
    frame = frame.sort_values(["adp", "name"], ignore_index=True)
    frame.insert(0, "Rank", range(1, len(frame) + 1))
    frame = frame.rename(
        columns={
            "player_id": "PlayerID",
            "name": "Player",
            "position": "POS",
            "team": "Team",
            "adp": "AVG",
            "times_drafted": "TimesDrafted",
            "high": "High",
            "low": "Low",
            "stdev": "StdDev",
        }
    )
    columns = [
        "Rank",
        "PlayerID",
        "Player",
        "Team",
        "POS",
        "AVG",
        "TimesDrafted",
        "High",
        "Low",
        "StdDev",
    ]
    return frame[columns]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("years", type=int, nargs="+")
    parser.add_argument("--teams", type=int, default=12)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw",
    )
    args = parser.parse_args()

    for year in sorted(set(args.years)):
        players, source_meta = download_adp(year, args.teams)
        adp = build_adp_export(players)
        destination = args.data_dir / str(year) / f"Pre_{year}_ADP(HPPR).csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        adp.to_csv(destination, index=False)
        metadata = {
            "season": year,
            "source": "Fantasy Football Calculator",
            "source_url": API_URL,
            "parameters": {
                "scoring": "half-ppr",
                "teams": args.teams,
                "position": "all",
            },
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_meta": source_meta,
            "positions": list(POSITIONS),
            "players": len(adp),
        }
        destination.with_suffix(".meta.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote {len(adp)} players to {destination}")


if __name__ == "__main__":
    main()
