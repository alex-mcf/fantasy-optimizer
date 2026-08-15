"""Normalize a draft platform's current rankings for live-board decisions."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from fantasyoptimizer.utils.data_loader import _normalize_player_key

PLATFORM_OPTIONS = (
    "Fantasy Football Calculator",
    "ESPN",
    "Yahoo",
    "Sleeper",
    "NFL.com",
    "Custom rankings",
)

PLAYER_COLUMNS = ("player", "playername", "name", "fullname", "full_name")
ADP_COLUMNS = (
    "adp",
    "avg",
    "average",
    "overall",
    "overallrank",
    "rank",
    "rk",
)
POSITION_COLUMNS = ("pos", "position", "positionname")
DEVIATION_COLUMNS = ("stddev", "stdev", "sd", "deviation")


def _column_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _find_column(columns: dict[str, object], candidates: tuple[str, ...]) -> object | None:
    for candidate in candidates:
        key = _column_key(candidate)
        if key in columns:
            return columns[key]
    return None


def parse_platform_adp(raw: pd.DataFrame) -> pd.DataFrame:
    """Accept common ranking exports and return a small canonical player/ADP table."""
    if raw.empty:
        raise ValueError("The uploaded platform ranking file is empty.")
    columns = {_column_key(column): column for column in raw.columns}
    player_column = _find_column(columns, PLAYER_COLUMNS)
    adp_column = _find_column(columns, ADP_COLUMNS)
    if player_column is None or adp_column is None:
        raise ValueError(
            "Platform CSV needs a Player/Name column and an ADP/Rank column."
        )
    position_column = _find_column(columns, POSITION_COLUMNS)
    deviation_column = _find_column(columns, DEVIATION_COLUMNS)
    parsed = pd.DataFrame(
        {
            "platform_player": raw[player_column].astype("string").str.strip(),
            "draft_adp": pd.to_numeric(raw[adp_column], errors="coerce"),
        }
    )
    parsed["player_key"] = _normalize_player_key(parsed["platform_player"])
    parsed["pos"] = (
        raw[position_column]
        .astype("string")
        .str.strip()
        .str.upper()
        .str.extract(r"^(QB|RB|WR|TE)", expand=False)
        if position_column is not None
        else pd.Series(pd.NA, index=raw.index, dtype="string")
    )
    parsed["draft_adp_stddev"] = (
        pd.to_numeric(raw[deviation_column], errors="coerce")
        if deviation_column is not None
        else np.nan
    )
    parsed = parsed.dropna(subset=["player_key", "draft_adp"])
    parsed = parsed[parsed["draft_adp"].gt(0)].sort_values("draft_adp")
    return parsed.drop_duplicates(["player_key", "pos"], keep="first").reset_index(
        drop=True
    )


def apply_platform_adp(
    forecast: pd.DataFrame,
    platform_adp: pd.DataFrame | None,
    platform_name: str,
) -> pd.DataFrame:
    """Attach a selected platform's cost while preserving the model's FFC baseline."""
    board = forecast.copy()
    board["draft_platform"] = platform_name
    board["draft_adp"] = board["adp_avg"].astype(float)
    board["draft_adp_stddev"] = (
        board["stddev"].astype(float)
        if "stddev" in board
        else pd.Series(12.0, index=board.index)
    )
    board["platform_match"] = platform_name == PLATFORM_OPTIONS[0]

    if platform_adp is not None and not platform_adp.empty:
        positioned = {
            (row.player_key, row.pos): row
            for row in platform_adp.itertuples()
            if pd.notna(row.pos)
        }
        by_name = {
            row.player_key: row
            for row in platform_adp.sort_values("draft_adp").itertuples()
        }
        for index, player in board.iterrows():
            match = positioned.get((player["player_key"], player["pos"]))
            if match is None:
                match = by_name.get(player["player_key"])
            if match is None:
                continue
            board.at[index, "draft_adp"] = float(match.draft_adp)
            board.at[index, "platform_match"] = True
            if pd.notna(match.draft_adp_stddev):
                board.at[index, "draft_adp_stddev"] = float(
                    match.draft_adp_stddev
                )

    board["draft_market_rank"] = board["draft_adp"].rank(
        method="first", ascending=True
    ).astype(int)
    board["platform_actionable_adp"] = (
        0.75 * board["draft_market_rank"] + 0.25 * board["fair_adp"]
    ).round().astype(int)
    board["platform_value_gap"] = (
        board["draft_market_rank"] - board["fair_adp"]
    )
    board["platform_pick_edge"] = (
        board["draft_adp"] - board["platform_actionable_adp"]
    )
    board["platform_source"] = np.where(
        board["platform_match"], platform_name, "FFC fallback"
    )
    return board
