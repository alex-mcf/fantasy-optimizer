#!/usr/bin/env python3
"""Validate local provenance, schemas, uniqueness, and cross-source match rates."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasyoptimizer.utils.data_loader import (  # noqa: E402
    available_adp_years,
    available_context_years,
    available_player_context_years,
    available_result_years,
    data_quality_report,
    load_adp,
    load_results,
    load_player_team_context,
    load_team_position_context,
)


def validate_archive(data_dir: Path) -> list[str]:
    errors: list[str] = []
    adp_years = available_adp_years(data_dir)
    result_years = available_result_years(data_dir)
    context_years = set(available_context_years(data_dir))
    player_context_years = set(available_player_context_years(data_dir))
    for year in adp_years:
        csv_path = data_dir / str(year) / f"Pre_{year}_ADP(HPPR).csv"
        meta_path = csv_path.with_suffix(".meta.json")
        if not meta_path.exists():
            errors.append(f"{year} ADP is missing provenance metadata")
        else:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if metadata.get("source") != "Fantasy Football Calculator":
                errors.append(f"{year} ADP has a different source")
            if metadata.get("parameters", {}).get("scoring") != "half-ppr":
                errors.append(f"{year} ADP is not marked half-PPR")
        adp = load_adp(year, data_dir)
        if adp.duplicated(["player_key", "pos"]).any():
            errors.append(f"{year} ADP contains duplicate player-position rows")

    for year in result_years:
        csv_path = data_dir / str(year) / f"Post_{year}_Results(HPPR).csv"
        meta_path = csv_path.with_suffix(".meta.json")
        if not meta_path.exists():
            errors.append(f"{year} results are missing provenance metadata")
        else:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if metadata.get("source") != "nflverse":
                errors.append(f"{year} results have a different source")
            if metadata.get("scoring", {}).get("reception") != 0.5:
                errors.append(f"{year} results are not marked half-PPR")
            if not metadata.get("team_position_context_file"):
                errors.append(f"{year} results metadata omits team-position context")
            if not metadata.get("player_team_context_file"):
                errors.append(f"{year} results metadata omits player-team context")
        results = load_results(year, data_dir)
        if results.duplicated(["player_key", "pos"]).any():
            errors.append(f"{year} results contain duplicate player-position rows")
        if (results["gp"] < 0).any():
            errors.append(f"{year} results contain negative games played")
        if results[["pts_ttl", "pts_avg", "gp"]].isna().any().any():
            errors.append(f"{year} results contain missing numeric values")
        if year not in context_years:
            errors.append(f"{year} is missing team-position context")
        else:
            context = load_team_position_context(year, data_dir)
            if context.duplicated(["year", "team", "pos"]).any():
                errors.append(f"{year} team-position context contains duplicates")
            expected_positions = {"QB", "RB", "WR", "TE"}
            if set(context["pos"]) != expected_positions:
                errors.append(f"{year} team-position context has incomplete positions")
        if year not in player_context_years:
            errors.append(f"{year} is missing player-team context")
        else:
            player_context = load_player_team_context(year, data_dir)
            if player_context.duplicated(
                ["year", "team", "pos", "player_key"]
            ).any():
                errors.append(f"{year} player-team context contains duplicates")
            if (
                player_context[["player_opportunities"]].fillna(0) < 0
            ).any().any():
                errors.append(f"{year} player-team context has negative opportunities")
    return errors


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "raw"
    errors = validate_archive(data_dir)
    adp_years = available_adp_years(data_dir)
    result_years = available_result_years(data_dir)
    complete = sorted(set(adp_years) & set(result_years))
    print(f"ADP seasons: {adp_years}")
    print(f"Result seasons: {result_years}")
    print(f"Team-position context seasons: {available_context_years(data_dir)}")
    print(f"Player-team context seasons: {available_player_context_years(data_dir)}")
    print(f"Complete retrospective seasons: {complete}")
    if complete:
        print(data_quality_report(complete, data_dir).to_string(index=False))
    if errors:
        print("\nValidation errors:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("\nArchive validation passed.")


if __name__ == "__main__":
    main()
