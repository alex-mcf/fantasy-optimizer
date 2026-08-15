import json
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _default_data_dir() -> Path:
    configured = os.environ.get("FANTASY_OPTIMIZER_DATA_DIR")
    if configured:
        return Path(configured).expanduser()
    working_tree_data = Path.cwd() / "data" / "raw"
    if working_tree_data.exists():
        return working_tree_data
    return PROJECT_ROOT / "data" / "raw"


DEFAULT_DATA_DIR = _default_data_dir()

TEAM_ALIASES = {
    "JAC": "JAX",
    "LA": "LAR",
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LAR",
    "WSH": "WAS",
}


def _normalize_position(series: pd.Series) -> pd.Series:
    """Convert ADP positions such as RB12 to their base position (RB)."""
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .str.extract(r"^([A-Z/]+)", expand=False)
    )


def _normalize_team(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper().replace(TEAM_ALIASES)


def _normalize_player_key(series: pd.Series) -> pd.Series:
    """Create a conservative cross-source matching key from a player name."""
    normalized = (
        series.astype("string")
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
        .str.lower()
        .str.strip()
        .str.replace(r"\b(jr|sr|ii|iii|iv)\.?$", "", regex=True)
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )
    return normalized.replace(
        {
            "hollywoodbrown": "marquisebrown",
            "joshuapalmer": "joshpalmer",
        }
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing data file: {path}. See README.md for data setup instructions."
        )
    # Do not interpret legitimate team abbreviations such as "NA" as missing.
    return pd.read_csv(path, keep_default_na=False, na_values=["", "-"])


def available_years(data_dir: Path | str = DEFAULT_DATA_DIR) -> list[int]:
    """Return years that contain both preseason ADP and postseason results."""
    data_dir = Path(data_dir)
    years: list[int] = []
    if not data_dir.exists():
        return years
    for year_dir in data_dir.iterdir():
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)
        if (
            (year_dir / f"Pre_{year}_ADP(HPPR).csv").exists()
            and (year_dir / f"Post_{year}_Results(HPPR).csv").exists()
        ):
            years.append(year)
    return sorted(years)


def available_adp_years(data_dir: Path | str = DEFAULT_DATA_DIR) -> list[int]:
    """Return years that contain a preseason ADP file."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    return sorted(
        int(path.name)
        for path in data_dir.iterdir()
        if path.is_dir()
        and path.name.isdigit()
        and (path / f"Pre_{path.name}_ADP(HPPR).csv").exists()
    )


def available_result_years(data_dir: Path | str = DEFAULT_DATA_DIR) -> list[int]:
    """Return years that contain a completed-season results file."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    return sorted(
        int(path.name)
        for path in data_dir.iterdir()
        if path.is_dir()
        and path.name.isdigit()
        and (path / f"Post_{path.name}_Results(HPPR).csv").exists()
    )


def available_context_years(data_dir: Path | str = DEFAULT_DATA_DIR) -> list[int]:
    """Return years that contain weekly-derived team-position context."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    return sorted(
        int(path.name)
        for path in data_dir.iterdir()
        if path.is_dir()
        and path.name.isdigit()
        and (path / f"Team_Position_{path.name}_Context(HPPR).csv").exists()
    )


def available_player_context_years(
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> list[int]:
    """Return years that contain weekly-derived player-team contributions."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    return sorted(
        int(path.name)
        for path in data_dir.iterdir()
        if path.is_dir()
        and path.name.isdigit()
        and (path / f"Player_Team_{path.name}_Context(HPPR).csv").exists()
    )


def _normalize_adp_frame(df: pd.DataFrame, year: int) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower()
    df["player"] = df["player"].str.strip().str.lower()
    df["player_key"] = _normalize_player_key(df["player"])
    df["pos"] = _normalize_position(df["pos"])
    df["team"] = _normalize_team(df["team"])
    df = df.rename(columns={"avg": "adp_avg"})
    keep_columns = ["rank", "player", "player_key", "team", "pos", "adp_avg"]
    optional_columns = ["playerid", "timesdrafted", "high", "low", "stddev"]
    keep_columns.extend(column for column in optional_columns if column in df.columns)
    df = df[keep_columns].copy()
    numeric_columns = ["rank", "adp_avg", "timesdrafted", "high", "low", "stddev"]
    for column in numeric_columns:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["player", "pos", "adp_avg"])
    df["year"] = year
    return df


def load_adp(year: int, data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    path = Path(data_dir) / str(year) / f"Pre_{year}_ADP(HPPR).csv"
    return _normalize_adp_frame(_read_csv(path), year)


def load_adp_metadata(
    year: int, data_dir: Path | str = DEFAULT_DATA_DIR
) -> dict[str, object]:
    path = Path(data_dir) / str(year) / f"Pre_{year}_ADP(HPPR).meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def available_adp_snapshots(
    year: int, data_dir: Path | str = DEFAULT_DATA_DIR
) -> list[Path]:
    snapshot_dir = Path(data_dir) / str(year) / "snapshots"
    if not snapshot_dir.exists():
        return []
    return sorted(snapshot_dir.glob("ADP_*.csv"))


def load_adp_snapshot_history(
    year: int, data_dir: Path | str = DEFAULT_DATA_DIR
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in available_adp_snapshots(year, data_dir):
        frame = _normalize_adp_frame(_read_csv(path), year)
        metadata_path = path.with_suffix(".meta.json")
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        frame["snapshot_at"] = pd.to_datetime(
            metadata.get("fetched_at_utc"), utc=True, errors="coerce"
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["snapshot_at", "adp_avg"], ignore_index=True
    )


def build_adp_movement(
    year: int, data_dir: Path | str = DEFAULT_DATA_DIR
) -> pd.DataFrame:
    history = load_adp_snapshot_history(year, data_dir)
    if history.empty or history["snapshot_at"].nunique() < 2:
        return pd.DataFrame()
    ordered = history.sort_values("snapshot_at")
    movement = ordered.groupby(["player_key", "pos"], as_index=False).agg(
        player=("player", "last"),
        team=("team", "last"),
        first_snapshot=("snapshot_at", "first"),
        latest_snapshot=("snapshot_at", "last"),
        first_adp=("adp_avg", "first"),
        latest_adp=("adp_avg", "last"),
        snapshots=("snapshot_at", "nunique"),
    )
    movement["adp_movement"] = movement["first_adp"] - movement["latest_adp"]
    return movement.sort_values("adp_movement", ascending=False, ignore_index=True)


# Load results data for a given year
def load_results(year: int, data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    path = Path(data_dir) / str(year) / f"Post_{year}_Results(HPPR).csv"
    df = _read_csv(path)

    # normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # normalize key columns
    df["player"] = df["player"].str.strip().str.lower()
    df["player_key"] = _normalize_player_key(df["player"])
    df["pos"] = _normalize_position(df["pos"])
    df["team"] = _normalize_team(df["team"])

    # rename avg / ttl to something results-specific
    df = df.rename(columns={"avg": "pts_avg", "ttl": "pts_ttl"})

    # keep only what we care about
    keep_columns = ["rank", "player", "player_key", "team", "pos", "pts_avg", "pts_ttl"]
    if "gp" in df.columns:
        keep_columns.append("gp")
    for optional in ["opp", "ppo"]:
        if optional in df.columns:
            keep_columns.append(optional)
    if "player_id" in df.columns:
        keep_columns.append("player_id")
    df = df[keep_columns].copy()
    for column in ["rank", "pts_avg", "pts_ttl", "gp", "opp", "ppo"]:
        if column not in df:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["player", "pos", "pts_avg", "pts_ttl"])
    if "gp" not in df:
        df["gp"] = (df["pts_ttl"] / df["pts_avg"].replace(0, pd.NA)).round()
    df["gp"] = df["gp"].fillna(0).astype(int)
    if "opp" not in df:
        df["opp"] = 0.0
    if "ppo" not in df:
        df["ppo"] = df["pts_ttl"] / df["opp"].replace(0, pd.NA)
    df["opp"] = pd.to_numeric(df["opp"], errors="coerce").fillna(0.0)
    df["ppo"] = pd.to_numeric(df["ppo"], errors="coerce").fillna(0.0)
    df["year"] = year
    return df


def load_team_position_context(
    year: int, data_dir: Path | str = DEFAULT_DATA_DIR
) -> pd.DataFrame:
    path = Path(data_dir) / str(year) / f"Team_Position_{year}_Context(HPPR).csv"
    df = _read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    required = {
        "season",
        "team",
        "position",
        "total_points",
        "leader_points",
        "top_two_points",
        "total_opportunities",
        "leader_opportunities",
        "top_two_opportunities",
        "contributors",
        "position_rank",
        "opportunity_rank",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Team-position context is missing columns: {sorted(missing)}")
    df = df.rename(columns={"season": "year", "position": "pos"})
    df["team"] = _normalize_team(df["team"])
    df["pos"] = _normalize_position(df["pos"])
    for column in [
        "year",
        "total_points",
        "leader_points",
        "top_two_points",
        "total_opportunities",
        "leader_opportunities",
        "top_two_opportunities",
        "contributors",
        "position_rank",
        "opportunity_rank",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["year", "team", "pos", "total_points"])
    df["year"] = df["year"].astype(int)
    team_counts = df.groupby(["year", "pos"])["team"].transform("nunique")
    df["rank_percentile"] = 1 - (
        (df["position_rank"] - 1) / (team_counts - 1).clip(lower=1)
    )
    df["opportunity_rank_percentile"] = 1 - (
        (df["opportunity_rank"] - 1) / (team_counts - 1).clip(lower=1)
    )
    return df


def load_player_team_context(
    year: int, data_dir: Path | str = DEFAULT_DATA_DIR
) -> pd.DataFrame:
    path = Path(data_dir) / str(year) / f"Player_Team_{year}_Context(HPPR).csv"
    df = _read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    required = {
        "season",
        "team",
        "position",
        "player_id",
        "player",
        "player_points",
        "player_opportunities",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Player-team context is missing columns: {sorted(missing)}")
    df = df.rename(columns={"season": "year", "position": "pos"})
    df["team"] = _normalize_team(df["team"])
    df["pos"] = _normalize_position(df["pos"])
    df["player"] = df["player"].astype("string").str.strip().str.lower()
    df["player_key"] = _normalize_player_key(df["player"])
    for column in ["year", "player_points", "player_opportunities"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["year", "team", "pos", "player_key"])
    df["year"] = df["year"].astype(int)
    return df


def join_year(year: int, data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Join a season's ADP and results for the same player and position."""
    adp_data = load_adp(year, data_dir)
    results_data = load_results(year, data_dir)

    merged = pd.merge(
        adp_data,
        results_data,
        on=["player_key", "pos", "year"],
        suffixes=("_adp", "_results"),
        validate="one_to_one",
    ).copy()

    merged["player"] = merged["player_results"].fillna(merged["player_adp"])
    merged = merged.drop(columns=["player_adp", "player_results"])
    merged = merged.sort_values(by="rank_adp").reset_index(drop=True)
    return merged


def data_quality_report(
    years: list[int], data_dir: Path | str = DEFAULT_DATA_DIR
) -> pd.DataFrame:
    """Summarize ADP/result matching and show representative misses."""
    reports: list[dict[str, object]] = []
    for year in sorted(set(years)):
        adp = load_adp(year, data_dir)
        results = load_results(year, data_dir)
        supported = ["QB", "RB", "WR", "TE"]
        adp = adp[adp["pos"].isin(supported)].copy()
        results = results[results["pos"].isin(supported)].copy()
        keys = ["player_key", "pos", "year"]
        audit = adp.merge(
            results,
            on=keys,
            how="outer",
            suffixes=("_adp", "_results"),
            indicator=True,
        )
        matched = int((audit["_merge"] == "both").sum())
        adp_only = audit.loc[audit["_merge"] == "left_only", "player_adp"]
        results_only = audit.loc[audit["_merge"] == "right_only", "player_results"]
        reports.append(
            {
                "year": year,
                "adp_players": len(adp),
                "result_players": len(results),
                "matched_players": matched,
                "adp_match_rate": matched / len(adp) if len(adp) else 0,
                "adp_only": ", ".join(adp_only.dropna().head(5)),
                "results_only": ", ".join(results_only.dropna().head(5)),
            }
        )
    return pd.DataFrame(reports)


# Load and combine data for multiple years
def build_history(
    years: list[int], data_dir: Path | str = DEFAULT_DATA_DIR
) -> pd.DataFrame:
    if not years:
        raise ValueError("No years provided to build_history().")
    dfs = [join_year(year, data_dir) for year in sorted(set(years))]
    return pd.concat(dfs, ignore_index=True).sort_values(
        ["year", "rank_adp"], ignore_index=True
    )
