from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def _normalize_position(series: pd.Series) -> pd.Series:
    """Convert ADP positions such as RB12 to their base position (RB)."""
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .str.extract(r"^([A-Z/]+)", expand=False)
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


def load_adp(year: int, data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    path = Path(data_dir) / str(year) / f"Pre_{year}_ADP(HPPR).csv"
    df = _read_csv(path)

    # normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # normalize key columns
    df["player"] = df["player"].str.strip().str.lower()
    df["pos"] = _normalize_position(df["pos"])

    # rename avg to something ADP-specific
    df = df.rename(columns={"avg": "adp_avg"})

    # keep only what we care about for now
    df = df[["rank", "player", "team", "pos", "adp_avg"]].copy()
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["adp_avg"] = pd.to_numeric(df["adp_avg"], errors="coerce")
    df = df.dropna(subset=["player", "pos", "adp_avg"])
    df["year"] = year
    return df


# Load results data for a given year
def load_results(year: int, data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    path = Path(data_dir) / str(year) / f"Post_{year}_Results(HPPR).csv"
    df = _read_csv(path)

    # normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # normalize key columns
    df["player"] = df["player"].str.strip().str.lower()
    df["pos"] = _normalize_position(df["pos"])

    # rename avg / ttl to something results-specific
    df = df.rename(columns={"avg": "pts_avg", "ttl": "pts_ttl"})

    # keep only what we care about
    df = df[["rank", "player", "team", "pos", "pts_avg", "pts_ttl"]].copy()
    for column in ["rank", "pts_avg", "pts_ttl"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["player", "pos", "pts_avg", "pts_ttl"])
    df["year"] = year
    return df


def join_year(year: int, data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Join a season's ADP and results for the same player and position."""
    adp_data = load_adp(year, data_dir)
    results_data = load_results(year, data_dir)

    merged = pd.merge(
        adp_data,
        results_data,
        on=["player", "pos", "year"],
        suffixes=("_adp", "_results"),
        validate="one_to_one",
    ).copy()

    merged = merged.sort_values(by="rank_adp").reset_index(drop=True)
    return merged


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
