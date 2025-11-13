# data_loader.py
import pandas as pd

# Load ADP data for a given year
def load_adp(year: int) -> pd.DataFrame:
    path = f"data/raw/{year}/Pre_{year}_ADP(HPPR).csv"
    df = pd.read_csv(path)

    # normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # normalize key columns
    df["player"] = df["player"].str.strip().str.lower()
    df["pos"] = df["pos"].str.strip().str.upper()

    # rename avg to something ADP-specific
    df = df.rename(columns={"avg": "adp_avg"})

    # keep only what we care about for now
    df = df[["rank", "player", "team", "pos", "adp_avg"]].copy()
    df["year"] = year
    return df


# Load results data for a given year
def load_results(year: int) -> pd.DataFrame:
    path = f"data/raw/{year}/Post_{year}_Results(HPPR).csv"
    df = pd.read_csv(path)

    # normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # normalize key columns
    df["player"] = df["player"].str.strip().str.lower()
    df["pos"] = df["pos"].str.strip().str.upper()

    # rename avg / ttl to something results-specific
    df = df.rename(columns={"avg": "pts_avg", "ttl": "pts_ttl"})

    # keep only what we care about
    df = df[["rank", "player", "team", "pos", "pts_avg", "pts_ttl"]].copy()
    df["year"] = year
    return df


# Load data for a given year and merge on player and position
def join_year(year: int) -> pd.DataFrame:
    adp_data = load_adp(year)
    results_data = load_results(year)

    merged = pd.merge(
        adp_data,
        results_data,
        on=["player"],
        suffixes=("_adp", "_results")  # distinguish rank/team if you keep both
    ).copy()


    merged["pos"] = merged["pos_results"].fillna(merged["pos_adp"])
    merged = merged.drop(columns=["pos_results", "pos_adp"])
    
    # optionally sort by ADP rank
    merged = merged.sort_values(by="rank_adp").reset_index(drop=True)
    return merged


# Load and combine data for multiple years
def build_history(years: list[int]):
    dfs = []
    if not years:
        raise ValueError("No years provided to build_history().")
    for year in years:
        dfs.append(join_year(year))
    return pd.concat(dfs, ignore_index=True)