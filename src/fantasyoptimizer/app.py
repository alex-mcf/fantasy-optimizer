#Redo


















'''
# app.py (via ChatGPT)
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Fantasy Optimizer", layout="wide")
st.title("Fantasy Optimizer – Scores & Filters")

try:
    from config.league_config import league_settings
except Exception as e:
    st.error(f"Failed to import league settings: {e}")
    st.stop()

try:
    from scoring.scoring_engine import calculate_player_value
except Exception as e:
    st.error(f"Failed to import scoring engine: {e}")
    st.stop()

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def load_joined(year: int) -> pd.DataFrame:
    adp = pd.read_csv(DATA / f"{year}/Pre_{year}_ADP(HPPR).csv")
    adp = adp[["Rank","Player","Team","POS","AVG"]]
    res = pd.read_csv(DATA / f"{year}/Post_{year}_Results(HPPR).csv")
    res = res[["Rank","Player","Pos","Team","AVG","TTL"]]
    joined = pd.merge(adp, res, on="Player", how="inner", suffixes=("_pre","_post"))
    joined["Year"] = year
    return joined

def safe_load(year: int) -> pd.DataFrame:
    try:
        return load_joined(year)
    except Exception as e:
        st.warning(f"Could not load {year} data: {e}")
        return pd.DataFrame(columns=["Player","AVG_pre","TTL","Pos","Year"])

def compute_scores(years):
    frames = [safe_load(y) for y in years]
    # calculate_player_value expects three frames; pad with empties if fewer than 3 selected
    while len(frames) < 3:
        frames.append(pd.DataFrame(columns=["Player","AVG_pre","TTL"]))
    scores = calculate_player_value(frames[0].copy(), frames[1].copy(), frames[2].copy())
    all_players = pd.concat(frames, ignore_index=True)
    return scores, all_players

# ---------- Sidebar ----------
st.sidebar.header("Filters")

years = st.sidebar.multiselect("Include years", [2022, 2023, 2024], default=[2022, 2023, 2024])
if not years:
    st.info("Pick at least one year in the sidebar.")
    st.stop()

exclude_positions = set(st.sidebar.multiselect(
    "Exclude positions", ["QB","RB","WR","TE","K","DST"], default=["QB","K","DST"]
))
top_n = st.sidebar.number_input("Show top N", 5, 500, 50, 5)
ascending = st.sidebar.checkbox("Sort ascending (worst → best)?", value=False)

# ---------- Compute & show ----------
try:
    scores, all_players = compute_scores(years)
except Exception as e:
    st.exception(e)
    st.stop()

if all_players.empty:
    st.info("No rows loaded. Check your /data files (names and folders).")
    st.stop()

# Normalize positions and filter
all_players["Pos_norm"] = all_players["Pos"].fillna("").str.strip().str.upper()
valid_names = (
    all_players.loc[~all_players["Pos_norm"].isin(exclude_positions), "Player"]
    .dropna().unique()
)

rows = [{"Player": p, "Score": s} for p, s in scores.items() if p in valid_names]
scores_df = pd.DataFrame(rows)

# Attach a representative position (most frequent in selected years)
if not scores_df.empty:
    pos_map = (
        all_players[all_players["Player"].isin(scores_df["Player"])]
        .groupby(["Player","Pos_norm"]).size()
        .reset_index(name="cnt")
        .sort_values(["Player","cnt"], ascending=[True, False])
        .drop_duplicates("Player")
        .set_index("Player")["Pos_norm"]
    )
    scores_df["Pos"] = scores_df["Player"].map(pos_map)

    agg = (
        all_players.groupby("Player")
        .agg(YearsPlayed=("Year", "nunique"), MeanTTL=("TTL","mean"), LastYear=("Year","max"))
    )
    scores_df = scores_df.join(agg, on="Player")

scores_df = scores_df.sort_values("Score", ascending=ascending).reset_index(drop=True)

st.caption(
    f"League size: {league_settings.get('league_size')} | Years: {', '.join(map(str, years))} "
    f"| Excluding: {', '.join(sorted(exclude_positions)) or 'None'}"
)

st.dataframe(scores_df.head(top_n), use_container_width=True)

csv = scores_df.to_csv(index=False)
st.download_button("Download CSV", data=csv, file_name="fantasy_scores.csv", mime="text/csv")

with st.expander("Show underlying rows used for positions/years"):
    st.dataframe(all_players, use_container_width=True)
'''