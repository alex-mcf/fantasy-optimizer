# FantasyOptimizer

FantasyOptimizer compares preseason average draft position (ADP) with realized
half-PPR season results. It ranks historical draft value and helps identify the
types of players who outperformed their cost. It is currently a historical
analyzer, not a future projection model or live roster optimizer.

## Setup

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Raw data is intentionally excluded from Git. For this repository, the original
2022–2025 CSV files can be restored from project history:

```bash
./scripts/restore_historical_data.sh
```

The dashboard only offers complete seasons that have both files below:

```text
data/raw/<year>/Pre_<year>_ADP(HPPR).csv
data/raw/<year>/Post_<year>_Results(HPPR).csv
```

The restored 2025 data contains preseason ADP only, so the dashboard currently
offers the complete 2022, 2023, and 2024 seasons.

## Run

```bash
streamlit run src/fantasyoptimizer/app.py
```

## Test

The test suite uses Python's standard-library test runner:

```bash
python -m unittest discover -s tests -v
```

## How the score works

For each player-season, the pipeline calculates:

- realized fantasy points from the postseason total;
- expected points at the player's same-season, same-position ADP cost;
- value over cost (realized minus expected points);
- value over a same-season, same-position replacement player (VORP); and
- a penalty for changes in the player's points per game across selected seasons.

Historical value is weighted toward recent seasons. Each season back has half
the influence of the following season. The final score combines recency-weighted
value over cost, positive VORP, and the risk penalty. This score is a heuristic
for retrospective analysis and should be calibrated before using it for draft
decisions.

## Next phase

A live optimizer will require upcoming-season projections, league settings wired
into replacement levels and roster constraints, and an optimization algorithm.
The placeholder under `src/fantasyoptimizer/optimizer/` does not yet implement
that functionality.
