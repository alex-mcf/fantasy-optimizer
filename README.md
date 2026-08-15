# Fantasy Draft Value Forecaster

FantasyOptimizer combines a consistent half-PPR history with a transparent
upcoming-season forecast. It provides two related views:

- retrospective value: preseason mock ADP versus completed results; and
- forecast value: an independent model rank versus current mock-draft ADP.

## Setup

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

Raw data is intentionally excluded from Git. Build the complete local archive:

```bash
python scripts/import_nflverse_results.py 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
python scripts/import_ffc_adp.py 2018 2019 2020 2021 2022 2023 2024 2025 2026
python scripts/validate_data.py
```

This produces 11 result seasons, eight complete retrospective ADP/results
seasons (2018–2025), and current 2026 mock ADP. See
[`data/SOURCES.md`](data/SOURCES.md) for exact sources, scoring, provenance, and
the concrete-data/model-output distinction.

## Run

```bash
streamlit run src/fantasyoptimizer/app.py
```

## Test

```bash
python -m unittest discover -s tests -v
```

## Forecast method

The baseline is deliberately small and auditable. For every target season it
builds player features from only the preceding three completed NFL seasons:
recent points per game, games played, total points, trend, available history,
and position. It also builds destination team-position features from actual
weekly production. The signals are deliberately separated into player ability
(points, points per game, and points per opportunity), role (share of the
position's opportunities), and environment (team-position volume plus teammate
points and opportunities with the candidate removed). The subtraction applies
whether the player stayed or changed teams. A ridge regression predicts season
points. League-specific replacement levels convert those point forecasts into
VORP and an overall model rank.

The current season's ADP is excluded from model features. It is joined after the
model rank is complete:

```text
value gap = mock ADP - model rank
```

A positive gap means the independent model would select the player earlier than
the mock market. Rookies and players without usable NFL history receive a
position-and-team-context baseline and are explicitly labeled low-information.

The dashboard reports expanding-window chronological backtests. A backtest for
season `Y` is trained only on target seasons before `Y`, avoiding random-split
and future-data leakage. It reports both point-forecast accuracy and a direct
draft-value test: players ranked at least one 12-team round ahead of normalized
market ADP are flagged as bargains, then checked against their realized
league-adjusted rank. The app compares their hit rate and average realized rank
surplus with players that were not flagged. Because late ADP picks naturally
have more room to rise, it also compares each bargain with non-bargains from the
same season and ADP round. Model-versus-market rank error and the rate at which a
model override was closer to reality are shown as stricter checks. Historical
player-level results can be downloaded for inspection.

The app also includes a team-position outlook. This ranks recent QB, RB, WR, and
TE environments independently of the current player name, then shows which
current candidate the model prefers for that role. The player table displays a
player-only forecast, contextual forecast, and the difference between them.

## Historical score

The retrospective tabs calculate realized value over same-season ADP cost,
value over a league-specific replacement player, and multi-season volatility.
Older seasons receive half the weight of the following season. This score is a
hindsight evaluation, separate from the upcoming-season forecast.
