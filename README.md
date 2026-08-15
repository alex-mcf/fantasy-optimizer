# Fantasy Draft Value Forecaster

FantasyOptimizer combines a consistent half-PPR history with a transparent
upcoming-season forecast. It provides two related views:

- retrospective value: preseason mock ADP versus completed results; and
- forecast value: a market-aware fair rank versus current mock-draft ADP.

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
python scripts/import_nflverse_players.py
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

The model is deliberately small and auditable. For every target season it
builds player features from only the preceding three completed NFL seasons:
recent points per game, games played, total points, trend, available history,
age, experience, rookie/sophomore status, NFL draft capital, and position. It
also builds destination team-position features from actual
weekly production. The signals are deliberately separated into player ability
(points, points per game, and points per opportunity), role (share of the
position's opportunities), and environment (team-position volume plus teammate
points and opportunities with the candidate removed). The subtraction applies
whether the player stayed or changed teams.

ADP is treated as a strong market baseline instead of discarded. Separate
position-specific ridge models estimate the production implied by ADP, then a
heavily regularized player/role/context model predicts the market's residual
error. Separate models estimate points per game and games played so performance
and availability are visible even though direct season points remain the primary
ranking target. League-specific replacement levels convert the adjusted point
forecast into VORP and fair ADP:

```text
expected pick value = normalized market rank - fair ADP
```

A positive value means the evidence supports selecting the player earlier than
the mock market. Beat-ADP probabilities are calibrated from comparable
out-of-sample historical signals and display their sample size/confidence.
Rookies and players without usable NFL history remain explicitly labeled
low-information.

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

The residual regularization was selected while developing against these
historical seasons. The splits are chronological and player outcomes never leak
backward, but the aggregate should still be treated as development evidence;
2026 is the first untouched prospective test of the finalized specification.

The app also includes a team-position outlook. This ranks recent QB, RB, WR, and
TE environments independently of the current player name, then shows which
current candidate the model prefers for that role. The player table displays a
player-only forecast, contextual forecast, and the difference between them.

The live draft decision board accepts the current pick, next pick, drafted
players, and roster counts. It estimates whether each available player will
survive to the next selection using mock-draft variability, then labels choices
as draft now, consider now, target while waiting, wait, or pass. This is an
auditable heuristic.

Paired historical snake-draft simulations now compare ADP-only drafting, pure
model drafting, and the actionable policy. The actionable policy deliberately
keeps 75% of market rank and applies only 25% of the model adjustment. This
bounded policy is used because the simulations show that replacing ADP with the
pure model can damage complete rosters even when average rank error improves.
Simulated lineups use realized season points and position/roster constraints,
but omit waivers and weekly start/sit decisions.

Every ADP refresh now preserves an immutable timestamped copy under that
season's `snapshots/` folder. Once multiple points in the draft season have been
collected, those snapshots can train and validate a separate closing-ADP
movement model.

## Historical score

The retrospective tabs calculate realized value over same-season ADP cost,
value over a league-specific replacement player, and multi-season volatility.
Older seasons receive half the weight of the following season. This score is a
hindsight evaluation, separate from the upcoming-season forecast.
