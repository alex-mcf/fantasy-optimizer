# Data sources and provenance

Raw CSVs and their JSON metadata sidecars are intentionally excluded from Git.
Every local season can be regenerated from the commands below.

## Mock-draft ADP: 2018–2026

The primary ADP archive comes from Fantasy Football Calculator's free
[ADP REST API](https://help.fantasyfootballcalculator.com/article/42-adp-rest-api).
The import uses one definition for every year:

- half-PPR scoring;
- 12-team leagues;
- all QB, RB, WR, and TE players; and
- average draft position from human mock drafts (computer picks are filtered by
  the provider).

The API has complete half-PPR responses for 2018 onward; it returns no half-PPR
players for 2015–2017. Each import retains the provider's player ID, number of
mock selections, high/low pick, and ADP standard deviation. The current-season
file is a market snapshot and should be refreshed during draft season. Each
refresh also preserves a timestamped copy below
`data/raw/<year>/snapshots/`, allowing future ADP movement to be modeled without
overwriting the earlier market state.

```bash
python scripts/import_ffc_adp.py 2018 2019 2020 2021 2022 2023 2024 2025 2026
```

Expected filename:

```text
data/raw/<year>/Pre_<year>_ADP(HPPR).csv
```

FantasyPros remains a useful optional comparison. Its
[Half-PPR consensus report](https://www.fantasypros.com/nfl/adp/half-point-ppr-overall.php)
exposes years back to 2015, but logged-out responses contain only five players,
so it is not the reproducible primary archive. Its underlying source mix also
changes by season.

## Completed results: 2015–2025

All results are generated from the open
[nflverse weekly player-stat releases](https://github.com/nflverse/nflverse-data/releases/tag/stats_player)
and game-level snap-count releases. The same code and scoring convention are
used for every year:

- nflverse standard fantasy points;
- plus 0.5 points per reception; and
- passing interception adjusted from nflverse's -2 to -1 to match the original
  FantasyPros historical half-PPR files.

Snap counts include active zero-point games when calculating games played.
Only QB, RB, WR, and TE rows are imported because those are the modeled roster
positions.

The same weekly file also produces one context row for every franchise and
position. Because aggregation happens before collapsing the player season, a
traded player's points are assigned to the team for which each week was played.
Context includes total room points, leader points, top-two points, contributor
count, same-position rank among NFL teams, and opportunity volume. QB opportunity
is defined as pass attempts plus carries; RB/WR/TE opportunity is carries plus
targets. A separate player-team file preserves each player's exact weekly team
contribution, allowing the forecast to remove the candidate from his own
environment signal even when he remains with the same franchise.

```bash
python scripts/import_nflverse_results.py 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
```

Expected filename:

```text
data/raw/<year>/Post_<year>_Results(HPPR).csv
```

## Player biography and draft metadata

Age, experience, rookie status, height/weight, college, and NFL draft capital
come from nflverse's maintained players-v2 release. Experience for a forecast
season is derived from `rookie_season`; the current `years_of_experience` value
is never copied backward into historical examples.

```bash
python scripts/import_nflverse_players.py
```

Expected filename:

```text
data/raw/players/players.csv
```

## Metadata and validation

Every CSV importer writes a neighboring `.meta.json` file containing the source
URL, fetch timestamp, parameters or scoring rules, positions, and row count.
Run the archive validator after any refresh:

```bash
python scripts/validate_data.py
```

The validator checks source consistency, half-PPR metadata, uniqueness, numeric
sanity, team-position completeness, available seasons, and ADP-to-results match
rates.

## What is concrete versus modeled

- Results are reconstructed observations from completed NFL games.
- ADP is observed market behavior from a finite sample of mock drafts; it is
  concrete for the provider's recorded sample, not a universal player value.
- Team-position history is observed production, but treating it as evidence for
  next season's role is a modeling assumption.
- Market-implied points use current ADP and its mock-sample uncertainty.
- Forecast points, fair ADP, VORP, edge probability, confidence, and expected
  pick value are model outputs. The forecast uses ADP as its baseline and prior
  player/role/team evidence to estimate the market's error.
