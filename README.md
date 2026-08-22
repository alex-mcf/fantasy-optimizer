# Fantasy Draft Value Forecaster

FantasyOptimizer is a draft-day board: a filterable player list that prices the
upcoming season against current mock-draft ADP, plus the tools you use at a pick.
ADP is the baseline, not the enemy — the model's job is to say where that
baseline is wrong, and to show you the evidence for each call.

The app has three tabs:

- **Board** — every modeled player, filtered by position, call strength,
  confidence, team, or the rounds they are going in. Columns start at the eight
  a draft needs and expand on request; selecting a player explains the call.
- **Draft room** — your pick, what is gone, what your roster still needs, and a
  shortlist that accounts for who will survive to your next pick.
- **Evidence** — the backtests, the draft simulation, and the measured limits of
  the model, kept behind the board rather than in front of it.

## Setup

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Raw data is intentionally excluded from Git. Build the complete local archive:

```bash
python scripts/import_nflverse_results.py 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
python scripts/import_ffc_adp.py 2018 2019 2020 2021 2022 2023 2024 2025 2026
python scripts/import_nflverse_players.py
python scripts/import_nflverse_roles.py 2026
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

The tests import `fantasyoptimizer`, so the editable install above must have
taken effect. On macOS with Python 3.13 or newer, an editable install inside a
directory carrying the `hidden` file flag is silently ignored: Python skips
hidden `.pth` files, so `import fantasyoptimizer` fails even though `pip` reports
success. Clear the flag once and the install works:

```bash
chflags -R nohidden .venv
```

## Forecast method

The model is deliberately small and auditable. For every target season it
builds player features from only the preceding three completed NFL seasons:
recent points per game, games played, total points, trend, available history,
age, experience, rookie/sophomore status, NFL draft capital, and position.

It also builds destination team-position features from actual weekly
production, separated into role (share of the position's opportunities) and
environment (team-position volume plus teammate points and opportunities with
the candidate removed, whether the player stayed or changed teams). These are
**not** model inputs. They are displayed for judgement, they drive the
team-position outlook, and a context-augmented model is scored in every backtest
window so the decision to leave them out stays measured — see below.

ADP is treated as a strong market baseline instead of discarded. Separate
position-specific ridge models estimate the production implied by ADP. That
market layer also measures each player's ADP rank and distance from the ADP
leader among current same-team, same-position candidates, providing a
historically reconstructible role-competition signal. Features built from how
many mock drafts an ADP row is based on are standardized within their own season,
because the raw counts collapsed over time — a median row went from 365 drafts in
2023 to 33 in 2025 — and a model trained on the old scale would otherwise be
asked to predict from values it never saw.

A deliberately narrow model then predicts the market's residual error: eleven
player-form features, no context features, no position dummies (each residual
model is already fitted inside one position). Its ridge penalty is not a constant
but is fitted per position by holding out whole seasons inside the training
window, and the search range extends far enough to switch the residual layer off
where it earns nothing. Both choices exist for the same reason: TE has roughly
130 training rows and QB roughly 160, and a wide, weakly penalized fit on that
many rows produces noise rather than insight. Separate models estimate points per
game and games played so performance and availability are visible even though
direct season points remain the primary ranking target. League-specific
replacement levels convert the adjusted point forecast into VORP and fair ADP:

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

The board also carries a hard limit: no player may be promoted more than a share
of his own market rank — 35% of it, with a one-round floor, so a player going at
pick 150 can move about 52 picks and one going at pick 30 can move 12. Ridge is linear, so it cannot express "efficient,
but on almost no volume", and nothing else bounded how far a prediction could
travel from the market — in 2024 it put a tight end with the position's best
points per opportunity, earned on 40 of them, at rank 1 against a market rank of
158. He finished 64th, and that one row was over half of that season's remaining
deficit. Shrinking the rate features toward their position prior was the obvious
fix and does not work: the model standardizes its features, so a monotone rescale
is undone, and it cost about five lineup points a season for the complexity. The
limit does work and is a no-op for ordinary bargains. It began as a flat three
rounds, which was the wrong shape in two ways. It was too loose where the model
is weakest — a promotion from round four to round two fits inside any flat limit,
and early-round quarterbacks are its worst category. And because every pinned
player was pushed to exactly the same distance from the market, their order
became market order: the model's opinion about which late tight end was best, its
single strongest segment at 89–95%, was erased.

Scaling the allowance with market rank fixes both. Early promotions are held
tighter, late ones can travel further, and because each player's limit differs
their relative order survives — on the 2026 board the pinned players now keep
their model ordering almost exactly (rank correlation 0.997, against market order
before). It is worth about 14 lineup points a season over the flat version, and
any share between 0.2 and 0.4 performs about the same.

It binds almost entirely on tight ends, and that is not a mistake being corrected.
The model promotes tight ends by 28 ranks on average and they realize a +35 rank
surplus: the tilt is right. What rank surplus cannot see is that a roster starts
one tight end, so being right about the fifth-best one buys very little. The guard
does not change how many tight ends a simulated draft ends up with — roster limits
already cap that — it changes when they are taken. In 2024 it moved the average
tight end from round 2.2 to round 3.8 and gained 44 lineup points. This is the
same lesson as the blend weight: rank metrics and roster outcomes are different
objectives, and only one of them is the point.

Alongside point error and rank error, the backtest tracks how far the model
moves the first 60 players off their market rank. Whole-pool error averages over
roughly 150 players, so a model can grow noisy exactly where a draft is decided
and still look flat on the headline metrics; this project learned that the
expensive way, and the drift column exists so it cannot happen quietly again.

The splits are chronological and player outcomes never leak backward. The blend
weight and the residual penalty are both fitted inside the window rather than
chosen by hand, but the model's shape — which features exist at all — was still
decided while looking at these seasons, so the aggregate remains development
evidence; 2026 is the first untouched prospective test of the finalized
specification.

### What the backtests actually support

The honest reading of the current numbers, stated plainly so the framing above is
not mistaken for a stronger claim:

- **Point accuracy is not the edge.** Against the market baseline the model's
  point MAE is close to flat. On raw season projections, ADP alone is competitive
  with this model.
- **Rank ordering is the edge, and it is modest.** Model rank MAE beats market
  rank MAE by about 1.9 ranks overall, and a model override is closer to reality
  than the market about 64% of the time across the flagged bargains. The
  promotion guard costs a little of that pooled rank accuracy and buys better
  drafts, which is the trade this project is explicitly making.
- **The context layer does not pay for itself.** Team, role, and environment
  features are no longer part of the trained model. A context-augmented model is
  still fitted and scored in every backtest window, and its rank MAE improvement
  over the shipped model averages −0.04 — it is not helping. The features remain
  in the pipeline for the team-position outlook and to keep that comparison
  running; if it ever turns positive, the column will say so.
- **The residual layer used to be actively harmful.** With one fixed penalty and
  49 features, drafting straight off the model board lost an average of 57 lineup
  points per season against ADP, and 2023–25 were far worse than that. Fitting the
  penalty per position, cutting the residual layer to eleven features, and bounding
  promotions turns that into +55, and the model's average drift from the market
  across the first 60 picks falls from 10.9 ranks to 8.2. The blended policy that
  ships improved much less (+72 to +77 lineup points), because a bounded tilt was
  already absorbing most of the damage; what the fixes bought is a model that is no
  longer carried by its damper.
- **The penalty floor is doing work.** Candidate penalties start at 300, and WR
  selects that floor every season. Adding lower rungs lets WR fall to 30, which
  costs roughly 14 lineup points a season and raises top-of-board drift — the
  selection criterion is whole-pool error, which cannot see that damage, so the
  floor is where that judgement lives.
- **The 2022 season is thin.** Its ADP file contains only 117 modeled players,
  against 144–191 in every other season. This is the upstream source, not a
  matching failure: 2022 joins to results at 100%, the best rate of any season.
  Results driven by 2022 still deserve less weight, since the pool is two rounds
  shallower than the seasons around it.

The app also includes a team-position outlook. This ranks recent QB, RB, WR, and
TE environments independently of the current player name, then shows which
current candidate the model prefers for that role. It is descriptive: those
context features are measured every backtest but are not inputs to the shipped
forecast.

An optional official-role import adds the latest depth rank, starter flag, and
weekly roster status to the current forecast and live draft board. The importer
selects the last dated depth-chart snapshot no later than the ADP sample's end
date. These fields are explanatory only: the upstream source changed after 2024,
and older files lack timestamps that prove alignment with historical ADP, so the
forecast does not train on them yet.

The draft room accepts the current pick, next pick, drafted players, and roster
counts. It estimates whether each available player will survive to the next
selection using mock-draft variability, then labels choices as draft now,
consider now, target while waiting, wait, or pass. Players marked drafted there
also drop off the board.

Its shortlist ordering is fitted rather than assumed. `fit_decision_weights.py`
scores candidate weightings on simulated drafts against the honest null
hypothesis — just take the best player available on the blended board — and the
original hand-chosen weights lost to that baseline by 20 points a season, and by
135 in the worst one. The fitted weights beat it by 31 and are positive in five
of six seasons, including out-of-sample when fitted only on earlier seasons.

Two things came out of that fit. Roster construction dominates marginal value: an
unfilled starting slot is worth roughly a starter's entire VORP, so that term is
an order of magnitude larger than it was. And the value-gap term shrank, because
the board rank already blends the gap in and counting it twice over-tilted the
shortlist toward the model.

```bash
python scripts/fit_decision_weights.py
```

The draft room also shows the **cost of waiting** at each position: the drop from
the best player available now to the best one likely to still be there at your
next pick. If the edge is positional, this is the question the tool exists to
answer, and it is the number that should drive a pick. It is displayed rather
than sorted on — as an ordering rule it was measured against the fitted decision
score and lost, matching it on average while being worse in the worst season.

## Freezing a board

```bash
python scripts/freeze_board.py
```

Everything here refits as seasons arrive, which is right for a forecaster and
wrong for an evaluation: by January the board will not be the board you drafted
from. This writes the current one to `data/boards/` with the settings that
produced it — fitted weights, guard parameters, residual features, ADP
provenance, and the positional allocation — so the first untouched test of the
model is graded against what it actually said.

The sidebar controls league size, starting roster, superflex, draft slot, and
total snake-draft rounds. Snake selections are calculated from your slot, with
manual pick-number overrides for keepers or traded picks. The supported scoring
definition remains half-PPR; changing roster settings changes replacement value,
not the underlying historical scoring data.

Fantasy Football Calculator remains the consistent market used to train and
backtest the predictor. For ESPN, Yahoo, Sleeper, NFL.com, or another draft room,
the user can upload a current CSV containing a player/name column and ADP/rank
column. The selected platform then controls live cost, platform value gap,
actionable ADP, and next-pick availability. Modeled players the upload does not
cover are visibly marked as FFC fallbacks, and uploaded rows that match no
modeled player are counted and listed rather than silently dropped.

Paired historical snake-draft simulations now compare ADP-only drafting, pure
model drafting, and the actionable policy. The actionable policy keeps most of
market rank and applies a bounded model tilt, because the simulations show that
replacing ADP with the pure model damages complete rosters even when average rank
error improves. Simulated lineups use realized season points and position/roster
constraints, but omit waivers and weekly start/sit decisions.

That tilt is no longer a hand-picked constant. Candidate weights are scored on
simulated draft outcomes, and every backtested season is scored with a weight
fitted only on strictly earlier seasons, so no reported season benefits from a
weight chosen on its own results. Three findings came out of making that nesting
explicit:

- The weight must be fitted on **draft outcomes**, not on rank error. Rank error
  keeps improving as the model takes over the board — it selects a pure-model
  board — while drafts run off that same board lose badly. Selecting on rank
  error turns the policy's measured lift from roughly +79 points into roughly
  −15.
- Taking the best-scoring weight overfits the selection seasons. Bigger tilts
  carry more outcome variance, so a lucky season buys a weight that then loses:
  under a plain argmax the fitter tilted to 0.75 for 2024 and gave back 32 lineup
  points that season. Candidates within one standard error of the best — measured
  on the paired, same-draw differences — are now treated as tied, and the most
  conservative one wins.
- Fitted this way the nested weights sit between 0.2 and 0.5, and the policy beats
  ADP by roughly +77 lineup points at a 64% win rate. The weight responds to model
  quality as it should: with the promotion guard bounding the model's worst calls,
  the fitter leans on it harder than it did before. It is also league-dependent —
  in a three-starting-RB league the same fitter drops the weight to zero, because
  there the model's tilt stops helping at all.

Importers write every CSV and metadata sidecar to a temporary file in the target
directory and rename it into place, so an interrupted download cannot leave a
truncated file where a clean archive season used to be.

Every ADP refresh now preserves an immutable timestamped copy under that
season's `snapshots/` folder. Once multiple points in the draft season have been
collected, those snapshots can train and validate a separate closing-ADP
movement model. Official-role refreshes preserve timestamped copies in the same
folder so depth-chart changes can be audited without overwriting prior states.

## What the edge actually is

The board's advantage over ADP decomposes into two very different things, and the
split matters more than the headline number:

- **The positional call carries it.** The model takes more quarterbacks and tight
  ends, and fewer receivers, than the market does in the early rounds. On the
  2026 board that is 8 QB and 6 TE in the first five rounds against the market's
  4 and 2. Isolated, that call is worth roughly +100 lineup points a season.
- **Within a position, the model has no measurable ordering skill.** Its rank
  correlation with realized points is 0.425 against the market's 0.428, the gap
  is under 0.1 ranks at every depth from the top three to the whole pool, and it
  identifies the best player at a position exactly as often as ADP (4 of 24
  position-seasons).

The obvious conclusion — take ADP's ordering inside each position — is wrong, and
measurably so. Drafting the model's positional plan with ADP's ordering is worth
+14 lineup points a season; drafting it with the model's own ordering is worth
+64. Blending toward the market degrades it monotonically.

The reason is not accuracy, it is correlation. Opponents draft by ADP, so a board
that shares ADP's ordering wants the same players at the same moments and pays
market price for them. An ordering that is equally accurate but *different* buys
equivalent players later: across the backtest, drafting off the model's ordering
acquires players at an average ADP of 57.8 against 55.5, and scores 1353 lineup
points against 1306. Being differently right is worth more than being more right.

The practical reading: treat a screen of values as **one correlated bet about
positions**, not as a list of independent player insights, and do not assume the
model knows which tight end is best — only that it wants a tight end.

## Using the board against a mock ADP

The dashboard's draft simulation answers "should I draft off this board?". The
narrower question in front of a mock ADP list — "the model disagrees about this
player, should I act on it?" — has its own evaluation:

```bash
python scripts/evaluate_adp_comparison.py
```

It reports how disagreements of each size have performed, where on the board and
at which position they pay, and whether they survive the market moving. Three
results shape how the board should be read:

- **Two rounds is the threshold.** Players the model moves up a single round beat
  their ADP 48% of the time against a 44% base rate — inside the noise. Two-round
  calls hit 61%, three-round 80%, and four-plus 92% with a mean realized surplus
  of +60 ranks. Both the board's headline count and the draft room's "value"
  label use two rounds for this reason.
- **Position and stage matter as much as size.** Tight-end bargains after round 7
  beat their ADP in 89–95% of cases. Quarterbacks the model promotes into the
  first three rounds are its worst category by a wide margin. A disagreement is
  not a single kind of claim.
- **The advice is mostly stable across markets.** Re-pricing the board on a
  different plausible mock market keeps about 80% of the flagged bargains, and
  they still beat their ADP roughly 70–77% of the time. Calls sitting near the
  threshold are the ones that move, which is another argument for acting on
  two-round gaps rather than one-round gaps.

## Historical score

`fantasyoptimizer.scoring` still calculates realized value over same-season ADP
cost, value over a league-specific replacement player, and multi-season
volatility, with each older season weighted half as much as the one after it. It
is a hindsight evaluation rather than a draft-day tool, so it no longer has a
place in the app; import it directly if you want that view.
