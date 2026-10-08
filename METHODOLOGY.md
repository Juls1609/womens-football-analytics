# Methodology: Liga F chance creation

Every number on the site comes from `src/build_liga_f_metrics.py`, which also runs `src/xg_model.py` and `src/expected_threat.py`. The constants at the top of those files are the only tuning parameters. This document explains each definition and why it was chosen.

## Scope

- Competition: Liga F 2023/24 (StatsBomb open data, competition 182, season 281). All 240 matches and 16 teams.
- Team measures are season totals or rates per match. Player measures are season totals and rates per 90 minutes.

## Reconciliation checks

The pipeline writes `liga_f_validation_report.csv` and **stops** before publishing anything if any of these checks fail:

| Check | Result |
|---|---|
| Matches with events | 240 / 240 |
| Goals from events (shot goals + `Own Goal For`) vs final scores | 755 = 755 |
| Shots with exactly one chance category | 6,166 / 6,166 |
| Player minutes vs eleven players per side for every match | 99.9% (the gap is red cards) |

## Goals and expected goals

- **xG** is StatsBomb's `shot_statsbomb_xg`. **npxG** excludes penalties. Team chance-quality comparisons use npxG, because a penalty (≈0.78 xG) would otherwise inflate xG per shot for teams that won more of them.
- **Goals minus xG** compares shot goals with xG. Own goals count towards team goals but have no xG.
- Liga F scored 740 goals from shots against 652.9 xG (+87). That gap could come from finishing, goalkeeping, or how well the xG model is calibrated for this league. This analysis cannot tell these apart, so it does not interpret the gap as finishing skill.

## Chance categories

Each shot receives one label. Checks run in this order:

1. **Penalty**: `shot_type == Penalty`.
2. **Direct free kick**: `shot_type == Free Kick`.
3. **Corner**: a direct shot from a corner, or a shot within **10 seconds** of a corner taken by the shooting team in the same possession.
4. **Indirect free kick**: a shot within 10 seconds of a free-kick pass, same possession.
5. **Attacking throw-in**: a shot within 10 seconds of a throw-in taken in the attacking third (`x ≥ 80`), same possession.
6. **Transition**: StatsBomb play pattern `From Counter`.
7. **Cutback**, **Cross**, **Through ball**: StatsBomb's own `pass_cut_back`, `pass_cross` and `pass_through_ball` flags on the direct assist. Cut-backs are a subtype of cross, so they are checked first.
8. **Other open play**: everything else, including unassisted shots and second phases more than 10 seconds after a restart.

**Why a 10-second window?** StatsBomb's `play_pattern` describes how a *possession* started. A throw-in on halfway followed by fifteen passes is still "From Throw In". Labelling set pieces by play pattern would classify 53% of all shots as set pieces. The 10-second rule gives 28%, which isolates shots that actually come from the restart.

**Why StatsBomb's cut-back flag?** An earlier geometric rule (wide start, central end) labelled 343 shots as cutbacks. Of those, 300 were passes StatsBomb flags as crosses and 218 went *forward*. StatsBomb's analysts tag cut-backs directly, so their flag is used instead.

## Derived xA and minutes

- **Derived xA** is the summed non-penalty xG of shots that a player's pass set up directly. The final passer is linked through `pass_assisted_shot_id`, using both StatsBomb's shot-assist and goal-assist flags (the final pass before a goal carries only the goal-assist flag). It is an event-data derivation, **not** StatsBomb's own xA model.
- **Goal assists** counts those same passes when the shot was scored.
- **Minutes** are rebuilt from Starting XI line-ups, substitutions and red cards (straight reds and second yellows). Each stint is clipped to each half, so stoppage time counts and the half-time break does not.
- **Per-90 rankings** require at least **600 minutes**. Season totals are also shown, but they favour players who played more.

## Passing

- **Progressive pass**: a completed **open-play** pass that ends at least 10 yards closer to the centre of the opponent's goal. Goal kicks, corners, free kicks, throw-ins and kick-offs are excluded, because a long goal kick is not progression.
- **Pass into the final third**: a completed open-play pass from before `x = 80` to at or beyond `x = 80`.
- **Pass into the box**: a completed open-play pass that starts outside the penalty area and ends inside it.
- **Passing network**: each team's eleven most frequent starters across the season. A player's dot is their mean location across the passes they made and received, and lines connect pairs with at least 10 completed passes in either direction. Because the network combines the whole season, it shows the team's usual structure, not any single match.

## Defending

- **xG / shots conceded** are the opponent's shots and xG in each match.
- **High regain**: a successful ball recovery (not flagged as a recovery failure) or a successful interception (`Won`, `Success In Play`, `Success Out`) at `x ≥ 80`, i.e. the attacking third. This is a simple indicator of winning the ball high, not a full pressing model.

## Team styles: PCA and K-means

- Seven standardised team measures: npxG per match, npxG per shot, completed passes, progressive passes, passes into the final third, high regains (all per match) and xG conceded per match.
- **PCA** reduces them to two components. A component's sign is arbitrary, so the pipeline fixes it: PC1 always rises with points per match, and each component's largest loading is positive. The dashboard writes its axis descriptions from the loadings, so they stay correct after a re-run.
- **Result:** PC1 explains about 90% of the variation and correlates with points per match at r ≈ 0.95. Liga F teams differ much more in *how much* they dominate than in *how* they play.
- **K-means** (k = 3) is fitted on the same standardised measures, and groups are numbered by average points. The silhouette score (≈0.41) shows the groups are only moderately distinct, so they are presented as performance levels, not tactical identities.
- **Sensitivity check**: the PCA and K-means are refitted without Barcelona. PC1 still explains about 87%, so Barcelona amplifies the pattern but does not create it.

## The xG gap and a Liga F xG model (`src/xg_model.py`)

All of this uses non-penalty shots only (6,114).

**Is the gap real?** If StatsBomb's xG were correct, each shot would be a coin flip with probability *p*, so season goals would vary around Σp with variance Σp(1−p). The observed gap (701 goals vs 612.2 xG, +89) is **4.2 standard deviations** above that. The same test is run for each shot context (distance, body part, chance type, and whether a defender stood between ball and goal in the freeze frame).

**Where is it?** Almost entirely on shots with no defender in the shooting lane (+86 goals, z = 5.2). With at least one defender in the way it's +3 (z = 0.2). Headers match their xG (+6, z = 0.6); shots with the foot do not (+83).

**Models compared**, all evaluated out of sample with 5-fold cross-validation **grouped by match**, so no match appears in both training and test data:

| Model | What it is |
|---|---|
| StatsBomb xG | As supplied. |
| StatsBomb, recalibrated | A logistic regression on logit(StatsBomb xG), fitted on training folds. It keeps StatsBomb's ranking of chances but re-fits the level and slope to Liga F. |
| Liga F logistic regression | Standardised features, L2-regularised. Also used to report feature effects. |
| Liga F gradient boosting | `HistGradientBoostingClassifier`, depth 3, 300 iterations, learning rate 0.05, at least 40 shots per leaf. |

**Features:** distance and goal angle; header / other body part; direct free kick; first-time, one-on-one, open goal and under pressure flags; volley/half-volley and lob; assist type (cross, cutback, through ball), set-piece and counter-attack origin; and from the freeze frame, the number of defenders inside the triangle between ball and posts, opponents within 2 yards, the goalkeeper's distance from goal centre, and whether the goalkeeper is inside the triangle. StatsBomb's `shot_deflected` flag is excluded because it describes what happened after the shot.

**Result:** StatsBomb ranks chances best (AUC 0.816 vs 0.810 for gradient boosting), which is unsurprising: about 6,000 shots from one league is a small training set for an xG model. StatsBomb's weakness is calibration (612 predicted vs 701 actual). Recalibrating it gives the lowest log loss of the four models. Team, shooter and goalkeeper tables compare goals with the out-of-fold Liga F gradient-boosting xG.

**Caveat:** there is no post-shot xG in the open data, so the analysis can't separate finishing from goalkeeping.

## Expected threat (`src/expected_threat.py`)

Karun Singh's formulation on a 16 × 12 grid, fitted on this season's open-play passes, carries and open-play shots:

`xT(cell) = P(shoot) · P(goal | shot) + P(move) · Σ T(cell → target) · xT(target)`

It is solved by 50 iterations. Failed passes count as move attempts but add no transitions. A completed open-play pass or carry earns `xT(end) − xT(start)`, and players are ranked per 90 minutes (600-minute minimum).

**Hidden creators** are players in the top 25 for xT per 90 but outside the top 40 for derived xA per 90.

**Caveat:** xT rewards possession volume, so players on dominant teams have more opportunities to add it.

## Player similarity

Twelve per-90 measures for outfield players with at least 600 minutes: non-penalty xG, shots, derived xA, created shots, xT from passes, xT from carries, progressive passes, passes into the box, completed passes, successful dribbles, ball regains and pressures. Each is standardised across all eligible players, and similarity is the cosine similarity of those vectors. Position groups (defender, midfielder, forward) come from the position each player was most often recorded in. Percentiles in the comparison chart are computed within the position group.

Similarity describes statistical output, which partly reflects team context, not quality or playing style.

## Known limitations

- All results are associations, not causal effects.
- The final table is rebuilt from match scores and breaks ties on goal difference rather than Liga F's head-to-head rule. In 2023/24 no two teams finished level on points, so this does not change the order. Points deductions, if any, would not be reflected.
- The chance-category rules are transparent first-pass definitions. The 10-second window and the attacking-third throw-in threshold are judgement calls, and both are constants in the pipeline.
