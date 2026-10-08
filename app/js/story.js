import {
  colours, num, fmt, pct, signed, ordinal, shortTeam,
  loadTables, showLoadError, headlineStats, chancePhrase, crestScatter, gapChart, calibrationChart, threatScatter,
} from "./common.js";

/** Fill every <span data-stat="name"> with its value, so the prose always matches the data. */
function fillStats(values) {
  document.querySelectorAll("[data-stat]").forEach(element => {
    const value = values[element.dataset.stat];
    if (value === undefined) console.warn(`No value for data-stat="${element.dataset.stat}"`);
    else element.textContent = value;
  });
}

async function initialise() {
  const data = await loadTables(["teams", "profiles", "pcaSummary", "xgGap", "xgModels", "xgCalibration", "xgTeams", "xgShooters", "validation"]);
  const stats = headlineStats(data);
  const { gapAll, gapOpen, gapBlocked, models, pcaFull, pcaReduced, xgTeams, ranked, hiddenCreators } = stats;

  // Hidden creators: the biggest gaps between threat rank and assist rank make the best examples.
  const examples = [...hiddenCreators].sort((a, b) => (b.xaRank - b.xtRank) - (a.xaRank - a.xtRank));
  // The second example comes from another team, so the point doesn't rest on one club.
  const secondExample = examples.find(player => player.team !== examples[0].team) || examples[1];
  const carriers = hiddenCreators.map(player => ({ ...player, carryShare: num(player.carry_xt_per90) / num(player.total_xt_per90) }))
    .sort((a, b) => b.carryShare - a.carryShare);
  const finisher = data.xgShooters[0];
  const leader = [...data.teams].sort((a, b) => num(a.final_rank) - num(b.final_rank))[0];
  const leaderFinishing = xgTeams[leader.team];
  const shrinking = data.xgTeams.filter(team => Math.abs(num(team.goals_minus_liga_f_xg)) < Math.abs(num(team.goals_minus_statsbomb_xg))).length;
  const groupName = group => group.toLowerCase();
  const bestModel = [...data.xgModels].sort((a, b) => num(a.log_loss) - num(b.log_loss))[0];

  fillStats({
    matches: fmt(data.validation.matches_in_metadata, 0),
    shots: fmt(data.validation.total_shots, 0),
    volumeGap: pct(stats.volumeGap),
    qualityGap: pct(stats.qualityGap),
    mixMax: (Math.abs(stats.mixGaps[0].gap) * 100).toFixed(1),
    pc1: pct(pcaFull.pc1_explained_variance_ratio, 1),
    pc1Correlation: fmt(pcaFull.pc1_points_correlation),
    pc1WithoutLeader: pct(pcaReduced.pc1_explained_variance_ratio, 1),
    excludedTeam: shortTeam(pcaReduced.excluded_team),
    npGoals: fmt(gapAll.goals, 0),
    npXg: fmt(gapAll.xg, 1),
    gap: signed(gapAll.goals_minus_xg, 0),
    gapPct: pct(num(gapAll.goals_per_xg) - 1),
    gapZ: fmt(gapAll.z_score, 1),
    gapChance: chancePhrase(num(gapAll.z_score)),
    gapOpen: signed(gapOpen.goals_minus_xg, 0),
    gapBlocked: signed(gapBlocked.goals_minus_xg, 0),
    aucStatsbomb: fmt(models.statsbomb.roc_auc, 3),
    aucOurs: fmt(models.boosting.roc_auc, 3),
    predictedStatsbomb: fmt(models.statsbomb.predicted_goals, 0),
    predictedRecalibrated: fmt(models.statsbomb_recalibrated.predicted_goals, 0),
    recalibratedVerdict: bestModel.model === "statsbomb_recalibrated"
      ? " and gives the best-scoring model of the four I tested"
      : `, although the ${bestModel.label.toLowerCase()} scores best overall`,
    actualGoals: fmt(models.statsbomb.actual_goals, 0),
    leaderTeam: shortTeam(leader.team),
    leaderGapStatsbomb: signed(leaderFinishing.goals_minus_statsbomb_xg, 0),
    leaderGapOurs: signed(leaderFinishing.goals_minus_liga_f_xg, 0),
    shrinkingTeams: `${shrinking} of ${data.xgTeams.length}`,
    finisher: finisher.shooter,
    finisherTeam: shortTeam(data.profiles.find(player => player.player === finisher.shooter)?.team || ""),
    finisherGoals: fmt(finisher.goals, 0),
    finisherXg: fmt(finisher.liga_f_xg, 1),
    eligiblePlayers: fmt(stats.eligiblePlayers, 0),
    xtCorrelation: fmt(stats.xtXaCorrelation),
    hiddenCount: fmt(hiddenCreators.length, 0),
    example1: examples[0].player, example1Group: groupName(examples[0].position_group), example1Team: shortTeam(examples[0].team),
    example1Xt: ordinal(examples[0].xtRank), example1Xa: ordinal(examples[0].xaRank),
    example2: secondExample.player, example2Group: groupName(secondExample.position_group), example2Team: shortTeam(secondExample.team),
    example2Xt: ordinal(secondExample.xtRank), example2Xa: ordinal(secondExample.xaRank),
    carrier: carriers[0].player, carrierShare: pct(carriers[0].carryShare),
  });

  crestScatter("story-volume", data.teams, {
    x: "non_penalty_shots_per_match", y: "npxg_per_shot", xTitle: "Non-penalty shots per match", yTitle: "Non-penalty xG per shot", yFormat: ".3f",
    hover: { data: team => [team.final_rank], template: "Final position: %{customdata[0]}<br>Shots / match: %{x:.1f}<br>xG / shot: %{y:.3f}" },
  });
  gapChart("story-gap", [gapOpen, gapBlocked, ...data.xgGap.filter(row => row.dimension === "Body part")]);
  calibrationChart("story-calibration", data.xgCalibration, [
    { key: "statsbomb", name: "StatsBomb xG", colour: colours.red },
    { key: "statsbomb_recalibrated", name: "StatsBomb, recalibrated to Liga F", colour: colours.blue },
  ]);
  threatScatter("story-threat", ranked, hiddenCreators, "story-hidden-list");
}

initialise().catch(showLoadError);
