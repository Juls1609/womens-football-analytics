import {
  CATEGORIES, SET_PIECE_CATEGORIES, DISTANCE_ORDER, MINIMUM_MINUTES, colours, plotConfig,
  num, fmt, pct, signed, mean, shortTeam, shortPlayer, escapeHtml, isTrue, ordinal, teamBadge,
  loadTables, showLoadError, headlineStats, chancePhrase, plotLayout, crestScatter, gapChart, calibrationChart,
  threatScatter, threatGridChart, pitchLayout,
} from "./common.js";

const clusterColours = { 1: colours.gold, 2: colours.blue, 3: colours.cyan };

// Must match PROFILE_FEATURES in src/build_liga_f_metrics.py.
const PROFILE_FEATURES = {
  npxg_per90: "Non-penalty xG", shots_per90: "Shots", derived_xa_per90: "Derived xA", shot_assists_per90: "Created shots",
  pass_xt_per90: "xT from passes", carry_xt_per90: "xT from carries", progressive_passes_per90: "Progressive passes",
  passes_into_box_per90: "Passes into the box", passes_completed_per90: "Completed passes", dribbles_per90: "Successful dribbles",
  regains_per90: "Ball regains", pressures_per90: "Pressures",
};

const state = { rendered: new Set(), pcaModel: "all_teams", creatorMode: "xa90", similarity: {} };
const playerKey = player => `${player.team}|${player.player}`;

// ---------- Overview ----------
function renderOverview() {
  const { teams, validation, stats } = state;
  const matches = validation.matches_in_metadata;
  document.querySelector("#matches-kpi").textContent = fmt(matches, 0);
  document.querySelector("#shots-kpi").textContent = fmt(validation.total_shots, 0);
  document.querySelector("#shots-kpi-note").textContent = `${fmt(validation.total_shots / matches, 1)} per match`;
  document.querySelector("#xg-kpi").textContent = fmt(validation.total_npxg, 1);
  document.querySelector("#xg-kpi-note").textContent = `${fmt(validation.total_xg, 1)} including penalties`;
  document.querySelector("#goals-kpi").textContent = fmt(validation.goals_from_events, 0);
  document.querySelector("#goals-kpi-note").textContent = `${signed(validation.shot_goals - validation.total_xg, 0)} goals from shots vs xG`;

  crestScatter("team-scatter", teams, {
    x: "non_penalty_shots_per_match", y: "npxg_per_shot", xTitle: "Non-penalty shots per match", yTitle: "Non-penalty xG per shot", yFormat: ".3f",
    hover: { data: team => [team.final_rank, num(team.npxg_per_match)], template: "Final position: %{customdata[0]}<br>Shots / match: %{x:.1f}<br>xG / shot: %{y:.3f}<br>npxG / match: %{customdata[1]:.2f}" },
  });

  const { volumeGap, qualityGap, rVolume, rQuality, mixGaps } = stats;
  const largest = mixGaps[0];
  const both = volumeGap > 0 && qualityGap > 0;
  const similarMix = Math.abs(largest.gap) < 0.05;
  const verdict = both
    ? (similarMix ? "mostly <em>more</em> and <em>better</em> chances, not different ones." : "more and better chances, with a somewhat different mix.")
    : "the picture is mixed.";
  const lead = both
    ? `The top four took ${pct(volumeGap)} more shots per match than the other twelve teams, and each shot was worth ${pct(qualityGap)} more xG.`
    : `The top four differed from the rest by ${signed(volumeGap * 100, 0)}% in shot volume and ${signed(qualityGap * 100, 0)}% in xG per shot.`;
  const mix = similarMix
    ? `Their chance mix was similar: no category differs by more than ${(Math.abs(largest.gap) * 100).toFixed(1)} percentage points (largest: ${largest.category.toLowerCase()}).`
    : `The clearest difference in chance mix is ${largest.category.toLowerCase()} (${signed(largest.gap * 100, 1)} percentage points of their shots).`;
  document.querySelector("#research-answer").innerHTML =
    `<strong>Answer:</strong> ${verdict} ${lead} Across all 16 teams, points per match correlate with shot volume at r = ${rVolume.toFixed(2)} and with chance quality at r = ${rQuality.toFixed(2)}. ${mix}`;
}

// ---------- Attacking ----------
function renderProfile(teamName) {
  const { teams } = state;
  const team = teams.find(item => item.team === teamName);
  const leagueAverage = CATEGORIES.map(category => mean(teams, `${category}_shot_share`));
  const teamValues = CATEGORIES.map(category => num(team[`${category}_shot_share`]));
  Plotly.react("profile-chart", [
    { type: "bar", x: CATEGORIES, y: teamValues, name: shortTeam(team.team), marker: { color: CATEGORIES.map(category => SET_PIECE_CATEGORIES.includes(category) ? colours.cyan : colours.blue) }, hovertemplate: "%{x}<br>%{y:.1%} of shots<extra></extra>" },
    { type: "scatter", x: CATEGORIES, y: leagueAverage, name: "League average", mode: "markers", marker: { color: colours.gold, size: 11, symbol: "line-ew-open", line: { width: 3, color: colours.gold } }, hovertemplate: "League average<br>%{y:.1%} of shots<extra></extra>" },
  ], plotLayout({ margin: { l: 56, r: 16, t: 64, b: 112 }, yaxis: { title: { text: "Share of shots" }, tickformat: ".0%", gridcolor: colours.grid, rangemode: "tozero" }, xaxis: { tickangle: -35 }, legend: { orientation: "h", x: 0, y: 1.2, yanchor: "bottom" }, bargap: 0.35,
    annotations: [{ x: 2, y: 1.02, xref: "x", yref: "paper", yanchor: "bottom", text: "Set pieces", showarrow: false, font: { size: 10, color: colours.cyan } }, { x: 7, y: 1.02, xref: "x", yref: "paper", yanchor: "bottom", text: "Open play", showarrow: false, font: { size: 10, color: colours.blue } }] }), plotConfig);

  const setPieceShare = SET_PIECE_CATEGORIES.reduce((sum, category) => sum + num(team[`${category}_shot_share`]), 0);
  const finishing = state.stats.xgTeams[team.team];
  const stats = [
    ["Final position", `${ordinal(team.final_rank)} · ${fmt(team.points, 0)} pts`],
    ["npxG per match", fmt(team.npxg_per_match)],
    ["Shots per match", fmt(team.shots_per_match, 1)],
    ["npxG per shot", fmt(team.npxg_per_shot, 3)],
    ["Set-piece share", pct(setPieceShare)],
    ["Goals vs Liga F xG", finishing ? signed(finishing.goals_minus_liga_f_xg) : "—"],
    ["Top two creators' share of xA", pct(team.top_two_creator_xa_share)],
  ];
  document.querySelector("#team-summary").innerHTML = `<div class="team-summary__name">${teamBadge(team.team, "large")}<strong>${escapeHtml(team.team)}</strong></div><dl>${stats.map(([label, value]) => `<div><dt>${label}</dt><dd>${value}</dd></div>`).join("")}</dl>`;
}

function renderShotMap() {
  const team = document.querySelector("#shot-team-select").value;
  const category = document.querySelector("#shot-category-select").value;
  const filtered = state.shots.filter(shot => (team === "All" || shot.team === team) && (category === "All" || shot.chance_category === category));
  const goals = filtered.filter(shot => isTrue(shot.is_goal_from_shot));
  const misses = filtered.filter(shot => !isTrue(shot.is_goal_from_shot));
  const trace = (items, name, colour, opacity) => ({
    x: items.map(shot => num(shot.shot_x)), y: items.map(shot => num(shot.shot_y)), name, type: "scattergl", mode: "markers",
    text: items.map(shot => `<b>${escapeHtml(shot.shooter)}</b><br>${escapeHtml(shot.team)} · ${shot.minute}'<br>${shot.chance_category}<br>xG ${fmt(shot.shot_statsbomb_xg)}`),
    marker: { size: items.map(shot => 4 + Math.sqrt(num(shot.shot_statsbomb_xg)) * 20), color: colour, opacity, line: { color: "#fff", width: 0.5 } },
    hovertemplate: "%{text}<extra></extra>",
  });
  Plotly.react("shot-map", [trace(misses, "No goal", colours.blue, 0.45), trace(goals, "Goal", colours.gold, 0.9)],
    pitchLayout({ xaxis: { range: [58, 122], visible: false, fixedrange: true }, legend: { orientation: "h", x: 0, y: 1.06 }, showlegend: true, margin: { l: 10, r: 10, t: 36, b: 10 } }), plotConfig);
  const xg = filtered.reduce((sum, shot) => sum + num(shot.shot_statsbomb_xg), 0);
  document.querySelector("#shot-summary").textContent = filtered.length
    ? `${fmt(filtered.length, 0)} shots · ${fmt(goals.length, 0)} goals · ${fmt(xg, 1)} xG (${fmt(xg / filtered.length, 3)} per shot)`
    : "No shots match this selection.";
}

// ---------- Finishing ----------
function renderFinishing() {
  const { stats, xgGap, xgCalibration, xgModels, xgTeams, xgShooters, xgGoalkeepers, xgFeatures } = state;
  const { gapAll, gapOpen, gapBlocked, models } = stats;
  document.querySelector("#finishing-summary").innerHTML =
    `Liga F scored <strong>${fmt(gapAll.goals, 0)}</strong> non-penalty goals from <strong>${fmt(gapAll.xg, 1)}</strong> StatsBomb xG, which is ${signed(gapAll.goals_minus_xg, 0)} goals, or ${pct(num(gapAll.goals_per_xg) - 1)} more than expected. If the xG values were right, a gap that large ${chancePhrase(num(gapAll.z_score))} (z = ${fmt(gapAll.z_score, 1)}). Almost all of it comes from shots with a clear sight of goal: ${signed(gapOpen.goals_minus_xg, 0)} goals when no defender stood between ball and goal, against ${signed(gapBlocked.goals_minus_xg, 0)} when one did.`;

  const renderGap = () => {
    const dimension = document.querySelector("#gap-dimension").value;
    let rows = xgGap.filter(row => row.dimension === dimension);
    if (dimension === "Distance") rows = DISTANCE_ORDER.map(label => rows.find(row => row.group === label)).filter(Boolean);
    else rows = rows.sort((a, b) => num(b.goals_per_xg) - num(a.goals_per_xg));
    gapChart("gap-chart", rows);
  };
  document.querySelector("#gap-dimension").onchange = renderGap;
  renderGap();

  calibrationChart("calibration-chart", xgCalibration, [
    { key: "statsbomb", name: "StatsBomb xG", colour: colours.red },
    { key: "statsbomb_recalibrated", name: "StatsBomb, recalibrated to Liga F", colour: colours.blue },
    { key: "boosting", name: "Liga F model", colour: colours.gold, dash: "dash" },
  ]);

  const best = [...xgModels].sort((a, b) => num(a.log_loss) - num(b.log_loss))[0];
  document.querySelector("#model-table").innerHTML = xgModels.map(row => `<tr${row.model === best.model ? ' class="is-best"' : ""}><td>${escapeHtml(row.label)}</td><td>${fmt(row.roc_auc, 3)}</td><td>${fmt(row.log_loss, 4)}</td><td>${pct(row.log_loss_skill, 1)}</td><td>${fmt(row.predicted_goals, 0)} / ${fmt(row.actual_goals, 0)}</td></tr>`).join("");
  const statsbomb = models.statsbomb, ours = models.boosting, recalibrated = models.statsbomb_recalibrated;
  document.querySelector("#model-verdict").innerHTML = num(statsbomb.roc_auc) >= num(ours.roc_auc)
    ? `<strong>Verdict:</strong> StatsBomb's model still ranks chances best (AUC ${fmt(statsbomb.roc_auc, 3)} against ${fmt(ours.roc_auc, 3)} for a model trained on one Liga F season), so it knows which shots are better. Its problem is the <em>level</em>: it predicts ${fmt(statsbomb.predicted_goals, 0)} goals where ${fmt(statsbomb.actual_goals, 0)} were scored. Re-fitting just its level and slope to Liga F gives ${fmt(recalibrated.predicted_goals, 0)}${best.model === "statsbomb_recalibrated" ? " and the best score of all four models" : `, although ${best.label.toLowerCase()} scores best overall`}.`
    : `<strong>Verdict:</strong> the Liga F model ranks chances better than StatsBomb's (AUC ${fmt(ours.roc_auc, 3)} vs ${fmt(statsbomb.roc_auc, 3)}).`;

  const features = [...xgFeatures].sort((a, b) => Math.abs(num(a.coefficient)) - Math.abs(num(b.coefficient)));
  Plotly.react("feature-chart", [{
    type: "bar", orientation: "h", y: features.map(row => row.label), x: features.map(row => num(row.coefficient)),
    marker: { color: features.map(row => num(row.coefficient) >= 0 ? colours.blue : colours.red) },
    hovertemplate: "%{y}<br>Effect: %{x:+.2f}<extra></extra>",
  }], plotLayout({ margin: { l: 16, r: 24, t: 16, b: 52 }, xaxis: { title: { text: "Standardised effect on the log-odds of scoring" }, gridcolor: colours.grid, zerolinecolor: colours.ink }, yaxis: { automargin: true } }), plotConfig);

  const teams = [...xgTeams].sort((a, b) => num(a.goals_minus_statsbomb_xg) - num(b.goals_minus_statsbomb_xg));
  Plotly.react("team-finishing-chart", [
    { x: teams.map(team => num(team.goals_minus_statsbomb_xg)), y: teams.map(team => shortTeam(team.team)), name: "vs StatsBomb xG", mode: "markers", marker: { color: colours.red, size: 11 }, hovertemplate: "%{y}: %{x:+.1f} goals vs StatsBomb xG<extra></extra>" },
    { x: teams.map(team => num(team.goals_minus_liga_f_xg)), y: teams.map(team => shortTeam(team.team)), name: "vs Liga F xG", mode: "markers", marker: { color: colours.blue, size: 11, symbol: "diamond" }, hovertemplate: "%{y}: %{x:+.1f} goals vs Liga F xG<extra></extra>" },
  ], plotLayout({
    margin: { l: 16, r: 24, t: 40, b: 52 }, xaxis: { title: { text: "Goals minus expected goals" }, gridcolor: colours.grid, zerolinecolor: colours.ink, zerolinewidth: 1.5 }, yaxis: { automargin: true },
    legend: { orientation: "h", x: 0, y: 1.08 },
    shapes: teams.map(team => ({ type: "line", x0: num(team.goals_minus_statsbomb_xg), x1: num(team.goals_minus_liga_f_xg), y0: shortTeam(team.team), y1: shortTeam(team.team), line: { color: colours.soft, width: 2 }, layer: "below" })),
  }), plotConfig);

  const row = (item, name) => `<tr><td>${escapeHtml(name)}</td><td>${fmt(item.shots, 0)}</td><td>${fmt(item.goals, 0)}</td><td>${fmt(item.liga_f_xg, 1)}</td><td>${signed(item.goals_minus_liga_f_xg)}</td></tr>`;
  document.querySelector("#shooter-table").innerHTML = xgShooters.slice(0, 10).map(item => row(item, item.shooter)).join("");
  const keepers = [...xgGoalkeepers].sort((a, b) => num(a.goals_minus_liga_f_xg) - num(b.goals_minus_liga_f_xg));
  document.querySelector("#keeper-table").innerHTML = keepers.slice(0, 10).map(item => row(item, item.goalkeeper)).join("");
}

// ---------- Styles ----------
function describeComponent(rows, positiveWord, negativeWord) {
  const strong = rows.filter(row => Math.abs(num(row.loading)) >= 0.25);
  const phrase = row => row.feature_label.replace(/^[A-Z](?=[a-z])/, letter => letter.toLowerCase());
  const up = strong.filter(row => num(row.loading) > 0).map(phrase);
  const down = strong.filter(row => num(row.loading) < 0).map(phrase);
  const parts = [];
  if (up.length) parts.push(`<strong>more</strong> ${up.join(", ")}`);
  if (down.length) parts.push(`<strong>less</strong> ${down.join(", ")}`);
  return `Moving ${positiveWord}: ${parts.join("; ")}. Moving ${negativeWord}: the opposite.`;
}

function renderStyles() {
  const model = state.pcaModel;
  const sensitivity = model !== "all_teams";
  const styles = sensitivity ? state.stylesSensitivity : state.styles;
  const loadings = sensitivity ? state.loadingsSensitivity : state.loadings;
  const summary = state.pcaSummary.find(row => row.model === model);
  const { pcaFull: full, pcaReduced: reduced } = state.stats;

  const groups = [...new Set(styles.map(row => row.style_cluster))].sort();
  // Alternate labels above and below in x order so neighbouring teams do not overlap.
  const labelSide = Object.fromEntries([...styles].sort((a, b) => num(a.pc1) - num(b.pc1)).map((row, index) => [row.team, index % 2 ? "bottom center" : "top center"]));
  Plotly.react("styles-chart", groups.map(group => {
    const members = styles.filter(row => row.style_cluster === group);
    return {
      x: members.map(row => num(row.pc1)), y: members.map(row => num(row.pc2)), text: members.map(row => shortTeam(row.team)),
      name: members[0].style_label, mode: "markers+text", textposition: members.map(row => labelSide[row.team]), textfont: { size: 11 }, cliponaxis: false,
      marker: { size: 16, color: clusterColours[group], line: { color: "#fff", width: 1.5 } },
      customdata: members.map(row => [row.team, row.final_rank]),
      hovertemplate: "<b>%{customdata[0]}</b><br>Final position: %{customdata[1]}<extra>%{fullData.name}</extra>",
    };
  }), plotLayout({
    xaxis: { title: { text: `PC1 · ${pct(summary.pc1_explained_variance_ratio, 1)} of variation → more points` }, gridcolor: colours.grid, zerolinecolor: "#c9d6e2" },
    yaxis: { title: { text: `PC2 · ${pct(summary.pc2_explained_variance_ratio, 1)}` }, gridcolor: colours.grid, zerolinecolor: "#c9d6e2" },
    legend: { orientation: "h", y: -0.18 }, margin: { l: 64, r: 24, t: 24, b: 96 },
  }), plotConfig);

  document.querySelectorAll(".pca-model-trigger").forEach(button => {
    const active = button.dataset.pcaModel === model;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active);
  });
  document.querySelector("#pca-model-status").textContent = sensitivity
    ? `Sensitivity model: ${reduced.excluded_team} is removed before standardising and fitting, so coordinates are not comparable with the full-league map. Compare the variance and loadings instead.`
    : `Full-league model with all ${full.teams_included} teams. Switch to the second model to test whether Barcelona's season is driving the structure.`;

  document.querySelector("#pca-summary").textContent = `Two components retain ${pct(summary.two_component_variance_ratio, 1)} of the variation across ${summary.teams_included} teams. PC1 alone explains ${pct(summary.pc1_explained_variance_ratio, 1)} and correlates with points per match at r = ${fmt(summary.pc1_points_correlation)}.${num(summary.pc1_explained_variance_ratio) >= 0.6 && num(summary.pc1_points_correlation) >= 0.8 ? " Teams in Liga F differ far more in how much they dominate than in how they play." : ""}`;
  const pc1Drop = (num(full.pc1_explained_variance_ratio) - num(reduced.pc1_explained_variance_ratio)) * 100;
  const persists = num(reduced.pc1_explained_variance_ratio) >= 0.6;
  document.querySelector("#pca-influence").innerHTML = `<strong>Barcelona check:</strong> without Barcelona, PC1 explains ${pct(reduced.pc1_explained_variance_ratio, 1)} instead of ${pct(full.pc1_explained_variance_ratio, 1)} (${signed(-pc1Drop, 1)} points). ${persists ? "Barcelona amplifies the dominance pattern but does not create it: the other 15 teams line up on the same axis." : "Much of the structure depends on Barcelona, so read the full-league map with care."}`;

  document.querySelector("#pca-component-cards").innerHTML = ["PC1", "PC2"].map(component => {
    const rows = loadings.filter(row => row.component === component).sort((a, b) => num(b.absolute_loading) - num(a.absolute_loading));
    const variance = num(rows[0].explained_variance_ratio);
    const isFirst = component === "PC1";
    const description = describeComponent(rows, isFirst ? "right" : "up", isFirst ? "left" : "down");
    const caution = variance < 0.15
      ? `<p class="interpretation-caution"><strong>Use cautiously:</strong> this axis explains only ${pct(variance, 1)} of the variation. Treat it as a lead to investigate, not a style label.</p>` : "";
    const loadingRows = rows.map(row => `<li><span>${row.feature_label}</span><strong class="${num(row.loading) >= 0 ? "positive" : "negative"}">${signed(row.loading, 2)}</strong></li>`).join("");
    return `<article class="component-card"><p class="component-label">${component} · ${pct(variance, 1)} explained</p><h4>${isFirst ? "Horizontal axis" : "Vertical axis"}</h4><div class="component-interpretation${caution ? " component-interpretation--caution" : ""}"><p>${description}</p>${caution}</div><details><summary>Show all seven loadings</summary><ul class="loading-list">${loadingRows}</ul></details></article>`;
  }).join("");

  const metrics = [
    ["npxg_per_match", "npxG / match", 2], ["passes_completed_per_match", "Completed passes / match", 0],
    ["passes_into_final_third_per_match", "Final-third entries / match", 1], ["high_regains_per_match", "High regains / match", 1],
    ["xg_conceded_per_match", "xG conceded / match", 2], ["points_per_match", "Points / match", 2],
  ];
  const teamLookup = Object.fromEntries(state.teams.map(team => [team.team, team]));
  document.querySelector("#cluster-intro").textContent = `K-means splits the teams into three groups using the same seven standardised measures, and groups are numbered by average points. A silhouette score of ${fmt(summary.kmeans_silhouette)} (0 means no structure, 1 means perfectly separated) shows the groups are only moderately distinct. Teams in the same colour share a level of output and control, not necessarily an attacking style.`;
  document.querySelector("#cluster-profiles").innerHTML = groups.map(group => {
    const members = styles.filter(row => row.style_cluster === group).map(row => ({ ...teamLookup[row.team], ...row }))
      .sort((a, b) => num(a.final_rank) - num(b.final_rank));
    const context = members.length === 1
      ? `${escapeHtml(members[0].team)} is far enough from every other team on these measures to form a group alone.`
      : `Final positions ${members[0].final_rank}–${members.at(-1).final_rank}. These teams share similar output and control.`;
    return `<article class="cluster-card cluster-card--${group}"><p class="cluster-rank">${members.length} ${members.length === 1 ? "team" : "teams"}</p><h4>${escapeHtml(members[0].style_label)}</h4><p class="cluster-teams">${members.map(team => escapeHtml(shortTeam(team.team))).join(" · ")}</p><ul class="cluster-metrics">${metrics.map(([key, label, digits]) => `<li><span>${label}</span><strong>${fmt(mean(members, key), digits)}</strong></li>`).join("")}</ul><p class="cluster-context">${context}</p></article>`;
  }).join("");
}

// ---------- Passing ----------
function renderPassing() {
  crestScatter("passing-chart", state.teams, {
    x: "passes_completed_per_match", y: "progressive_passes_per_match", xTitle: "Completed passes per match", yTitle: "Open-play progressive passes per match", xFormat: ".0f", yFormat: ".0f",
    hover: { data: team => [num(team.pass_completion_pct), team.final_rank], template: "Completed passes / match: %{x:.0f}<br>Progressive passes / match: %{y:.1f}<br>Completion: %{customdata[0]:.1f}%<br>Final position: %{customdata[1]}" },
  });
}

function renderNetwork(teamName) {
  const nodes = state.networkNodes.filter(node => node.team === teamName);
  const nodeLookup = Object.fromEntries(nodes.map(node => [node.player, node]));
  const pairs = {};
  state.networkEdges.filter(edge => edge.team === teamName).forEach(edge => {
    const [a, b] = [edge.player, edge.pass_recipient].sort();
    const key = `${a}|${b}`;
    pairs[key] ??= { a, b, total: 0, detail: [] };
    pairs[key].total += num(edge.passes);
    pairs[key].detail.push(`${shortPlayer(edge.player)} → ${shortPlayer(edge.pass_recipient)}: ${edge.passes}`);
  });
  const edges = Object.values(pairs).filter(pair => nodeLookup[pair.a] && nodeLookup[pair.b]);
  const maxPasses = Math.max(...edges.map(edge => edge.total), 1);
  const maxInvolvement = Math.max(...nodes.map(node => num(node.involvements)), 1);

  const traces = edges.map(edge => ({
    x: [num(nodeLookup[edge.a].x), num(nodeLookup[edge.b].x)], y: [num(nodeLookup[edge.a].y), num(nodeLookup[edge.b].y)],
    mode: "lines", hoverinfo: "skip", showlegend: false,
    line: { color: `rgba(11,76,158,${0.15 + 0.6 * edge.total / maxPasses})`, width: 1 + 8 * edge.total / maxPasses },
  }));
  traces.push({
    x: edges.map(edge => (num(nodeLookup[edge.a].x) + num(nodeLookup[edge.b].x)) / 2),
    y: edges.map(edge => (num(nodeLookup[edge.a].y) + num(nodeLookup[edge.b].y)) / 2),
    text: edges.map(edge => edge.detail.join("<br>")), mode: "markers", marker: { size: 12, opacity: 0 },
    hovertemplate: "%{text}<extra></extra>", showlegend: false,
  });
  traces.push({
    x: nodes.map(node => num(node.x)), y: nodes.map(node => num(node.y)), text: nodes.map(node => shortPlayer(node.player)),
    customdata: nodes.map(node => [node.player, node.starts, Math.round(num(node.minutes))]),
    mode: "markers+text", textposition: "bottom center", textfont: { size: 10, color: colours.ink },
    marker: { size: nodes.map(node => 12 + 18 * num(node.involvements) / maxInvolvement), color: colours.cyan, line: { color: colours.navy, width: 1.5 } },
    hovertemplate: "<b>%{customdata[0]}</b><br>%{customdata[1]} starts · %{customdata[2]} minutes<extra></extra>", showlegend: false,
  });
  Plotly.react("network-chart", traces, pitchLayout(), plotConfig);
}

// ---------- Defending ----------
function renderDefending() {
  crestScatter("defending-chart", state.teams, {
    x: "high_regains_per_match", y: "xg_conceded_per_match", xTitle: "High regains per match", yTitle: "xG conceded per match (lower is better)",
    hover: { data: team => [num(team.shots_conceded_per_match), team.final_rank], template: "High regains / match: %{x:.1f}<br>xG conceded / match: %{y:.2f}<br>Shots conceded / match: %{customdata[0]:.1f}<br>Final position: %{customdata[1]}" },
  });
}

// ---------- Creators ----------
/** Short names on phones, unless two players would share a label (Plotly would merge their bars). */
function barLabels(names) {
  if (innerWidth >= 600) return names;
  const short = names.map(shortPlayer);
  return new Set(short).size === short.length ? short : names;
}

const CREATOR_MODES = {
  xa90: { key: "derived_xa_per90", axis: "Derived xA per 90 minutes", note: count => `Value of the final pass, per 90 minutes. Players with at least ${MINIMUM_MINUTES} minutes (${count} eligible).`, perNinety: true },
  xt90: { key: "total_xt_per90", axis: "Open-play xT per 90 minutes", note: count => `Threat added by every completed open-play pass and carry, per 90 minutes. Players with at least ${MINIMUM_MINUTES} minutes (${count} eligible).`, perNinety: true },
  total: { key: "derived_xa", axis: "Derived xA (season total)", note: () => "Season totals favour players who played the most. Switch to a per-90 view to compare rates.", perNinety: false },
};

function renderCreators() {
  const team = document.querySelector("#creator-team").value;
  const mode = CREATOR_MODES[state.creatorMode];
  const profileLookup = Object.fromEntries(state.profiles.map(player => [playerKey(player), player]));
  const rows = state.players.map(player => ({ ...player, total_xt_per90: profileLookup[playerKey(player)]?.total_xt_per90 }));
  // Players who never created a shot still appear in the xT ranking.
  state.profiles.forEach(profile => { if (!rows.some(row => playerKey(row) === playerKey(profile))) rows.push({ ...profile, direct_shot_assists: 0, derived_xa: 0, goal_assists: 0, meets_minutes_threshold: "True" }); });
  const pool = rows.filter(player => (team === "All" || player.team === team) && (!mode.perNinety || (isTrue(player.meets_minutes_threshold) && player[mode.key] !== undefined)));
  const ranked = [...pool].sort((a, b) => num(b[mode.key]) - num(a[mode.key]));
  const top = ranked.slice(0, 12).reverse();

  document.querySelectorAll("[data-creator-mode]").forEach(button => {
    const active = button.dataset.creatorMode === state.creatorMode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active);
  });
  document.querySelector("#creator-note").textContent = mode.note(pool.length);

  Plotly.react("creator-chart", [{
    type: "bar", orientation: "h", y: barLabels(top.map(player => player.player)), x: top.map(player => num(player[mode.key])),
    text: top.map(player => player.player), textposition: "none",
    marker: { color: state.creatorMode === "xt90" ? colours.cyan : colours.blue },
    customdata: top.map(player => [player.team, Math.round(num(player.minutes)), player.goal_assists, player.direct_shot_assists]),
    hovertemplate: `<b>%{text}</b><br>%{customdata[0]}<br>${mode.axis}: %{x:.3f}<br>Minutes: %{customdata[1]}<br>Goal assists: %{customdata[2]}<br>Created shots: %{customdata[3]}<extra></extra>`,
  }], plotLayout({ margin: { l: 16, r: 24, t: 16, b: 52 }, xaxis: { title: { text: mode.axis }, gridcolor: colours.grid }, yaxis: { automargin: true } }), plotConfig);

  document.querySelector("#creator-table").innerHTML = ranked.slice(0, 20).map(player => `<tr><td>${escapeHtml(player.player)}</td><td><span class="team-cell">${teamBadge(player.team)}${escapeHtml(shortTeam(player.team))}</span></td><td>${fmt(player.minutes, 0)}</td><td>${fmt(player.direct_shot_assists, 0)}</td><td>${fmt(player.derived_xa)}</td><td>${player.derived_xa_per90 ? fmt(player.derived_xa_per90) : "—"}</td><td>${player.total_xt_per90 ? fmt(player.total_xt_per90, 3) : "—"}</td><td>${fmt(player.goal_assists, 0)}</td></tr>`).join("")
    || `<tr><td colspan="8">No players match this selection.</td></tr>`;
}

function renderThreat() {
  threatGridChart("xt-grid-chart", state.xtGrid);
  const { ranked, hiddenCreators, xtXaCorrelation } = state.stats;
  threatScatter("threat-scatter", ranked, hiddenCreators, "threat-hidden-list");
  const groups = hiddenCreators.reduce((counts, player) => ({ ...counts, [player.position_group]: (counts[player.position_group] || 0) + 1 }), {});
  document.querySelector("#threat-summary").innerHTML = `Across ${ranked.length} outfield players, xT and derived xA per 90 correlate at r = ${fmt(xtXaCorrelation)}: related, but far from the same thing. <strong>${hiddenCreators.length}</strong> players rank in the top 25 for xT but outside the top 40 for xA (${Object.entries(groups).map(([group, count]) => `${count} ${group.toLowerCase()}${count === 1 ? "" : "s"}`).join(", ")}). These are the players who move the ball into dangerous areas without making the final pass.`;
}

// ---------- Players (similarity) ----------
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let index = 0; index < a.length; index += 1) { dot += a[index] * b[index]; na += a[index] ** 2; nb += b[index] ** 2; }
  return dot / Math.sqrt(na * nb);
}

function renderSimilarity() {
  const { profiles } = state;
  const input = document.querySelector("#player-search");
  const target = state.profileByLabel[input.value];
  const message = document.querySelector("#similar-message");
  if (!target) { message.textContent = "Choose a player from the list (players with at least 600 minutes)."; return; }
  message.textContent = "";
  const sameGroup = document.querySelector("#same-position").checked;
  const vector = player => Object.keys(PROFILE_FEATURES).map(feature => num(player[`z_${feature}`]));
  const targetVector = vector(target);
  const similar = profiles
    .filter(player => playerKey(player) !== playerKey(target) && (!sameGroup || player.position_group === target.position_group))
    .map(player => ({ player, score: cosine(targetVector, vector(player)) }))
    .sort((a, b) => b.score - a.score).slice(0, 8);
  state.similarity = { target, similar, compare: similar[0]?.player };

  document.querySelector("#similar-target").innerHTML = `${teamBadge(target.team, "large")}<div><strong>${escapeHtml(target.player)}</strong><span>${escapeHtml(shortTeam(target.team))} · ${target.position} · ${fmt(target.minutes, 0)} minutes</span></div>`;
  document.querySelector("#similar-list").innerHTML = similar.map(({ player, score }, index) => `<li><button type="button" class="similar-item${index === 0 ? " active" : ""}" data-index="${index}" aria-pressed="${index === 0}"><span class="similar-rank">${index + 1}</span>${teamBadge(player.team)}<span class="similar-name"><strong>${escapeHtml(player.player)}</strong><small>${escapeHtml(shortTeam(player.team))} · ${player.position}</small></span><span class="similar-score">${Math.max(0, score * 100).toFixed(0)}%</span></button></li>`).join("");
  renderComparison();
}

function renderComparison() {
  const { target, compare } = state.similarity;
  if (!compare) return;
  document.querySelectorAll(".similar-item").forEach(button => {
    const active = state.similarity.similar[num(button.dataset.index)].player === compare;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active);
  });
  const features = Object.keys(PROFILE_FEATURES).reverse();
  const trace = (player, name, colour) => ({
    type: "bar", orientation: "h", name, y: features.map(feature => PROFILE_FEATURES[feature]), x: features.map(feature => num(player[`pct_${feature}`]) * 100),
    marker: { color: colour }, customdata: features.map(feature => num(player[feature])),
    hovertemplate: `${escapeHtml(name)}<br>%{y}: %{customdata:.2f} per 90<br>Percentile: %{x:.0f}<extra></extra>`,
  });
  Plotly.react("similarity-chart", [trace(compare, compare.player, colours.cyan), trace(target, target.player, colours.blue)], plotLayout({
    barmode: "group", bargap: 0.3, bargroupgap: 0.05, margin: { l: 16, r: 24, t: 40, b: 52 },
    xaxis: { title: { text: `Percentile among ${target.position_group.toLowerCase()}s` }, range: [0, 100], gridcolor: colours.grid },
    yaxis: { automargin: true }, legend: { orientation: "h", x: 0, y: 1.08, traceorder: "reversed" },
  }), plotConfig);
}

function setUpSimilarity() {
  const { profiles } = state;
  state.profileByLabel = Object.fromEntries(profiles.map(player => [`${player.player} (${shortTeam(player.team)})`, player]));
  const labels = Object.keys(state.profileByLabel).sort((a, b) => a.localeCompare(b));
  document.querySelector("#player-options").innerHTML = labels.map(label => `<option value="${escapeHtml(label)}"></option>`).join("");
  const input = document.querySelector("#player-search");
  input.value = labels.find(label => label.startsWith("Aitana")) || labels[0];
  input.addEventListener("change", renderSimilarity);
  document.querySelector("#same-position").addEventListener("change", renderSimilarity);
  document.querySelector("#similar-list").addEventListener("click", event => {
    const button = event.target.closest(".similar-item");
    if (!button) return;
    state.similarity.compare = state.similarity.similar[num(button.dataset.index)].player;
    renderComparison();
  });
}

// ---------- Method ----------
function renderMethod() {
  const check = state.validation;
  const { pcaFull: full, models, gapAll } = state.stats;
  const cards = [
    ["Data and reconciliation", `StatsBomb open event data for all ${fmt(check.matches_with_cached_events, 0)} matches and ${fmt(check.teams_in_match_metadata, 0)} teams: ${fmt(check.total_shots, 0)} shots. Event goals (${fmt(check.shot_goals, 0)} from shots + ${fmt(check.own_goal_for_events, 0)} own goals) match the ${fmt(check.goals_from_match_metadata, 0)} goals in the final scores.`],
    ["Chance categories", `Each shot gets one label, in order: penalty, direct free kick, corner, indirect free kick or attacking throw-in. For the last three, the restart must come within 10 seconds of the shot in the same possession. Next comes transition (StatsBomb's counter-attack pattern), then StatsBomb's cut-back, cross and through-ball flags on the final pass, and otherwise other open play. Labelling by the start of the possession instead would call ${pct(check.shots_in_possessions_started_by_restart_share)} of shots set pieces; the 10-second rule gives ${pct(check.set_piece_shot_share)}.`],
    ["Derived xA and minutes", `Derived xA is the summed non-penalty xG of shots that a player's pass set up directly. It is not StatsBomb's own xA model. Minutes are rebuilt from line-ups, substitutions and red cards, and they total ${pct(check.player_minutes_vs_eleven_per_side, 1)} of eleven players per side. Per-90 rankings require ${MINIMUM_MINUTES} minutes.`],
    ["Liga F xG model", `Trained on ${fmt(gapAll.shots, 0)} non-penalty shots with location, body part, technique, assist type and freeze-frame features (defenders and goalkeeper between ball and goal, nearby pressure). Every model is scored on matches it never saw, using 5-fold cross-validation grouped by match. Gradient boosting reaches AUC ${fmt(models.boosting.roc_auc, 3)}, and StatsBomb reaches ${fmt(models.statsbomb.roc_auc, 3)}.`],
    ["Expected threat (xT)", "A 16 × 12 grid model in the style of Karun Singh, fitted on this season's open-play passes, carries and shots. A completed pass or carry earns the xT of the cell it ends in minus the cell it starts in. Players on dominant teams have more chances to add threat, so compare players within similar teams where possible."],
    ["Player similarity", `Cosine similarity across ${Object.keys(PROFILE_FEATURES).length} standardised per-90 measures for outfield players with at least ${MINIMUM_MINUTES} minutes. Percentiles are computed within each position group. Similarity describes statistical output, not playing style or quality.`],
    ["PCA and groups", `Seven standardised team measures. PC1 explains ${pct(full.pc1_explained_variance_ratio, 1)} of the variation and tracks points per match (r = ${fmt(full.pc1_points_correlation)}). K-means uses three groups (silhouette ${fmt(full.kmeans_silhouette)}). A second model without Barcelona checks how much one team drives the result.`],
    ["Limitations", "All results are associations, not causal effects. There is no post-shot xG, so goalkeeper numbers reflect the chances faced, not shot-stopping alone. Passing networks combine the whole season. Everything comes from a single season of one league."],
  ];
  document.querySelector("#method-grid").innerHTML = cards.map(([title, text]) => `<article><h3>${title}</h3><p>${text}</p></article>`).join("");
}

// ---------- Tabs ----------
const tabNames = () => [...document.querySelectorAll(".tab-trigger")].map(button => button.dataset.tab);

function renderTab(tabName) {
  if (state.rendered.has(tabName)) return;
  state.rendered.add(tabName);
  if (tabName === "overview") renderOverview();
  if (tabName === "attacking") { renderProfile(document.querySelector("#team-select").value); renderShotMap(); }
  if (tabName === "finishing") renderFinishing();
  if (tabName === "styles") renderStyles();
  if (tabName === "passing") { renderPassing(); renderNetwork(document.querySelector("#network-team-select").value); }
  if (tabName === "defending") renderDefending();
  if (tabName === "creators") { renderCreators(); renderThreat(); }
  if (tabName === "players") renderSimilarity();
  if (tabName === "method") renderMethod();
}

function activateTab(tabName, { focus = false, updateHash = true } = {}) {
  if (!tabNames().includes(tabName)) tabName = "overview";
  document.querySelectorAll(".tab-trigger").forEach(button => {
    const active = button.dataset.tab === tabName;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", active);
    button.tabIndex = active ? 0 : -1;
    if (active && focus) button.focus();
    // On narrow screens the strip scrolls sideways; keep the active tab in view.
    if (active) button.parentElement.scrollTo({ left: button.offsetLeft - (button.parentElement.clientWidth - button.offsetWidth) / 2 });
  });
  document.querySelectorAll("[data-tab-panel]").forEach(panel => { panel.hidden = panel.dataset.tabPanel !== tabName; });
  if (updateHash) history.replaceState(null, "", `#${tabName}`);
  renderTab(tabName);
  requestAnimationFrame(() => document.querySelectorAll(`[data-tab-panel="${tabName}"] .js-plotly-plot`).forEach(plot => Plotly.Plots.resize(plot)));
}

function setUpTabs() {
  const strip = document.querySelector(".tab-strip");
  strip.addEventListener("click", event => { const button = event.target.closest(".tab-trigger"); if (button) activateTab(button.dataset.tab); });
  strip.addEventListener("keydown", event => {
    const names = tabNames(), current = names.indexOf(document.activeElement.dataset.tab);
    const moves = { ArrowRight: current + 1, ArrowLeft: current - 1, Home: 0, End: names.length - 1 };
    if (!(event.key in moves)) return;
    event.preventDefault();
    activateTab(names[(moves[event.key] + names.length) % names.length], { focus: true });
  });
  window.addEventListener("hashchange", () => activateTab(location.hash.slice(1), { updateHash: false }));
}

// ---------- Start ----------
function fillSelect(selector, options, selected) {
  const select = document.querySelector(selector);
  options.forEach(option => select.add(new Option(option, option)));
  if (selected) select.value = selected;
  return select;
}

async function initialise() {
  Object.assign(state, await loadTables([
    "teams", "players", "profiles", "shots", "styles", "loadings", "stylesSensitivity", "loadingsSensitivity", "pcaSummary",
    "networkNodes", "networkEdges", "validation", "xtGrid", "xgModels", "xgCalibration", "xgGap", "xgTeams", "xgShooters", "xgGoalkeepers", "xgFeatures",
  ]));
  state.stats = headlineStats(state);

  const teamNames = state.teams.map(team => team.team).sort((a, b) => a.localeCompare(b));
  const leader = [...state.teams].sort((a, b) => num(a.final_rank) - num(b.final_rank))[0].team;
  fillSelect("#team-select", teamNames, leader).addEventListener("change", event => renderProfile(event.target.value));
  fillSelect("#shot-team-select", teamNames).addEventListener("change", renderShotMap);
  fillSelect("#shot-category-select", CATEGORIES).addEventListener("change", renderShotMap);
  fillSelect("#network-team-select", teamNames, leader).addEventListener("change", event => renderNetwork(event.target.value));
  fillSelect("#creator-team", teamNames).addEventListener("change", renderCreators);
  document.querySelectorAll(".pca-model-trigger").forEach(button => button.addEventListener("click", () => { state.pcaModel = button.dataset.pcaModel; renderStyles(); }));
  document.querySelectorAll("[data-creator-mode]").forEach(button => button.addEventListener("click", () => { state.creatorMode = button.dataset.creatorMode; renderCreators(); }));
  setUpSimilarity();

  setUpTabs();
  activateTab(location.hash.slice(1) || "overview", { updateHash: false });
}

initialise().catch(showLoadError);
