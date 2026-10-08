// Shared data loading, formatting and chart helpers for the story and the dashboard.

export const DATA = {
  teams: "data/liga_f_team_attacking_profile.csv",
  players: "data/liga_f_player_derived_xa.csv",
  profiles: "data/liga_f_player_profiles.csv",
  shots: "data/liga_f_shot_map.csv",
  styles: "data/liga_f_team_styles.csv",
  loadings: "data/liga_f_pca_loadings.csv",
  stylesSensitivity: "data/liga_f_team_styles_without_barcelona.csv",
  loadingsSensitivity: "data/liga_f_pca_loadings_without_barcelona.csv",
  pcaSummary: "data/liga_f_pca_sensitivity_summary.csv",
  networkNodes: "data/liga_f_passing_network_nodes.csv",
  networkEdges: "data/liga_f_passing_network_edges.csv",
  validation: "data/liga_f_validation_report.csv",
  xtGrid: "data/liga_f_xt_grid.csv",
  xgModels: "data/liga_f_xg_model_comparison.csv",
  xgCalibration: "data/liga_f_xg_calibration.csv",
  xgGap: "data/liga_f_xg_gap_breakdown.csv",
  xgTeams: "data/liga_f_xg_team_finishing.csv",
  xgShooters: "data/liga_f_xg_shooter_finishing.csv",
  xgGoalkeepers: "data/liga_f_xg_goalkeeper_faced.csv",
  xgFeatures: "data/liga_f_xg_feature_effects.csv",
};

// Must match CHANCE_CATEGORIES, MINIMUM_PLAYER_MINUTES and DISTANCE_LABELS in src/.
export const CATEGORIES = ["Penalty", "Direct free kick", "Corner", "Indirect free kick", "Attacking throw-in", "Transition", "Cutback", "Cross", "Through ball", "Other open play"];
export const SET_PIECE_CATEGORIES = CATEGORIES.slice(0, 5);
export const DISTANCE_ORDER = ["0–6 yd", "6–12 yd", "12–18 yd", "18–25 yd", "25+ yd"];
export const MINIMUM_MINUTES = 600;

export const colours = { navy: "#062a68", blue: "#0b4c9e", cyan: "#00a9e0", gold: "#f2b234", red: "#c2414b", ink: "#092660", muted: "#627595", grid: "#e5e9eb", soft: "#9fb4cf" };
export const plotConfig = { responsive: true, displayModeBar: false };

const teamLogos = {
  "Athletic Club Bilbao": "athletic", "Atletico Madrid W": "atletico", "Barcelona WFC": "barcelona",
  "CD Sporting de Huelva": "sporting-huelva", "Costa Adeje Tenerife": "tenerife", "FC Levante Badalona": "badalona",
  "Granada W": "granada", "Levante UD Femenino": "levante", "Madrid CFF": "madrid-cff",
  "Real Betis Balompié": "betis", "Real Madrid CF W": "real-madrid", "Real Sociedad de Fútbol": "real-sociedad",
  "SD Eibar": "eibar", "Sevilla FC": "sevilla", "Valencia CF": "valencia", "Villarreal": "villarreal",
};
export const logoPath = team => teamLogos[team] ? `assets/logos/${teamLogos[team]}.png` : null;

// ---------- Formatting ----------
export const num = value => Number(value);
export const fmt = (value, digits = 2) => num(value).toLocaleString("en-GB", { minimumFractionDigits: digits, maximumFractionDigits: digits });
export const pct = (value, digits = 0) => `${(num(value) * 100).toFixed(digits)}%`;
export const signed = (value, digits = 1) => `${num(value) >= 0 ? "+" : "−"}${Math.abs(num(value)).toFixed(digits)}`;
export const mean = (items, key) => items.reduce((sum, item) => sum + num(item[key]), 0) / items.length;
export const shortTeam = team => team.replace(" WFC", "").replace(" CF W", "").replace(" Femenino", "").replace(/ W$/, "").replace(" de Fútbol", "");
export const shortPlayer = name => { const words = name.trim().split(/\s+/); return words.length > 1 ? `${words[0]} ${words[1][0]}.` : name; };
export const escapeHtml = text => String(text).replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
export const isTrue = value => String(value).toLowerCase() === "true";
export const ordinal = value => { const n = num(value), tens = n % 100; return `${n}${tens >= 11 && tens <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" }[n % 10] || "th")}`; };

/** Plain-English odds that a gap of z standard deviations (or more) arises by chance. */
export function chancePhrase(z) {
  if (z >= 3.72) return "would happen by chance less than once in ten thousand seasons";
  if (z >= 3.09) return "would happen by chance less than once in a thousand seasons";
  if (z >= 2.33) return "would happen by chance less than once in a hundred seasons";
  if (z >= 1.65) return "would happen by chance less than once in twenty seasons";
  return "could plausibly be chance";
}

export function correlation(items, xKey, yKey) {
  const mx = mean(items, xKey), my = mean(items, yKey);
  let sxy = 0, sxx = 0, syy = 0;
  items.forEach(item => { const dx = num(item[xKey]) - mx, dy = num(item[yKey]) - my; sxy += dx * dy; sxx += dx * dx; syy += dy * dy; });
  return sxy / Math.sqrt(sxx * syy);
}

export function teamBadge(team, size = "small") {
  const src = logoPath(team);
  return src ? `<img class="team-badge team-badge--${size}" src="${src}" alt="" onerror="this.remove()">` : "";
}

// ---------- Data ----------
async function loadCsv(path) {
  // Data files keep their names between pipeline runs, so always revalidate them.
  const response = await fetch(path, { cache: "no-cache" });
  if (!response.ok) throw new Error(`Could not load ${path}`);
  return d3.csvParse(await response.text());
}

/** Load the named tables from DATA; `validation` is returned as a {check: value} lookup. */
export async function loadTables(names) {
  const tables = await Promise.all(names.map(name => loadCsv(DATA[name])));
  const result = Object.fromEntries(names.map((name, index) => [name, tables[index]]));
  if (result.validation) result.validation = Object.fromEntries(result.validation.map(row => [row.check, num(row.value)]));
  return result;
}

export function showLoadError(error) {
  console.error(error);
  document.body.insertAdjacentHTML("afterbegin", `<p class="load-error" role="alert">The data could not be loaded. If you opened the HTML file directly, serve the <code>app</code> folder with a local web server instead (see app/README.md).</p>`);
}

/** Stats that several views quote, computed once so the story and dashboard always agree. */
export function headlineStats({ teams, xgGap, xgModels, xgTeams, profiles, pcaSummary }) {
  const stats = {};
  if (teams) {
    const ranked = [...teams].sort((a, b) => num(a.final_rank) - num(b.final_rank));
    const topFour = ranked.slice(0, 4), rest = ranked.slice(4);
    stats.volumeGap = mean(topFour, "non_penalty_shots_per_match") / mean(rest, "non_penalty_shots_per_match") - 1;
    stats.qualityGap = mean(topFour, "npxg_per_shot") / mean(rest, "npxg_per_shot") - 1;
    stats.rVolume = correlation(teams, "non_penalty_shots_per_match", "points_per_match");
    stats.rQuality = correlation(teams, "npxg_per_shot", "points_per_match");
    stats.mixGaps = CATEGORIES.map(category => ({ category, gap: mean(topFour, `${category}_shot_share`) - mean(rest, `${category}_shot_share`) }))
      .sort((a, b) => Math.abs(b.gap) - Math.abs(a.gap));
  }
  if (pcaSummary) {
    stats.pcaFull = pcaSummary.find(row => row.model === "all_teams");
    stats.pcaReduced = pcaSummary.find(row => row.model !== "all_teams");
  }
  if (xgGap) {
    const find = (dimension, group) => xgGap.find(row => row.dimension === dimension && row.group === group);
    stats.gapAll = find("All non-penalty shots", "All");
    stats.gapOpen = find("Freeze frame", "No defender between ball and goal");
    stats.gapBlocked = find("Freeze frame", "Defender(s) in the way");
  }
  if (xgModels) stats.models = Object.fromEntries(xgModels.map(row => [row.model, row]));
  if (xgTeams) stats.xgTeams = Object.fromEntries(xgTeams.map(row => [row.team, row]));
  if (profiles) {
    const byXt = [...profiles].sort((a, b) => num(b.total_xt_per90) - num(a.total_xt_per90));
    const byXa = [...profiles].sort((a, b) => num(b.derived_xa_per90) - num(a.derived_xa_per90));
    const xaRank = Object.fromEntries(byXa.map((player, index) => [`${player.team}|${player.player}`, index + 1]));
    stats.xtXaCorrelation = correlation(profiles, "total_xt_per90", "derived_xa_per90");
    stats.ranked = byXt.map((player, index) => ({ ...player, xtRank: index + 1, xaRank: xaRank[`${player.team}|${player.player}`] }));
    // "Hidden creators": top 25 for threat added, outside the top 40 for final-pass value.
    stats.hiddenCreators = stats.ranked.filter(player => player.xtRank <= 25 && player.xaRank > 40);
    stats.eligiblePlayers = profiles.length;
  }
  return stats;
}

// ---------- Charts ----------
export function plotLayout(extra = {}) {
  return {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Montserrat, Arial, sans-serif", color: colours.ink, size: 12 },
    margin: { l: 64, r: 24, t: 24, b: 56 },
    hoverlabel: { font: { family: "Montserrat, Arial, sans-serif" } },
    ...extra,
  };
}

export function paddedRange(values, padding = 0.08) {
  const min = Math.min(...values), max = Math.max(...values), span = max - min || 1;
  return [min - span * padding, max + span * padding];
}

/** Team scatter where the crest is the marker; dashed lines mark the league average. */
export function crestScatter(elementId, teams, { x, y, xTitle, yTitle, xFormat = ".1f", yFormat = ".2f", hover }) {
  const xs = teams.map(team => num(team[x])), ys = teams.map(team => num(team[y]));
  const xRange = paddedRange(xs), yRange = paddedRange(ys);
  const sizeX = (xRange[1] - xRange[0]) * 0.055, sizeY = (yRange[1] - yRange[0]) * 0.085;
  const xMean = mean(teams, x), yMean = mean(teams, y);
  const averageLine = { type: "line", line: { color: colours.muted, width: 1, dash: "dot" }, layer: "below" };
  Plotly.react(elementId, [{
    x: xs, y: ys, text: teams.map(team => team.team), mode: "markers",
    marker: { size: 30, opacity: 0 },
    customdata: teams.map(hover.data),
    hovertemplate: `<b>%{text}</b><br>${hover.template}<extra></extra>`,
  }], plotLayout({
    xaxis: { title: { text: xTitle }, range: xRange, gridcolor: colours.grid, zeroline: false, tickformat: xFormat },
    yaxis: { title: { text: yTitle }, range: yRange, gridcolor: colours.grid, zeroline: false, tickformat: yFormat },
    showlegend: false,
    shapes: [
      { ...averageLine, x0: xMean, x1: xMean, y0: yRange[0], y1: yRange[1] },
      { ...averageLine, x0: xRange[0], x1: xRange[1], y0: yMean, y1: yMean },
    ],
    annotations: [{ x: xRange[1], y: yMean, text: "League average", showarrow: false, xanchor: "right", yanchor: "bottom", font: { size: 10, color: colours.muted } }],
    images: teams.filter(team => logoPath(team.team)).map(team => ({
      source: logoPath(team.team), x: num(team[x]), y: num(team[y]), xref: "x", yref: "y",
      xanchor: "center", yanchor: "middle", sizex: sizeX, sizey: sizeY, sizing: "contain", layer: "above",
    })),
  }), plotConfig);
}

/** Goals per StatsBomb xG for each group, with the break-even line at 1. */
export function gapChart(elementId, rows, { height } = {}) {
  const ordered = [...rows].reverse();
  Plotly.react(elementId, [{
    type: "bar", orientation: "h",
    y: ordered.map(row => row.group), x: ordered.map(row => num(row.goals_per_xg) - 1), base: 1,
    marker: { color: ordered.map(row => num(row.z_score) >= 2 ? colours.blue : colours.soft) },
    customdata: ordered.map(row => [row.shots, row.goals, num(row.xg), num(row.goals_minus_xg), num(row.z_score)]),
    hovertemplate: "<b>%{y}</b><br>%{customdata[0]} shots · %{customdata[1]} goals vs %{customdata[2]:.1f} xG<br>Gap: %{customdata[3]:+.1f} goals (z = %{customdata[4]:.1f})<extra></extra>",
  }], plotLayout({
    height, margin: { l: 16, r: 24, t: 16, b: 52 },
    xaxis: { title: { text: "Goals per StatsBomb xG (1 = as expected)" }, gridcolor: colours.grid, zeroline: false },
    yaxis: { automargin: true },
    shapes: [{ type: "line", x0: 1, x1: 1, yref: "paper", y0: 0, y1: 1, line: { color: colours.ink, width: 1.5 } }],
  }), plotConfig);
}

/** Reliability diagram: predicted chance of scoring against how often shots were scored. */
export function calibrationChart(elementId, rows, models) {
  const traces = models.map(({ key, name, colour, dash }) => {
    const points = rows.filter(row => row.model === key);
    return {
      x: points.map(row => num(row.mean_predicted)), y: points.map(row => num(row.goal_rate)), name,
      mode: "lines+markers", line: { color: colour, width: 2.5, dash }, marker: { size: points.map(row => 5 + Math.sqrt(num(row.shots)) / 4), color: colour },
      customdata: points.map(row => row.shots),
      hovertemplate: `${name}<br>Predicted: %{x:.1%}<br>Scored: %{y:.1%}<br>%{customdata} shots<extra></extra>`,
    };
  });
  traces.unshift({ x: [0, 0.8], y: [0, 0.8], mode: "lines", name: "Perfect calibration", line: { color: colours.muted, dash: "dot", width: 1 }, hoverinfo: "skip" });
  Plotly.react(elementId, traces, plotLayout({
    xaxis: { title: { text: "Predicted chance of scoring" }, tickformat: ".0%", range: [0, 0.8], gridcolor: colours.grid },
    yaxis: { title: { text: "Share of shots scored" }, tickformat: ".0%", range: [0, 0.8], gridcolor: colours.grid },
    legend: { orientation: "h", x: 0, y: -0.2 }, margin: { l: 64, r: 24, t: 16, b: 96 },
  }), plotConfig);
}

/**
 * xT per 90 against derived xA per 90. Highlighted players are listed below the chart in
 * `listId` (labels would overlap in the dense middle); only the league leader is annotated.
 */
export function threatScatter(elementId, ranked, highlighted, listId) {
  const highlightKeys = new Set(highlighted.map(player => `${player.team}|${player.player}`));
  const isHighlighted = player => highlightKeys.has(`${player.team}|${player.player}`);
  const trace = (items, name, colour, emphasised) => ({
    x: items.map(player => num(player.derived_xa_per90)), y: items.map(player => num(player.total_xt_per90)), name, mode: "markers",
    marker: { size: emphasised ? 12 : 7, color: colour, opacity: emphasised ? 1 : 0.55, line: { color: emphasised ? colours.ink : "#fff", width: 1 } },
    customdata: items.map(player => [player.player, shortTeam(player.team), player.position_group, player.xtRank, player.xaRank]),
    hovertemplate: "<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[2]}<br>xT / 90: %{y:.3f} (rank %{customdata[3]})<br>xA / 90: %{x:.3f} (rank %{customdata[4]})<extra></extra>",
  });
  const leader = ranked[0];
  Plotly.react(elementId, [
    trace(ranked.filter(player => !isHighlighted(player)), "Other players", colours.soft, false),
    trace(ranked.filter(isHighlighted), "High threat, few assists", colours.gold, true),
  ], plotLayout({
    xaxis: { title: { text: "Derived xA per 90 (value of the final pass)" }, gridcolor: colours.grid, zeroline: false, rangemode: "tozero" },
    yaxis: { title: { text: "Open-play xT per 90 (passes and carries)" }, gridcolor: colours.grid, zeroline: false, rangemode: "tozero" },
    legend: { orientation: "h", x: 0, y: 1.08 }, margin: { l: 64, r: 24, t: 32, b: 56 },
    annotations: [{ x: num(leader.derived_xa_per90), y: num(leader.total_xt_per90), text: `${shortPlayer(leader.player)} (league leader)`, showarrow: true, arrowhead: 0, ax: -70, ay: 18, font: { size: 11, color: colours.ink } }],
  }), plotConfig);

  if (listId) {
    document.getElementById(listId).innerHTML = [...highlighted].sort((a, b) => a.xtRank - b.xtRank).map(player =>
      `<li>${teamBadge(player.team)}<span><strong>${escapeHtml(player.player)}</strong><small>${escapeHtml(shortTeam(player.team))} · ${player.position_group} · xT ${ordinal(player.xtRank)}, xA ${ordinal(player.xaRank)}</small></span></li>`).join("");
  }
}

export function pitchShapes() {
  const line = { color: "rgba(9,38,96,.45)", width: 1.3 };
  const box = (x0, x1, y0, y1) => ({ type: "rect", x0, x1, y0, y1, line, fillcolor: "rgba(0,0,0,0)", layer: "above" });
  return [
    box(0, 120, 0, 80), { type: "line", x0: 60, x1: 60, y0: 0, y1: 80, line, layer: "above" },
    { type: "circle", x0: 50, x1: 70, y0: 30, y1: 50, line, layer: "above" },
    box(102, 120, 18, 62), box(0, 18, 18, 62), box(114, 120, 30, 50), box(0, 6, 30, 50),
    { type: "line", x0: 120, x1: 120, y0: 36, y1: 44, line: { color: colours.navy, width: 4 } },
    { type: "line", x0: 0, x1: 0, y0: 36, y1: 44, line: { color: colours.navy, width: 4 } },
  ];
}

export function pitchLayout(extra = {}) {
  return plotLayout({
    xaxis: { range: [-2, 122], visible: false, fixedrange: true },
    yaxis: { range: [82, -2], visible: false, scaleanchor: "x", scaleratio: 1, fixedrange: true },
    shapes: pitchShapes(), margin: { l: 10, r: 10, t: 10, b: 10 }, ...extra,
  });
}

/** Heatmap of the xT grid on a pitch. */
export function threatGridChart(elementId, grid) {
  const columns = Math.max(...grid.map(cell => num(cell.column))) + 1, rows = Math.max(...grid.map(cell => num(cell.row))) + 1;
  const z = Array.from({ length: rows }, () => Array(columns).fill(0));
  grid.forEach(cell => { z[num(cell.row)][num(cell.column)] = num(cell.xt); });
  const xs = Array.from({ length: columns }, (_, index) => (index + 0.5) * 120 / columns);
  const ys = Array.from({ length: rows }, (_, index) => (index + 0.5) * 80 / rows);
  Plotly.react(elementId, [{
    type: "heatmap", x: xs, y: ys, z, zmin: 0, colorscale: [[0, "#f3f9f1"], [0.15, "#bfe6f4"], [0.5, colours.cyan], [1, colours.navy]],
    colorbar: { title: { text: "xT" }, thickness: 10, len: 0.8 },
    hovertemplate: "Possession here is worth %{z:.3f} expected goals<extra></extra>",
  }], pitchLayout({ margin: { l: 10, r: 10, t: 10, b: 10 } }), plotConfig);
}
