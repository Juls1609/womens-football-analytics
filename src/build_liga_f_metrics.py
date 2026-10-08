"""Build validated Liga F 2023/24 chance-creation tables from cached StatsBomb events.

Expected input layout
---------------------
<project-dir>/data/raw/events/<match_id>.pkl      (written by src/download_liga_f.py)
<project-dir>/data/processed/liga_f_matches.csv   (written by src/download_liga_f.py)

This script does not download anything. Every definition used here is described
in METHODOLOGY.md; the constants below are the only tuning knobs.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from expected_threat import run_expected_threat
from xg_model import run_xg_analysis


EXPECTED_MATCHES = 240
COMPETITION_NAME = "Liga F 2023/24"
MINIMUM_PLAYER_MINUTES = 600
SENSITIVITY_EXCLUDED_TEAM = "Barcelona WFC"

# StatsBomb coordinates are 120 x 80, with the attacking goal on the right.
ATTACKING_THIRD_X = 80
BOX_X = 102
BOX_Y_MIN, BOX_Y_MAX = 18, 62
PROGRESSIVE_DISTANCE = 10

# A shot is a set-piece shot when the shooting team took a restart in the same
# possession no more than this many seconds earlier.
SET_PIECE_WINDOW_SECONDS = 10
DEAD_BALL_PASS_TYPES = {"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"}

# Passing networks show each team's most frequent starters.
NETWORK_PLAYERS = 11
NETWORK_MIN_PASSES = 10

CHANCE_CATEGORIES = [
    "Penalty",
    "Direct free kick",
    "Corner",
    "Indirect free kick",
    "Attacking throw-in",
    "Transition",
    "Cutback",
    "Cross",
    "Through ball",
    "Other open play",
]

STYLE_FEATURES = {
    "npxg_per_match": "Non-penalty xG per match",
    "npxg_per_shot": "Non-penalty xG per shot",
    "passes_completed_per_match": "Completed passes per match",
    "progressive_passes_per_match": "Progressive passes per match",
    "passes_into_final_third_per_match": "Passes into final third per match",
    "high_regains_per_match": "High regains per match",
    "xg_conceded_per_match": "xG conceded per match",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Folder containing data/raw/events and data/processed/liga_f_matches.csv.",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: list[str], table_name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {', '.join(missing)}")


def coordinate(values: pd.Series, index: int) -> pd.Series:
    """Return one StatsBomb coordinate per row, or NaN for a missing/malformed location."""
    return values.map(
        lambda value: value[index] if isinstance(value, (list, tuple)) and len(value) > index else np.nan
    ).astype(float)


def flag(values: pd.Series) -> pd.Series:
    """StatsBomb boolean qualifiers are True or missing; never treat missing as True."""
    return values.map(
        lambda value: value.strip().lower() in {"true", "1", "yes"} if isinstance(value, str)
        else (False if pd.isna(value) else bool(value))
    ).astype(bool)


def in_box(x: pd.Series, y: pd.Series) -> pd.Series:
    return (x >= BOX_X) & y.between(BOX_Y_MIN, BOX_Y_MAX)


def load_cached_events(
    events_directory: Path,
    combined_events_path: Path,
) -> tuple[pd.DataFrame, list[int]]:
    """Load either a combined season pickle or the resumable per-match cache."""
    if combined_events_path.exists():
        events = pd.read_pickle(combined_events_path)
        print(f"Loaded combined season events from: {combined_events_path}")
        return events, []

    cache_files = sorted(events_directory.glob("*.pkl"))
    if not cache_files:
        raise FileNotFoundError(
            f"No cached event files found in {events_directory}. Run src/download_liga_f.py first."
        )

    event_frames: list[pd.DataFrame] = []
    failed_match_ids: list[int] = []
    for cache_file in cache_files:
        try:
            match_events = pd.read_pickle(cache_file)
            if "match_id" not in match_events.columns:
                match_events["match_id"] = int(cache_file.stem)
            event_frames.append(match_events)
        except Exception as error:  # Keep the audit trail rather than silently skipping a file.
            print(f"WARNING: could not load {cache_file.name}: {error}")
            try:
                failed_match_ids.append(int(cache_file.stem))
            except ValueError:
                pass

    if not event_frames:
        raise RuntimeError("All cached event files failed to load.")
    return pd.concat(event_frames, ignore_index=True), failed_match_ids


def prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    """Add the clock and coordinate columns every later step relies on."""
    require_columns(
        events,
        ["id", "index", "match_id", "period", "minute", "second", "possession", "team", "type", "location"],
        "events",
    )
    events = events.copy()
    # StatsBomb's minute keeps counting across halves (the second half starts at 45:00).
    events["game_seconds"] = events["minute"] * 60 + events["second"]
    events["x"] = coordinate(events["location"], 0)
    events["y"] = coordinate(events["location"], 1)
    return events


def build_shots(events: pd.DataFrame) -> pd.DataFrame:
    shots = events.loc[events["type"].eq("Shot")].copy()
    require_columns(shots, ["shot_statsbomb_xg", "shot_outcome", "shot_type", "play_pattern"], "shots")
    keep = [
        "id", "index", "match_id", "period", "possession", "game_seconds", "minute", "team", "player",
        "play_pattern", "x", "y", "shot_statsbomb_xg", "shot_outcome", "shot_type", "shot_body_part",
    ]
    shots = shots[[column for column in keep if column in shots.columns]].rename(
        columns={"id": "shot_id", "player": "shooter", "x": "shot_x", "y": "shot_y"}
    )
    shots["is_goal_from_shot"] = shots["shot_outcome"].eq("Goal")
    shots["is_penalty"] = shots["shot_type"].eq("Penalty")
    shots["npxg"] = np.where(shots["is_penalty"], 0.0, shots["shot_statsbomb_xg"])
    return shots


def build_direct_assists(events: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        events,
        ["pass_end_location", "pass_shot_assist", "pass_goal_assist", "pass_assisted_shot_id",
         "pass_cross", "pass_cut_back", "pass_through_ball"],
        "events needed for derived xA",
    )
    passes = events.loc[events["type"].eq("Pass")].copy()
    # StatsBomb marks the final pass before a scored shot as a goal assist instead of
    # a shot assist, so both flags are needed to cover every directly assisted shot.
    passes["is_goal_assist"] = flag(passes["pass_goal_assist"])
    assisted = flag(passes["pass_shot_assist"]) | passes["is_goal_assist"]
    assists = passes.loc[assisted & passes["pass_assisted_shot_id"].notna()].copy()

    for column in ["pass_cross", "pass_cut_back", "pass_through_ball"]:
        assists[column] = flag(assists[column])
    assists["assist_end_x"] = coordinate(assists["pass_end_location"], 0)
    assists["assist_end_y"] = coordinate(assists["pass_end_location"], 1)
    assists = assists.rename(
        columns={
            "id": "assist_event_id",
            "team": "assisting_team",
            "player": "assister",
            "x": "assist_start_x",
            "y": "assist_start_y",
            "pass_assisted_shot_id": "shot_id",
        }
    )[[
        "assist_event_id", "match_id", "assisting_team", "assister", "assist_start_x", "assist_start_y",
        "assist_end_x", "assist_end_y", "pass_cross", "pass_cut_back", "pass_through_ball",
        "is_goal_assist", "shot_id",
    ]]

    shot_lookup = shots[
        ["shot_id", "match_id", "team", "shot_statsbomb_xg", "is_penalty", "is_goal_from_shot"]
    ].rename(columns={"team": "shooting_team"})
    assists = assists.merge(shot_lookup, on=["shot_id", "match_id"], how="left", validate="one_to_one")

    missing_shots = int(assists["shot_statsbomb_xg"].isna().sum())
    if missing_shots:
        print(f"WARNING: {missing_shots} direct assists could not be matched to a shot.")
    inconsistent_team = int(
        (assists["shooting_team"].notna() & assists["assisting_team"].ne(assists["shooting_team"])).sum()
    )
    if inconsistent_team:
        print(f"WARNING: {inconsistent_team} matched assists have different passing/shooting teams.")

    # Penalty xG is deliberately excluded from derived xA.
    assists["derived_xa"] = np.where(assists["is_penalty"].fillna(False), 0.0, assists["shot_statsbomb_xg"])
    return assists


def attach_restart_context(events: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """Find the latest restart the shooting team took earlier in the same possession."""
    restarts = events.loc[
        events["type"].eq("Pass") & events["pass_type"].isin(["Corner", "Free Kick", "Throw-in"]),
        ["match_id", "period", "possession", "team", "index", "game_seconds", "pass_type", "x"],
    ].rename(columns={"index": "restart_index", "game_seconds": "restart_seconds",
                      "pass_type": "restart_type", "x": "restart_x"})

    keys = ["match_id", "period", "possession", "team"]
    shots = shots.sort_values("index")
    restarts = restarts.sort_values("restart_index")
    shots = pd.merge_asof(
        shots, restarts, left_on="index", right_on="restart_index", by=keys,
        direction="backward", allow_exact_matches=False,
    )
    shots["seconds_since_restart"] = shots["game_seconds"] - shots["restart_seconds"]
    return shots


def chance_category(row: pd.Series) -> str:
    """Assign exactly one category: dead-ball origin first, then phase, then final pass."""
    shot_type = row["shot_type"]
    if shot_type == "Penalty":
        return "Penalty"
    if shot_type == "Free Kick":
        return "Direct free kick"
    if shot_type == "Corner":
        return "Corner"

    recent_restart = (
        pd.notna(row["restart_type"]) and row["seconds_since_restart"] <= SET_PIECE_WINDOW_SECONDS
    )
    if recent_restart:
        if row["restart_type"] == "Corner":
            return "Corner"
        if row["restart_type"] == "Free Kick":
            return "Indirect free kick"
        if row["restart_type"] == "Throw-in" and row["restart_x"] >= ATTACKING_THIRD_X:
            return "Attacking throw-in"

    if row["play_pattern"] == "From Counter":
        return "Transition"
    if pd.notna(row["assist_event_id"]):
        # StatsBomb's cut-back flag is a subtype of cross, so it is checked first.
        if row["pass_cut_back"]:
            return "Cutback"
        if row["pass_cross"]:
            return "Cross"
        if row["pass_through_ball"]:
            return "Through ball"
    return "Other open play"


def attach_chance_categories(events: pd.DataFrame, shots: pd.DataFrame, assists: pd.DataFrame) -> pd.DataFrame:
    assist_context = assists[[
        "shot_id", "match_id", "assist_event_id", "assister", "pass_cross", "pass_cut_back",
        "pass_through_ball", "assist_start_x", "assist_start_y", "assist_end_x", "assist_end_y",
    ]]
    shots = shots.merge(assist_context, on=["shot_id", "match_id"], how="left", validate="one_to_one")
    for column in ["pass_cross", "pass_cut_back", "pass_through_ball"]:
        shots[column] = shots[column].fillna(False).astype(bool)
    shots = attach_restart_context(events, shots)
    shots["chance_category"] = shots.apply(chance_category, axis=1)
    return shots.sort_values(["match_id", "index"]).reset_index(drop=True)


def build_player_minutes(events: pd.DataFrame) -> pd.DataFrame:
    """Minutes per player from line-ups, substitutions and red cards."""
    period_bounds = (
        events.loc[events["period"].isin([1, 2])]
        .groupby(["match_id", "period"])["game_seconds"].max()
        .rename("period_end").reset_index()
    )
    period_bounds["period_start"] = np.where(period_bounds["period"].eq(1), 0, 45 * 60)

    stints: list[dict] = []
    for _, lineup in events.loc[events["type"].eq("Starting XI")].iterrows():
        for entry in lineup["tactics"]["lineup"]:
            stints.append({"match_id": lineup["match_id"], "team": lineup["team"],
                           "player": entry["player"]["name"], "on": 0.0, "started": True})

    substitutions = events.loc[events["type"].eq("Substitution")]
    for _, substitution in substitutions.iterrows():
        stints.append({"match_id": substitution["match_id"], "team": substitution["team"],
                       "player": substitution["substitution_replacement"],
                       "on": substitution["game_seconds"], "started": False})
    stints_frame = pd.DataFrame(stints)

    exits = substitutions[["match_id", "team", "player", "game_seconds"]]
    card_columns = [column for column in ["foul_committed_card", "bad_behaviour_card"] if column in events]
    sent_off = events.loc[
        events[card_columns].isin(["Red Card", "Second Yellow"]).any(axis=1),
        ["match_id", "team", "player", "game_seconds"],
    ]
    exits = (
        pd.concat([exits, sent_off]).groupby(["match_id", "team", "player"], as_index=False)["game_seconds"].min()
        .rename(columns={"game_seconds": "off"})
    )
    stints_frame = stints_frame.merge(exits, on=["match_id", "team", "player"], how="left")
    stints_frame["off"] = stints_frame["off"].fillna(np.inf)

    # Clip each stint to each half so stoppage time is counted and the half-time gap is not.
    played = stints_frame.merge(period_bounds, on="match_id")
    played["seconds"] = (
        np.minimum(played["off"], played["period_end"]) - np.maximum(played["on"], played["period_start"])
    ).clip(lower=0)
    per_match = played.groupby(["match_id", "team", "player", "started"], as_index=False)["seconds"].sum()

    minutes = per_match.groupby(["team", "player"], as_index=False).agg(
        minutes=("seconds", lambda seconds: seconds.sum() / 60),
        appearances=("match_id", "nunique"),
        starts=("started", "sum"),
    )
    return minutes


def build_player_creation_table(assists: pd.DataFrame, minutes: pd.DataFrame) -> pd.DataFrame:
    player_creation = (
        assists.loc[assists["derived_xa"].notna()]
        .groupby(["assisting_team", "assister"], as_index=False)
        .agg(
            direct_shot_assists=("assist_event_id", "count"),
            derived_xa=("derived_xa", "sum"),
            goal_assists=("is_goal_assist", "sum"),
        )
        .rename(columns={"assisting_team": "team", "assister": "player"})
        .merge(minutes, on=["team", "player"], how="left")
    )
    player_creation["minutes"] = player_creation["minutes"].fillna(0).round(0)
    has_minutes = player_creation["minutes"] > 0
    player_creation["derived_xa_per90"] = np.where(
        has_minutes, player_creation["derived_xa"] / player_creation["minutes"] * 90, np.nan
    )
    player_creation["shot_assists_per90"] = np.where(
        has_minutes, player_creation["direct_shot_assists"] / player_creation["minutes"] * 90, np.nan
    )
    player_creation["meets_minutes_threshold"] = player_creation["minutes"] >= MINIMUM_PLAYER_MINUTES
    return player_creation.sort_values("derived_xa", ascending=False)


def build_final_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Calculate the final league table directly from the match metadata."""
    home = matches[["home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "goals_for", "away_score": "goals_against"}
    )
    away = matches[["away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "goals_for", "home_score": "goals_against"}
    )
    table = pd.concat([home, away], ignore_index=True).dropna()
    table["win"] = table["goals_for"] > table["goals_against"]
    table["draw"] = table["goals_for"] == table["goals_against"]
    table["points"] = table["win"].astype(int) * 3 + table["draw"].astype(int)
    final_table = table.groupby("team", as_index=False).agg(
        matches_played=("team", "size"),
        points=("points", "sum"),
        wins=("win", "sum"),
        draws=("draw", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
    )
    final_table["goal_difference"] = final_table["goals_for"] - final_table["goals_against"]
    # Liga F breaks ties on head-to-head first; this uses goal difference. No two teams
    # finished level on points in 2023/24, so the order is unaffected.
    final_table = final_table.sort_values(
        ["points", "goal_difference", "goals_for"], ascending=False
    ).reset_index(drop=True)
    final_table["final_rank"] = final_table.index + 1
    final_table["points_per_match"] = final_table["points"] / final_table["matches_played"]
    return final_table


def completed_passes(events: pd.DataFrame) -> pd.DataFrame:
    passes = events.loc[events["type"].eq("Pass") & events["pass_outcome"].isna()].copy()
    passes["end_x"] = coordinate(passes["pass_end_location"], 0)
    passes["end_y"] = coordinate(passes["pass_end_location"], 1)
    passes["open_play"] = ~passes["pass_type"].isin(DEAD_BALL_PASS_TYPES)
    return passes


def classified_passes(events: pd.DataFrame) -> pd.DataFrame:
    """Completed passes flagged as progressive / into the final third / into the box (open play only)."""
    passes = completed_passes(events)
    start_distance = np.hypot(120 - passes["x"], 40 - passes["y"])
    end_distance = np.hypot(120 - passes["end_x"], 40 - passes["end_y"])
    passes["progressive"] = passes["open_play"] & ((start_distance - end_distance) >= PROGRESSIVE_DISTANCE)
    passes["into_final_third"] = (
        passes["open_play"] & (passes["x"] < ATTACKING_THIRD_X) & (passes["end_x"] >= ATTACKING_THIRD_X)
    )
    passes["into_box"] = (
        passes["open_play"] & ~in_box(passes["x"], passes["y"]) & in_box(passes["end_x"], passes["end_y"])
    )
    return passes


def build_team_passing_profile(events: pd.DataFrame, matches_played: pd.DataFrame) -> pd.DataFrame:
    """Passing volume and progression. Progression measures use open-play passes only."""
    attempted = events.loc[events["type"].eq("Pass")].groupby("team").size().rename("passes_attempted")
    passes = classified_passes(events)
    profile = passes.groupby("team").agg(
        passes_completed=("team", "size"),
        progressive_passes=("progressive", "sum"),
        passes_into_final_third=("into_final_third", "sum"),
        passes_into_box=("into_box", "sum"),
    ).join(attempted).reset_index()
    profile = profile.merge(matches_played, on="team", how="left")
    profile["pass_completion_pct"] = 100 * profile["passes_completed"] / profile["passes_attempted"]
    for column in ["passes_attempted", "passes_completed", "progressive_passes",
                   "passes_into_final_third", "passes_into_box"]:
        profile[f"{column}_per_match"] = profile[column] / profile["matches_played"]
    return profile


def defensive_events(events: pd.DataFrame) -> pd.DataFrame:
    """Defensive actions, flagged for pressures and successful high regains."""
    defensive = events.loc[
        events["type"].isin(["Pressure", "Interception", "Ball Recovery", "Block", "Clearance", "Duel"])
    ].copy()
    failed_recovery = flag(defensive["ball_recovery_recovery_failure"])
    successful_interception = defensive["interception_outcome"].isin(["Won", "Success In Play", "Success Out"])
    defensive["is_regain"] = (
        (defensive["type"].eq("Ball Recovery") & ~failed_recovery)
        | (defensive["type"].eq("Interception") & successful_interception)
    )
    defensive["is_high_regain"] = defensive["is_regain"] & (defensive["x"] >= ATTACKING_THIRD_X)
    defensive["is_pressure"] = defensive["type"].eq("Pressure")
    return defensive


def build_team_defensive_profile(
    matches: pd.DataFrame,
    events: pd.DataFrame,
    shots: pd.DataFrame,
    matches_played: pd.DataFrame,
) -> pd.DataFrame:
    """Defensive workload, opponent chance quality and successful high regains."""
    opponents = pd.concat(
        [
            matches[["match_id", "home_team", "away_team"]].rename(columns={"home_team": "team", "away_team": "opponent"}),
            matches[["match_id", "away_team", "home_team"]].rename(columns={"away_team": "team", "home_team": "opponent"}),
        ],
        ignore_index=True,
    )
    conceded = (
        shots.merge(opponents, on=["match_id", "team"], how="left")
        .groupby("opponent", as_index=False)
        .agg(
            shots_conceded=("shot_id", "size"),
            xg_conceded=("shot_statsbomb_xg", "sum"),
            npxg_conceded=("npxg", "sum"),
            shot_goals_conceded=("is_goal_from_shot", "sum"),
        )
        .rename(columns={"opponent": "team"})
    )

    defensive = defensive_events(events)
    defensive_profile = defensive.groupby("team", as_index=False).agg(
        defensive_actions=("team", "size"),
        pressures=("is_pressure", "sum"),
        high_regains=("is_high_regain", "sum"),
    )
    profile = matches_played.merge(conceded, on="team", how="left").merge(defensive_profile, on="team", how="left").fillna(0)
    for column in ["shots_conceded", "xg_conceded", "npxg_conceded", "shot_goals_conceded",
                   "defensive_actions", "pressures", "high_regains"]:
        profile[f"{column}_per_match"] = profile[column] / profile["matches_played"]
    return profile


def build_team_profile(
    shots: pd.DataFrame,
    assists: pd.DataFrame,
    player_creation: pd.DataFrame,
    events: pd.DataFrame,
    final_table: pd.DataFrame,
    passing_profile: pd.DataFrame,
    defensive_profile: pd.DataFrame,
) -> pd.DataFrame:
    matches_played = final_table[["team", "matches_played"]]
    own_goals_for = events.loc[events["type"].eq("Own Goal For")].groupby("team").size()

    baseline = (
        shots.groupby("team", as_index=False)
        .agg(
            shots=("shot_id", "count"),
            non_penalty_shots=("is_penalty", lambda penalty: (~penalty).sum()),
            shot_goals=("is_goal_from_shot", "sum"),
            xg=("shot_statsbomb_xg", "sum"),
            npxg=("npxg", "sum"),
        )
        .merge(matches_played, on="team", how="outer")
        .fillna({"shots": 0, "non_penalty_shots": 0, "shot_goals": 0, "xg": 0, "npxg": 0})
    )
    baseline["own_goals_for"] = baseline["team"].map(own_goals_for).fillna(0).astype(int)
    baseline["goals"] = baseline["shot_goals"] + baseline["own_goals_for"]
    baseline["goals_minus_xg"] = baseline["shot_goals"] - baseline["xg"]
    baseline["xg_per_match"] = baseline["xg"] / baseline["matches_played"]
    baseline["npxg_per_match"] = baseline["npxg"] / baseline["matches_played"]
    baseline["shots_per_match"] = baseline["shots"] / baseline["matches_played"]
    baseline["non_penalty_shots_per_match"] = baseline["non_penalty_shots"] / baseline["matches_played"]
    baseline["xg_per_shot"] = baseline["xg"] / baseline["shots"]
    baseline["npxg_per_shot"] = baseline["npxg"] / baseline["non_penalty_shots"]

    category_counts = pd.crosstab(shots["team"], shots["chance_category"]).reindex(
        columns=CHANCE_CATEGORIES, fill_value=0
    )
    category_share = category_counts.div(category_counts.sum(axis=1), axis=0).add_suffix("_shot_share").reset_index()

    team_xa = (
        assists.groupby("assisting_team", as_index=False)["derived_xa"].sum()
        .rename(columns={"assisting_team": "team"})
    )
    top_two_xa = (
        player_creation.sort_values(["team", "derived_xa"], ascending=[True, False])
        .groupby("team").head(2)
        .groupby("team", as_index=False)["derived_xa"].sum()
        .rename(columns={"derived_xa": "top_two_derived_xa"})
    )

    profile = (
        baseline.merge(team_xa, on="team", how="left")
        .merge(category_share, on="team", how="left")
        .merge(top_two_xa, on="team", how="left")
        .merge(final_table.drop(columns="matches_played"), on="team", how="left")
        .merge(passing_profile.drop(columns="matches_played"), on="team", how="left")
        .merge(defensive_profile.drop(columns="matches_played"), on="team", how="left")
        .fillna(0)
    )
    profile["derived_xa_per_match"] = profile["derived_xa"] / profile["matches_played"]
    profile["top_two_creator_xa_share"] = np.where(
        profile["derived_xa"] > 0, profile["top_two_derived_xa"] / profile["derived_xa"], np.nan
    )
    return profile.sort_values("final_rank")


def short_team_name(team: str) -> str:
    return team.replace(" WFC", "").replace(" CF W", "").replace(" Femenino", "").replace(" W", "")


def build_team_styles(
    team_profile: pd.DataFrame,
    excluded_team: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Exploratory PCA map, K-means groups and an auditable loading table.

    PCA axes have no intrinsic football meaning and an arbitrary sign. Signs are
    fixed so PC1 rises with points per match and each PC2's largest loading is
    positive; the labels shown in the app are written from the loadings.
    """
    features = list(STYLE_FEATURES)
    analysis_profile = team_profile.copy()
    if excluded_team is not None:
        if excluded_team not in set(analysis_profile["team"]):
            raise ValueError(f"{excluded_team} is not in the team profile.")
        analysis_profile = analysis_profile.loc[analysis_profile["team"].ne(excluded_team)].copy()

    scaled = StandardScaler().fit_transform(analysis_profile[features].astype(float))
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(scaled)
    loadings_matrix = pca.components_.copy()

    if np.corrcoef(components[:, 0], analysis_profile["points_per_match"])[0, 1] < 0:
        components[:, 0] *= -1
        loadings_matrix[0] *= -1
    if loadings_matrix[1][np.argmax(np.abs(loadings_matrix[1]))] < 0:
        components[:, 1] *= -1
        loadings_matrix[1] *= -1

    clusters = KMeans(n_clusters=3, n_init=50, random_state=42).fit_predict(scaled)
    styles = analysis_profile[["team", "final_rank", "points_per_match"]].copy()
    styles["pc1"] = components[:, 0]
    styles["pc2"] = components[:, 1]

    # Number groups by average points so colours and names are stable between runs.
    cluster_points = styles.assign(cluster=clusters).groupby("cluster")["points_per_match"].mean()
    order = {cluster: rank for rank, cluster in enumerate(cluster_points.sort_values(ascending=False).index, start=1)}
    styles["style_cluster"] = [order[cluster] for cluster in clusters]
    sizes = styles["style_cluster"].value_counts()
    shared_groups = sorted(group for group in sizes.index if sizes[group] > 1)
    shared_names = {
        1: ["Shared profile"],
        2: ["Higher-control group", "Lower-control group"],
        3: ["Highest-control group", "Middle-control group", "Lower-control group"],
    }[len(shared_groups)]
    group_names = dict(zip(shared_groups, shared_names))
    styles["style_label"] = [
        f"{short_team_name(team)}: a group of its own" if sizes[group] == 1 else group_names[group]
        for team, group in zip(styles["team"], styles["style_cluster"])
    ]

    loadings = pd.DataFrame(
        {
            "feature": features * 2,
            "feature_label": [STYLE_FEATURES[feature] for feature in features] * 2,
            "component": ["PC1"] * len(features) + ["PC2"] * len(features),
            "loading": np.concatenate(loadings_matrix),
            "explained_variance_ratio": np.repeat(pca.explained_variance_ratio_, len(features)),
        }
    )
    loadings["absolute_loading"] = loadings["loading"].abs()
    loadings = loadings.sort_values(["component", "absolute_loading"], ascending=[True, False])
    summary = pd.DataFrame(
        [{
            "model": "without_excluded_team" if excluded_team else "all_teams",
            "excluded_team": excluded_team or "",
            "teams_included": len(analysis_profile),
            "pc1_explained_variance_ratio": pca.explained_variance_ratio_[0],
            "pc2_explained_variance_ratio": pca.explained_variance_ratio_[1],
            "two_component_variance_ratio": pca.explained_variance_ratio_.sum(),
            "kmeans_silhouette": silhouette_score(scaled, clusters),
            "pc1_points_correlation": np.corrcoef(components[:, 0], analysis_profile["points_per_match"])[0, 1],
        }]
    )
    return styles.sort_values("final_rank"), loadings, summary


def build_passing_network(events: pd.DataFrame, minutes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Season passing network of each team's most frequent starters.

    Node positions are each player's mean location across the passes they made
    and the passes they received, so one player has one position.
    """
    core = (
        minutes.sort_values(["team", "starts", "minutes"], ascending=[True, False, False])
        .groupby("team").head(NETWORK_PLAYERS)[["team", "player", "starts", "minutes"]]
    )
    passes = completed_passes(events)
    passes = passes.loc[passes["player"].notna() & passes["pass_recipient"].notna()]

    made = passes[["team", "player", "x", "y"]]
    received = passes[["team", "pass_recipient", "end_x", "end_y"]].rename(
        columns={"pass_recipient": "player", "end_x": "x", "end_y": "y"}
    )
    touches = pd.concat([made, received], ignore_index=True)
    positions = touches.groupby(["team", "player"], as_index=False).agg(
        x=("x", "mean"), y=("y", "mean"), involvements=("x", "size")
    )
    nodes = core.merge(positions, on=["team", "player"], how="left")

    edges = passes.merge(core[["team", "player"]], on=["team", "player"]).merge(
        core[["team", "player"]].rename(columns={"player": "pass_recipient"}), on=["team", "pass_recipient"]
    )
    edges = edges.groupby(["team", "player", "pass_recipient"], as_index=False).size().rename(columns={"size": "passes"})
    edges = edges.loc[edges["passes"] >= NETWORK_MIN_PASSES].sort_values(["team", "passes"], ascending=[True, False])
    return nodes, edges


POSITION_GROUPS = {
    "Goalkeeper": "Goalkeeper",
    "Back": "Defender", "Center Back": "Defender", "Wing Back": "Defender",
    "Defensive Midfield": "Midfielder", "Center Midfield": "Midfielder", "Midfield": "Midfielder",
    "Attacking Midfield": "Midfielder",
    "Wing": "Forward", "Forward": "Forward", "Striker": "Forward",
}

# Per-90 measures used to describe and compare outfield players.
PROFILE_FEATURES = {
    "npxg_per90": "Non-penalty xG",
    "shots_per90": "Shots",
    "derived_xa_per90": "Derived xA",
    "shot_assists_per90": "Created shots",
    "pass_xt_per90": "xT from passes",
    "carry_xt_per90": "xT from carries",
    "progressive_passes_per90": "Progressive passes",
    "passes_into_box_per90": "Passes into the box",
    "passes_completed_per90": "Completed passes",
    "dribbles_per90": "Successful dribbles",
    "regains_per90": "Ball regains",
    "pressures_per90": "Pressures",
}


def position_group(position: str) -> str:
    """Map StatsBomb's detailed positions to four groups (longest matching suffix wins)."""
    for suffix in sorted(POSITION_GROUPS, key=len, reverse=True):
        if position.endswith(suffix):
            return POSITION_GROUPS[suffix]
    return "Midfielder"


def build_player_profiles(
    events: pd.DataFrame,
    shots: pd.DataFrame,
    player_creation: pd.DataFrame,
    minutes: pd.DataFrame,
    xt_players: pd.DataFrame,
) -> pd.DataFrame:
    """Per-90 profiles of outfield players with enough minutes, plus z-scores and percentiles."""
    keys = ["team", "player"]
    profile = minutes.loc[minutes["minutes"] >= MINIMUM_PLAYER_MINUTES, keys + ["minutes", "appearances", "starts"]].copy()

    on_ball = events.loc[events["player"].notna() & events["position"].notna() & ~events["position"].eq("Substitute")]
    main_position = on_ball.groupby(keys)["position"].agg(lambda values: values.value_counts().index[0]).rename("position")
    profile = profile.merge(main_position.reset_index(), on=keys, how="left")
    profile["position_group"] = profile["position"].fillna("").map(position_group)
    profile = profile.loc[profile["position_group"].ne("Goalkeeper")].copy()

    open_shots = shots.loc[~shots["is_penalty"]].rename(columns={"shooter": "player"})
    shooting = open_shots.groupby(keys).agg(npxg=("shot_statsbomb_xg", "sum"), np_shots=("shot_id", "size"))
    creation = player_creation.set_index(keys)[["derived_xa", "direct_shot_assists"]]
    passes = classified_passes(events).groupby(keys).agg(
        passes_completed=("team", "size"), progressive_passes=("progressive", "sum"), passes_into_box=("into_box", "sum")
    )
    dribbles = (
        events.loc[events["type"].eq("Dribble") & events["dribble_outcome"].eq("Complete")].groupby(keys).size().rename("dribbles")
    )
    defensive = defensive_events(events).groupby(keys).agg(regains=("is_regain", "sum"), pressures=("is_pressure", "sum"))
    threat = xt_players.set_index(keys)[["pass_xt", "carry_xt"]]

    totals = pd.concat([shooting, creation, passes, dribbles, defensive, threat], axis=1).reset_index()
    profile = profile.merge(totals, on=keys, how="left").fillna(0)
    per90 = profile["minutes"] / 90
    for source, target in [
        ("npxg", "npxg_per90"), ("np_shots", "shots_per90"), ("derived_xa", "derived_xa_per90"),
        ("direct_shot_assists", "shot_assists_per90"), ("pass_xt", "pass_xt_per90"), ("carry_xt", "carry_xt_per90"),
        ("progressive_passes", "progressive_passes_per90"), ("passes_into_box", "passes_into_box_per90"),
        ("passes_completed", "passes_completed_per90"), ("dribbles", "dribbles_per90"),
        ("regains", "regains_per90"), ("pressures", "pressures_per90"),
    ]:
        profile[target] = profile[source] / per90

    for feature in PROFILE_FEATURES:
        values = profile[feature]
        profile[f"z_{feature}"] = (values - values.mean()) / values.std()
        profile[f"pct_{feature}"] = profile.groupby("position_group")[feature].rank(pct=True)
    profile["total_xt_per90"] = profile["pass_xt_per90"] + profile["carry_xt_per90"]
    return profile.sort_values("total_xt_per90", ascending=False)


def build_validation_report(
    matches: pd.DataFrame,
    events: pd.DataFrame,
    shots: pd.DataFrame,
    assists: pd.DataFrame,
    minutes: pd.DataFrame,
    failed_match_ids: list[int],
) -> pd.DataFrame:
    match_score_total = int(matches["home_score"].sum() + matches["away_score"].sum())
    shot_goals = int(shots["is_goal_from_shot"].sum())
    own_goal_for_events = int(events["type"].eq("Own Goal For").sum())

    match_length = (
        events.loc[events["type"].eq("Half End")].groupby(["match_id", "period"])["game_seconds"].max()
        .reset_index()
    )
    match_length["seconds"] = match_length["game_seconds"] - np.where(match_length["period"].eq(1), 0, 45 * 60)
    total_match_minutes = match_length["seconds"].sum() / 60
    expected_player_minutes = total_match_minutes * 2 * 11  # Two teams of eleven, before red cards.

    possession_origin = shots["play_pattern"].isin(["From Corner", "From Free Kick", "From Throw In"])
    rows = [
        ("expected_matches", EXPECTED_MATCHES),
        ("matches_in_metadata", int(matches["match_id"].nunique())),
        ("matches_with_cached_events", int(events["match_id"].nunique())),
        ("teams_in_match_metadata", int(pd.concat([matches["home_team"], matches["away_team"]]).nunique())),
        ("unreadable_cached_event_files", len(failed_match_ids)),
        ("total_shots", int(len(shots))),
        ("total_xg", round(float(shots["shot_statsbomb_xg"].sum()), 4)),
        ("total_npxg", round(float(shots["npxg"].sum()), 4)),
        ("shot_goals", shot_goals),
        ("own_goal_for_events", own_goal_for_events),
        ("goals_from_events", shot_goals + own_goal_for_events),
        ("goals_from_match_metadata", match_score_total),
        ("shots_with_one_chance_category", int(shots["chance_category"].isin(CHANCE_CATEGORIES).sum())),
        ("direct_assists_unmatched_to_shot", int(assists["shot_statsbomb_xg"].isna().sum())),
        ("set_piece_shot_share", round(float(shots["chance_category"].isin(
            ["Penalty", "Direct free kick", "Corner", "Indirect free kick", "Attacking throw-in"]).mean()), 4)),
        ("shots_in_possessions_started_by_restart_share", round(float(possession_origin.mean()), 4)),
        ("player_minutes_vs_eleven_per_side", round(float(minutes["minutes"].sum() / expected_player_minutes), 4)),
    ]
    return pd.DataFrame(rows, columns=["check", "value"])


def assert_validation(validation: pd.DataFrame, shots: pd.DataFrame) -> None:
    """Stop the pipeline instead of publishing numbers that do not reconcile."""
    checks = dict(zip(validation["check"], validation["value"]))
    problems = []
    if checks["goals_from_events"] != checks["goals_from_match_metadata"]:
        problems.append("event goals do not match the match scores")
    if checks["shots_with_one_chance_category"] != len(shots):
        problems.append("some shots have no chance category")
    if checks["matches_with_cached_events"] != EXPECTED_MATCHES:
        problems.append("not every match has events")
    if not 0.98 <= checks["player_minutes_vs_eleven_per_side"] <= 1.0:
        problems.append("player minutes do not reconcile with match length")
    if problems:
        raise RuntimeError("Validation failed: " + "; ".join(problems))


def main() -> None:
    args = parse_arguments()
    project_dir = args.project_dir.expanduser().resolve()
    raw_events_dir = project_dir / "data" / "raw" / "events"
    processed_dir = project_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    matches_path = processed_dir / "liga_f_matches.csv"
    if not matches_path.exists():
        raise FileNotFoundError(f"{matches_path} does not exist. Run src/download_liga_f.py first.")
    matches = pd.read_csv(matches_path)
    require_columns(matches, ["match_id", "home_team", "away_team", "home_score", "away_score"], "match metadata")

    events, failed_match_ids = load_cached_events(raw_events_dir, processed_dir / "liga_f_events_raw.pkl")
    events = prepare_events(events)
    shots = build_shots(events)
    assists = build_direct_assists(events, shots)
    shots = attach_chance_categories(events, shots, assists)
    minutes = build_player_minutes(events)
    player_creation = build_player_creation_table(assists, minutes)
    final_table = build_final_table(matches)
    matches_played = final_table[["team", "matches_played"]]
    passing_profile = build_team_passing_profile(events, matches_played)
    defensive_profile = build_team_defensive_profile(matches, events, shots, matches_played)
    team_profile = build_team_profile(
        shots, assists, player_creation, events, final_table, passing_profile, defensive_profile
    )
    team_styles, pca_loadings, pca_summary = build_team_styles(team_profile)
    styles_sensitivity, loadings_sensitivity, summary_sensitivity = build_team_styles(
        team_profile, excluded_team=SENSITIVITY_EXCLUDED_TEAM
    )
    network_nodes, network_edges = build_passing_network(events, minutes)
    threat = run_expected_threat(events)
    player_profiles = build_player_profiles(events, shots, player_creation, minutes, threat["xt_players"])
    xg_results = run_xg_analysis(events, shots)
    validation = build_validation_report(matches, events, shots, assists, minutes, failed_match_ids)
    assert_validation(validation, shots)

    shot_map = shots[["team", "shooter", "minute", "shot_x", "shot_y", "shot_statsbomb_xg",
                      "is_goal_from_shot", "chance_category"]]
    outputs = {
        "liga_f_shots_enriched.csv": shots.drop(columns=["index", "restart_index"]),
        "liga_f_shot_map.csv": shot_map,
        "liga_f_direct_assists.csv": assists,
        "liga_f_player_minutes.csv": minutes,
        "liga_f_player_derived_xa.csv": player_creation,
        "liga_f_team_attacking_profile.csv": team_profile,
        "liga_f_final_table.csv": final_table,
        "liga_f_team_passing_profile.csv": passing_profile,
        "liga_f_team_defensive_profile.csv": defensive_profile,
        "liga_f_team_styles.csv": team_styles,
        "liga_f_pca_loadings.csv": pca_loadings,
        "liga_f_team_styles_without_barcelona.csv": styles_sensitivity,
        "liga_f_pca_loadings_without_barcelona.csv": loadings_sensitivity,
        "liga_f_pca_sensitivity_summary.csv": pd.concat([pca_summary, summary_sensitivity], ignore_index=True),
        "liga_f_passing_network_nodes.csv": network_nodes,
        "liga_f_passing_network_edges.csv": network_edges,
        "liga_f_validation_report.csv": validation,
        "liga_f_player_profiles.csv": player_profiles,
        "liga_f_xt_grid.csv": threat["xt_grid"],
        "liga_f_xt_teams.csv": threat["xt_teams"],
        **{f"liga_f_{name}.csv": frame for name, frame in xg_results.items()},
    }
    for filename, frame in outputs.items():
        frame.to_csv(processed_dir / filename, index=False)

    # The static dashboard reads these lightweight files; keep them in sync on every run.
    dashboard_files = [
        "liga_f_shot_map.csv",
        "liga_f_player_derived_xa.csv",
        "liga_f_team_attacking_profile.csv",
        "liga_f_team_styles.csv",
        "liga_f_pca_loadings.csv",
        "liga_f_team_styles_without_barcelona.csv",
        "liga_f_pca_loadings_without_barcelona.csv",
        "liga_f_pca_sensitivity_summary.csv",
        "liga_f_passing_network_nodes.csv",
        "liga_f_passing_network_edges.csv",
        "liga_f_validation_report.csv",
        "liga_f_player_profiles.csv",
        "liga_f_xt_grid.csv",
        "liga_f_xg_model_comparison.csv",
        "liga_f_xg_calibration.csv",
        "liga_f_xg_gap_breakdown.csv",
        "liga_f_xg_team_finishing.csv",
        "liga_f_xg_shooter_finishing.csv",
        "liga_f_xg_goalkeeper_faced.csv",
        "liga_f_xg_feature_effects.csv",
    ]
    dashboard_data_dir = project_dir / "app" / "data"
    dashboard_data_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in dashboard_data_dir.glob("*.csv"):
        if stale_file.name not in dashboard_files:
            stale_file.unlink()
    for filename in dashboard_files:
        shutil.copy2(processed_dir / filename, dashboard_data_dir / filename)

    print(f"\n{COMPETITION_NAME} pipeline complete.\n")
    print(validation.to_string(index=False))
    print("\nChance categories:")
    print(shots["chance_category"].value_counts().to_string())
    eligible = player_creation.loc[player_creation["meets_minutes_threshold"]]
    print(f"\nTop 10 players by derived xA per 90 (minimum {MINIMUM_PLAYER_MINUTES} minutes):")
    print(eligible.sort_values("derived_xa_per90", ascending=False).head(10)[
        ["player", "team", "minutes", "derived_xa", "derived_xa_per90"]].to_string(index=False))
    print("\nxG models (out-of-fold, non-penalty shots):")
    print(xg_results["xg_model_comparison"][["label", "log_loss", "roc_auc", "predicted_goals", "actual_goals"]].round(3).to_string(index=False))
    print("\nTop 10 players by open-play xT per 90:")
    print(player_profiles.head(10)[["player", "team", "position_group", "total_xt_per90", "derived_xa_per90"]].round(3).to_string(index=False))
    print(f"\nSaved outputs to: {processed_dir}")


if __name__ == "__main__":
    main()
