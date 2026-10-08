"""Expected threat (xT): credit every pass and carry, not just the final pass.

Follows Karun Singh's formulation. The pitch is split into a grid; for each cell, xT is
the probability that possession there ends in a goal within the next few actions:

    xT(cell) = P(shoot) * P(goal | shot) + P(move) * sum over targets T(cell -> target) * xT(target)

It is solved by iteration. A successful open-play pass or carry is worth
xT(end cell) - xT(start cell); failed moves contribute nothing to the transition matrix.
Restarts are excluded because their value comes from the set-piece situation, not the move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


GRID_X, GRID_Y = 16, 12
PITCH_X, PITCH_Y = 120.0, 80.0
ITERATIONS = 50
DEAD_BALL_PASS_TYPES = {"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"}


def cell_index(x: pd.Series, y: pd.Series) -> pd.Series:
    column = np.clip((x / PITCH_X * GRID_X).astype(int), 0, GRID_X - 1)
    row = np.clip((y / PITCH_Y * GRID_Y).astype(int), 0, GRID_Y - 1)
    return row * GRID_X + column


def open_play_moves(events: pd.DataFrame) -> pd.DataFrame:
    """All open-play passes and carries with start/end coordinates and success."""
    passes = events.loc[events["type"].eq("Pass") & ~events["pass_type"].isin(DEAD_BALL_PASS_TYPES)].copy()
    passes["end"] = passes["pass_end_location"]
    passes["successful"] = passes["pass_outcome"].isna()
    passes["move_type"] = "pass"
    carries = events.loc[events["type"].eq("Carry")].copy()
    carries["end"] = carries["carry_end_location"]
    carries["successful"] = True  # A carry that loses the ball is followed by a separate event.
    carries["move_type"] = "carry"
    moves = pd.concat([passes, carries], ignore_index=True)
    moves["end_x"] = moves["end"].map(lambda value: value[0] if isinstance(value, list) else np.nan)
    moves["end_y"] = moves["end"].map(lambda value: value[1] if isinstance(value, list) else np.nan)
    moves = moves.dropna(subset=["x", "y", "end_x", "end_y"])
    moves["start_cell"] = cell_index(moves["x"], moves["y"])
    moves["end_cell"] = cell_index(moves["end_x"], moves["end_y"])
    return moves[["match_id", "team", "player", "move_type", "successful", "x", "y", "end_x", "end_y", "start_cell", "end_cell"]]


def fit_expected_threat(events: pd.DataFrame, moves: pd.DataFrame) -> np.ndarray:
    cells = GRID_X * GRID_Y
    shots = events.loc[events["type"].eq("Shot") & events["shot_type"].eq("Open Play")]
    shot_cells = cell_index(shots["x"], shots["y"])
    shot_count = np.bincount(shot_cells, minlength=cells).astype(float)
    goal_count = np.bincount(shot_cells[shots["shot_outcome"].eq("Goal")], minlength=cells).astype(float)
    move_count = np.bincount(moves["start_cell"], minlength=cells).astype(float)

    actions = shot_count + move_count
    shoot_probability = np.divide(shot_count, actions, out=np.zeros(cells), where=actions > 0)
    move_probability = np.divide(move_count, actions, out=np.zeros(cells), where=actions > 0)
    goal_probability = np.divide(goal_count, shot_count, out=np.zeros(cells), where=shot_count > 0)

    successful = moves.loc[moves["successful"]]
    transitions = np.zeros((cells, cells))
    np.add.at(transitions, (successful["start_cell"].to_numpy(), successful["end_cell"].to_numpy()), 1)
    transitions = np.divide(transitions, move_count[:, None], out=np.zeros_like(transitions), where=move_count[:, None] > 0)

    threat = np.zeros(cells)
    for _ in range(ITERATIONS):
        threat = shoot_probability * goal_probability + move_probability * (transitions @ threat)
    return threat


def value_moves(moves: pd.DataFrame, threat: np.ndarray) -> pd.DataFrame:
    valued = moves.loc[moves["successful"]].copy()
    valued["xt_added"] = threat[valued["end_cell"].to_numpy()] - threat[valued["start_cell"].to_numpy()]
    return valued


def grid_table(threat: np.ndarray) -> pd.DataFrame:
    cells = np.arange(GRID_X * GRID_Y)
    return pd.DataFrame({
        "column": cells % GRID_X, "row": cells // GRID_X,
        "x_center": (cells % GRID_X + 0.5) * PITCH_X / GRID_X,
        "y_center": (cells // GRID_X + 0.5) * PITCH_Y / GRID_Y,
        "xt": threat,
    })


def player_threat(valued: pd.DataFrame) -> pd.DataFrame:
    by_type = valued.pivot_table(index=["team", "player"], columns="move_type", values="xt_added", aggfunc="sum", fill_value=0.0)
    by_type = by_type.rename(columns={"pass": "pass_xt", "carry": "carry_xt"}).reset_index()
    by_type.columns.name = None
    by_type["total_xt"] = by_type["pass_xt"] + by_type["carry_xt"]
    return by_type


def team_threat(valued: pd.DataFrame) -> pd.DataFrame:
    return valued.groupby("team", as_index=False).agg(open_play_xt=("xt_added", "sum"))


def run_expected_threat(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    moves = open_play_moves(events)
    threat = fit_expected_threat(events, moves)
    valued = value_moves(moves, threat)
    return {"xt_grid": grid_table(threat), "xt_players": player_threat(valued), "xt_teams": team_threat(valued)}
