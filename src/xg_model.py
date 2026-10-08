"""Where does Liga F's goals-above-xG gap come from, and does a league-specific xG model fix it?

Two questions, answered on non-penalty shots only:
1. Is the gap real (larger than chance), and where is it concentrated?
2. A model trained on Liga F shots, with freeze-frame features, is compared with StatsBomb's
   xG and with StatsBomb's xG recalibrated to this league. Every model is evaluated out of
   sample, with folds grouped by match so no match is in both training and test data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


GOAL_X, GOAL_Y = 120.0, 40.0
POST_LOW, POST_HIGH = 36.0, 44.0
FOLDS = 5
CALIBRATION_BINS = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 1.0]

FEATURE_LABELS = {
    "distance": "Distance to goal",
    "angle": "Goal angle",
    "is_header": "Header",
    "is_other_body_part": "Other body part",
    "is_direct_free_kick": "Direct free kick",
    "first_time": "First-time shot",
    "one_on_one": "One-on-one",
    "open_goal": "Open goal",
    "under_pressure": "Under pressure",
    "is_volley": "Volley or half-volley",
    "is_lob": "Lob",
    "from_cross": "Assisted by a cross",
    "from_cutback": "Assisted by a cutback",
    "from_through_ball": "Assisted by a through ball",
    "from_set_piece": "From a set piece",
    "from_transition": "From a counter-attack",
    "defenders_in_cone": "Defenders between ball and goal",
    "pressure_within_2y": "Opponents within 2 yards",
    "gk_distance_to_goal": "Goalkeeper distance from goal centre",
    "gk_in_cone": "Goalkeeper between ball and goal",
}
# Only penalties lack freeze frames, and they are excluded, so missing values are rare;
# they are filled with the median rather than given their own feature.
DISTANCE_BANDS = [0, 6, 12, 18, 25, 120]
DISTANCE_LABELS = ["0–6 yd", "6–12 yd", "12–18 yd", "18–25 yd", "25+ yd"]


def in_shot_cone(px: float, py: float, sx: float, sy: float) -> bool:
    """Is (px, py) inside the triangle between the shot location and the two posts?"""
    def side(ax, ay, bx, by, cx, cy):
        return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)
    d1 = side(px, py, sx, sy, GOAL_X, POST_LOW)
    d2 = side(px, py, GOAL_X, POST_LOW, GOAL_X, POST_HIGH)
    d3 = side(px, py, GOAL_X, POST_HIGH, sx, sy)
    has_negative = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_positive = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_negative and has_positive)


def freeze_frame_features(frame: object, sx: float, sy: float) -> dict:
    if not isinstance(frame, list) or not frame:
        return {"defenders_in_cone": np.nan, "pressure_within_2y": np.nan, "gk_distance_to_goal": np.nan,
                "gk_in_cone": np.nan, "missing_freeze_frame": 1, "goalkeeper": None}
    defenders_in_cone = pressure = 0
    gk_distance, gk_in_cone, goalkeeper = np.nan, 0, None
    for player in frame:
        if player.get("teammate"):
            continue
        px, py = player["location"][:2]
        if player.get("position", {}).get("name") == "Goalkeeper":
            gk_distance = float(np.hypot(GOAL_X - px, GOAL_Y - py))
            gk_in_cone = int(in_shot_cone(px, py, sx, sy))
            goalkeeper = player["player"]["name"]
            continue
        defenders_in_cone += in_shot_cone(px, py, sx, sy)
        pressure += np.hypot(px - sx, py - sy) <= 2
    return {"defenders_in_cone": defenders_in_cone, "pressure_within_2y": pressure,
            "gk_distance_to_goal": gk_distance, "gk_in_cone": gk_in_cone,
            "missing_freeze_frame": 0, "goalkeeper": goalkeeper}


def build_shot_features(events: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """Model features for every non-penalty shot. `shots` must carry chance categories."""
    raw = events.loc[events["type"].eq("Shot"), [
        "id", "shot_freeze_frame", "shot_first_time", "shot_one_on_one", "shot_open_goal",
        "under_pressure", "shot_technique", "shot_body_part",
    ]].rename(columns={"id": "shot_id"})
    data = shots.loc[~shots["is_penalty"]].merge(raw, on="shot_id", how="left", suffixes=("", "_raw"))

    dx, dy = GOAL_X - data["shot_x"], GOAL_Y - data["shot_y"]
    data["distance"] = np.hypot(dx, dy)
    # Angle subtended by the goal mouth, in radians.
    data["angle"] = np.abs(np.arctan2(POST_HIGH - data["shot_y"], dx) - np.arctan2(POST_LOW - data["shot_y"], dx))
    body = data["shot_body_part_raw"] if "shot_body_part_raw" in data else data["shot_body_part"]
    data["is_header"] = body.eq("Head").astype(int)
    data["is_other_body_part"] = body.eq("Other").astype(int)
    data["is_direct_free_kick"] = data["shot_type"].eq("Free Kick").astype(int)
    for source, target in [("shot_first_time", "first_time"), ("shot_one_on_one", "one_on_one"),
                           ("shot_open_goal", "open_goal"), ("under_pressure", "under_pressure")]:
        data[target] = data[source].eq(True).astype(int)
    data["is_volley"] = data["shot_technique"].isin(["Volley", "Half Volley"]).astype(int)
    data["is_lob"] = data["shot_technique"].eq("Lob").astype(int)
    category = data["chance_category"]
    data["from_cross"] = category.eq("Cross").astype(int)
    data["from_cutback"] = category.eq("Cutback").astype(int)
    data["from_through_ball"] = category.eq("Through ball").astype(int)
    data["from_set_piece"] = category.isin(["Corner", "Indirect free kick", "Attacking throw-in"]).astype(int)
    data["from_transition"] = category.eq("Transition").astype(int)

    frame_features = pd.DataFrame([
        freeze_frame_features(frame, sx, sy)
        for frame, sx, sy in zip(data["shot_freeze_frame"], data["shot_x"], data["shot_y"])
    ], index=data.index)
    data = pd.concat([data, frame_features], axis=1)
    for column in ["defenders_in_cone", "pressure_within_2y", "gk_distance_to_goal", "gk_in_cone"]:
        data[column] = data[column].fillna(data[column].median())
    data["goal"] = data["is_goal_from_shot"].astype(int)
    data["distance_band"] = pd.cut(data["distance"], DISTANCE_BANDS, labels=DISTANCE_LABELS, right=False)
    return data.drop(columns=["shot_freeze_frame"])


def logit(p: pd.Series) -> np.ndarray:
    p = np.clip(p.astype(float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).to_numpy().reshape(-1, 1)


def out_of_fold_predictions(data: pd.DataFrame) -> pd.DataFrame:
    features = list(FEATURE_LABELS)
    X, y, groups = data[features].astype(float), data["goal"].to_numpy(), data["match_id"].to_numpy()
    predictions = pd.DataFrame(index=data.index, columns=["logistic", "boosting", "statsbomb_recalibrated"], dtype=float)
    for train, test in GroupKFold(n_splits=FOLDS).split(X, y, groups):
        logistic = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        logistic.fit(X.iloc[train], y[train])
        predictions.iloc[test, 0] = logistic.predict_proba(X.iloc[test])[:, 1]

        boosting = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=40, l2_regularization=1.0, random_state=42
        )
        boosting.fit(X.iloc[train], y[train])
        predictions.iloc[test, 1] = boosting.predict_proba(X.iloc[test])[:, 1]

        # Keep StatsBomb's ranking of chances, but re-fit its level and slope to Liga F.
        recalibration = LogisticRegression(max_iter=1000)
        recalibration.fit(logit(data["shot_statsbomb_xg"].iloc[train]), y[train])
        predictions.iloc[test, 2] = recalibration.predict_proba(logit(data["shot_statsbomb_xg"].iloc[test]))[:, 1]
    return predictions


def evaluate(data: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    models = {
        "statsbomb": ("StatsBomb xG", data["shot_statsbomb_xg"]),
        "statsbomb_recalibrated": ("StatsBomb xG, recalibrated to Liga F", predictions["statsbomb_recalibrated"]),
        "logistic": ("Liga F model: logistic regression", predictions["logistic"]),
        "boosting": ("Liga F model: gradient boosting", predictions["boosting"]),
    }
    baseline = np.full(len(data), data["goal"].mean())
    baseline_loss = log_loss(data["goal"], baseline)
    rows = []
    for key, (label, values) in models.items():
        values = values.astype(float).clip(1e-4, 1 - 1e-4)
        loss = log_loss(data["goal"], values)
        rows.append({
            "model": key, "label": label,
            "log_loss": loss,
            "log_loss_skill": 1 - loss / baseline_loss,
            "brier": brier_score_loss(data["goal"], values),
            "roc_auc": roc_auc_score(data["goal"], values),
            "predicted_goals": values.sum(),
            "actual_goals": int(data["goal"].sum()),
        })
    return pd.DataFrame(rows)


def calibration_table(data: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, values in [("statsbomb", data["shot_statsbomb_xg"]), ("boosting", predictions["boosting"]),
                        ("statsbomb_recalibrated", predictions["statsbomb_recalibrated"])]:
        bins = pd.cut(values.astype(float), CALIBRATION_BINS, include_lowest=True)
        grouped = data.assign(prediction=values.astype(float), bin=bins).groupby("bin", observed=True)
        table = grouped.agg(shots=("goal", "size"), mean_predicted=("prediction", "mean"), goal_rate=("goal", "mean")).reset_index()
        table["model"] = key
        table["bin"] = table["bin"].astype(str)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def gap_breakdown(data: pd.DataFrame) -> pd.DataFrame:
    """Goals minus StatsBomb xG by shot context, with a z-score against pure chance.

    If every xG value were correct, goals would vary around their sum with variance
    sum(p * (1 - p)); z is how many of those standard deviations the gap is.
    """
    data = data.assign(variance=data["shot_statsbomb_xg"] * (1 - data["shot_statsbomb_xg"]))
    dimensions = {
        "All non-penalty shots": pd.Series("All", index=data.index),
        "Distance": data["distance_band"].astype(str),
        "Body part": np.where(data["is_header"].eq(1), "Header", "Foot / other"),
        "Chance type": data["chance_category"],
        "Freeze frame": np.where(data["defenders_in_cone"].eq(0), "No defender between ball and goal", "Defender(s) in the way"),
    }
    rows = []
    for dimension, values in dimensions.items():
        grouped = data.assign(group=values).groupby("group", observed=True).agg(
            shots=("goal", "size"), goals=("goal", "sum"), xg=("shot_statsbomb_xg", "sum"), variance=("variance", "sum")
        ).reset_index()
        grouped.insert(0, "dimension", dimension)
        rows.append(grouped)
    table = pd.concat(rows, ignore_index=True)
    table["goals_minus_xg"] = table["goals"] - table["xg"]
    table["goals_per_xg"] = table["goals"] / table["xg"]
    table["z_score"] = table["goals_minus_xg"] / np.sqrt(table["variance"])
    return table.drop(columns="variance")


def group_finishing(data: pd.DataFrame, predictions: pd.DataFrame, column: str, minimum_shots: int) -> pd.DataFrame:
    """Goals against both xG models for teams, shooters or goalkeepers faced."""
    frame = data.assign(liga_f_xg=predictions["boosting"].astype(float))
    grouped = frame.groupby(column).agg(
        shots=("goal", "size"), goals=("goal", "sum"), statsbomb_xg=("shot_statsbomb_xg", "sum"), liga_f_xg=("liga_f_xg", "sum")
    ).reset_index()
    grouped = grouped.loc[grouped["shots"] >= minimum_shots].copy()
    grouped["goals_minus_statsbomb_xg"] = grouped["goals"] - grouped["statsbomb_xg"]
    grouped["goals_minus_liga_f_xg"] = grouped["goals"] - grouped["liga_f_xg"]
    return grouped.sort_values("goals_minus_liga_f_xg", ascending=False)


def feature_importance(data: pd.DataFrame) -> pd.DataFrame:
    """Standardised logistic coefficients: the direction and size of each feature's effect."""
    features = list(FEATURE_LABELS)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    model.fit(data[features].astype(float), data["goal"])
    coefficients = model[-1].coef_[0]
    return pd.DataFrame({"feature": features, "label": [FEATURE_LABELS[f] for f in features], "coefficient": coefficients}) \
        .sort_values("coefficient", key=np.abs, ascending=False)


def run_xg_analysis(events: pd.DataFrame, shots: pd.DataFrame) -> dict[str, pd.DataFrame]:
    data = build_shot_features(events, shots)
    predictions = out_of_fold_predictions(data)
    shot_predictions = data[["shot_id", "match_id", "team", "shooter", "goalkeeper", "shot_x", "shot_y",
                             "chance_category", "shot_statsbomb_xg", "goal"]].assign(
        liga_f_xg=predictions["boosting"].astype(float),
        statsbomb_recalibrated_xg=predictions["statsbomb_recalibrated"].astype(float),
    )
    return {
        "xg_model_comparison": evaluate(data, predictions),
        "xg_calibration": calibration_table(data, predictions),
        "xg_gap_breakdown": gap_breakdown(data),
        "xg_team_finishing": group_finishing(data, predictions, "team", 50),
        "xg_shooter_finishing": group_finishing(data, predictions, "shooter", 30),
        "xg_goalkeeper_faced": group_finishing(data.loc[data["goalkeeper"].notna()], predictions, "goalkeeper", 60),
        "xg_feature_effects": feature_importance(data),
        "xg_shot_predictions": shot_predictions,
    }
