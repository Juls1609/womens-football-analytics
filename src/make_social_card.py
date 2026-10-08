"""Draw the 1200 x 630 preview image that LinkedIn, X and Slack show when the site is shared.

Run after the pipeline: python src/make_social_card.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Arc, Rectangle


PROJECT_DIR = Path(__file__).resolve().parents[1]
NAVY, CYAN, CYAN_LIGHT, GOLD, WHITE = "#062a68", "#00a9e0", "#7be1ff", "#f2b234", "#ffffff"


def draw_half_pitch(axis: plt.Axes) -> None:
    line = {"color": WHITE, "alpha": 0.35, "linewidth": 1.4}
    axis.plot([60, 120, 120, 60, 60], [0, 0, 80, 80, 0], **line)
    axis.add_patch(Rectangle((102, 18), 18, 44, fill=False, **line))
    axis.add_patch(Rectangle((114, 30), 6, 20, fill=False, **line))
    axis.add_patch(Arc((60, 40), 20, 20, theta1=270, theta2=90, **line))
    axis.set_xlim(58, 122)
    axis.set_ylim(82, -2)
    axis.set_aspect("equal")
    axis.axis("off")


def main() -> None:
    shots = pd.read_csv(PROJECT_DIR / "app" / "data" / "liga_f_shot_map.csv")
    validation = pd.read_csv(PROJECT_DIR / "app" / "data" / "liga_f_validation_report.csv").set_index("check")["value"]
    goals = shots["is_goal_from_shot"].astype(str).str.lower().eq("true")

    figure = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=NAVY)
    pitch = figure.add_axes([0.52, 0.06, 0.46, 0.88], facecolor=NAVY)
    draw_half_pitch(pitch)
    sizes = 4 + shots["shot_statsbomb_xg"] ** 0.5 * 60
    pitch.scatter(shots.loc[~goals, "shot_x"], shots.loc[~goals, "shot_y"], s=sizes[~goals], color=CYAN, alpha=0.22, linewidths=0)
    pitch.scatter(shots.loc[goals, "shot_x"], shots.loc[goals, "shot_y"], s=sizes[goals], color=GOLD, alpha=0.7, linewidths=0)

    text = figure.add_axes([0, 0, 0.55, 1], facecolor=NAVY)
    text.axis("off")
    text.text(0.1, 0.84, "LIGA F 2023/24 · STATSBOMB OPEN DATA", color=CYAN_LIGHT, fontsize=13, fontweight="bold")
    text.text(0.1, 0.52, "More, better\nand underrated.", color=WHITE, fontsize=46, fontweight="bold", linespacing=1.05)
    text.text(0.1, 0.33, f"{int(validation['matches_in_metadata'])} matches · {int(validation['total_shots']):,} shots\n"
              "Chance creation, a Liga F xG model\nand expected threat", color="#cfe3f6", fontsize=15, linespacing=1.5, va="top")
    text.text(0.1, 0.08, "Julieta Schwartz Cerveny · Football Data Scientist", color=WHITE, fontsize=13, fontweight="bold")

    output = PROJECT_DIR / "app" / "assets" / "social-card.png"
    figure.savefig(output, dpi=100, facecolor=NAVY)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
