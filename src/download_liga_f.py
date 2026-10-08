"""Download Liga F 2023/24 from StatsBomb open data into the cache the pipeline reads.

Writes
------
<project-dir>/data/processed/liga_f_matches.csv
<project-dir>/data/raw/events/<match_id>.pkl   (one file per match; existing files are skipped)

The download is resumable: re-running it only fetches matches that are not cached yet.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from statsbombpy import sb


COMPETITION_ID = 182  # Liga F
SEASON_ID = 281       # 2023/2024


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--limit", type=int, default=None, help="Only download this many matches (for testing).")
    parser.add_argument("--pause", type=float, default=0.5, help="Seconds to wait between requests.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    project_dir = args.project_dir.expanduser().resolve()
    events_dir = project_dir / "data" / "raw" / "events"
    processed_dir = project_dir / "data" / "processed"
    events_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
    matches.to_csv(processed_dir / "liga_f_matches.csv", index=False)
    print(f"Saved metadata for {len(matches)} matches.")

    match_ids = sorted(matches["match_id"])[: args.limit]
    pending = [match_id for match_id in match_ids if not (events_dir / f"{match_id}.pkl").exists()]
    print(f"{len(match_ids) - len(pending)} already cached, {len(pending)} to download.")

    failed: list[int] = []
    for position, match_id in enumerate(pending, start=1):
        try:
            events = sb.events(match_id=match_id)
            events["match_id"] = match_id
            # Write to a temporary name first so an interrupted run never leaves a half-written file.
            temporary = events_dir / f"{match_id}.pkl.part"
            events.to_pickle(temporary)
            temporary.rename(events_dir / f"{match_id}.pkl")
            print(f"[{position}/{len(pending)}] {match_id}: {len(events)} events")
        except Exception as error:  # Keep going; the summary lists what to retry.
            failed.append(match_id)
            print(f"[{position}/{len(pending)}] {match_id}: FAILED ({error})")
        time.sleep(args.pause)

    if failed:
        print(f"\n{len(failed)} matches failed; re-run the script to retry: {failed}")
    else:
        print("\nAll requested matches are cached.")


if __name__ == "__main__":
    main()
