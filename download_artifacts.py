#!/usr/bin/env python
"""Download all model artifacts from a W&B project, saved by run name."""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import wandb


def parse_date(s: str) -> str:
    """Parse YYYY-MM-DD into a naive ISO 8601 string for W&B filters.

    W&B's MongoDB-style filters expect a naive timestamp like
    'YYYY-MM-DDTHH:MM:SS' (no timezone suffix). Times are interpreted in UTC.
    """
    return datetime.strptime(s, "%Y-%m-%d").strftime("%Y-%m-%dT%H:%M:%S")


def download_one(run_name: str, artifact, output_dir: str) -> str:
    safe_name = artifact.name.replace(":", "_")
    save_dir = os.path.join(output_dir, run_name, safe_name)
    artifact.download(root=save_dir)
    return f"[{run_name}] {artifact.name} ({artifact.type}) -> {save_dir}"


def main():
    parser = argparse.ArgumentParser(description="Download model artifacts from a W&B project")
    parser.add_argument("project", nargs="?", default="gip-technion/neural-cotangent-weights",
                        help="W&B project path (entity/project)")
    parser.add_argument("--output-dir", "-o", default="artifacts",
                        help="Root output directory (default: artifacts)")
    parser.add_argument("--type", "-t", default=None,
                        help="Artifact type filter (e.g. 'model', 'checkpoint'). Default: all types")
    parser.add_argument("--only-finished", action="store_true",
                        help="Only download from finished runs")
    parser.add_argument("--start-date", type=parse_date, default=None,
                        help="Only runs on/after this date (YYYY-MM-DD, UTC)")
    parser.add_argument("--end-date", type=parse_date, default=None,
                        help="Only runs on/before this date (YYYY-MM-DD, UTC)")
    parser.add_argument("--date-field", choices=("heartbeat_at", "created_at"),
                        default="heartbeat_at",
                        help="Which timestamp the date range applies to. "
                             "'heartbeat_at' = last activity (default), "
                             "'created_at' = run start time")
    parser.add_argument("--workers", "-j", type=int, default=8,
                        help="Number of parallel download workers (default: 8)")
    args = parser.parse_args()

    # Build server-side filters
    filters: dict = {}
    if args.only_finished:
        filters["state"] = "finished"
    date_filter: dict = {}
    if args.start_date:
        date_filter["$gte"] = args.start_date
    if args.end_date:
        date_filter["$lte"] = args.end_date
    if date_filter:
        filters[args.date_field] = date_filter

    api = wandb.Api()
    runs = api.runs(args.project, filters=filters or None)

    print(f"Fetching runs from {args.project}...")
    tasks = []
    for run in runs:
        for artifact in run.logged_artifacts():
            if args.type and artifact.type != args.type:
                continue
            tasks.append((run.name, artifact))

    if not tasks:
        print("No matching artifacts found.")
        return

    print(f"Queued {len(tasks)} artifact(s); downloading with {args.workers} worker(s).")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(download_one, name, art, args.output_dir): (name, art.name)
            for name, art in tasks
        }
        for fut in as_completed(futures):
            name, art_name = futures[fut]
            try:
                print(fut.result())
            except Exception as e:
                print(f"[{name}] FAILED {art_name}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()