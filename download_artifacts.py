#!/usr/bin/env python
"""Download all model artifacts from a W&B project, saved by run name."""

import argparse
import os

import wandb


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
    args = parser.parse_args()

    api = wandb.Api()
    filters = {"state": "finished"} if args.only_finished else None
    runs = api.runs(args.project, filters=filters)

    print(f"Fetching runs from {args.project}...")
    for run in runs:
        for artifact in run.logged_artifacts():
            if args.type and artifact.type != args.type:
                continue
            safe_name = artifact.name.replace(":", "_")
            save_dir = os.path.join(args.output_dir, run.name, safe_name)
            print(f"[{run.name}] Downloading {artifact.name} ({artifact.type}) -> {save_dir}")
            artifact.download(root=save_dir)

    print("Done.")


if __name__ == "__main__":
    main()