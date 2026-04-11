#!/usr/bin/env python
"""Download full W&B run history to CSV.

Usage: python dl_wandb.py entity/project/run_id
"""
import sys, wandb, pandas as pd

path = sys.argv[1]  # e.g. "entity/project/run_id"
api = wandb.Api()
run = api.run(path)
df = run.history(samples=run.lastHistoryStep + 1, pandas=True)  # all steps
df.to_csv(f"wandb_{run.id}.csv", index=False)
print(f"Saved {len(df)} rows to wandb_{run.id}.csv")