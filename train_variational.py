# Variational / Dirichlet-energy training entry point.
#
# Mirrors train.py but points Hydra at training_config_variational/.  The
# core ``main`` and ``wandb_sweep_main`` are imported from train.py so the
# DDP / determinism / sweep wiring lives in exactly one place.

import os

# Must be set BEFORE importing torch (cuBLAS workspace requirement)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"

import argparse

import hydra
from omegaconf import DictConfig
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import wandb

from train import main, wandb_sweep_main


@hydra.main(version_base="1.2", config_path="training_config_variational")
def main_hydra(cfg: DictConfig) -> None:
    main(cfg=cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-id", type=str, default=None,
                        help="W&B sweep ID (<entity>/<project>/<sweep_id>). "
                             "When provided, runs as a sweep agent instead of Hydra.")
    parser.add_argument("--override", nargs="*", default=[],
                        help="OmegaConf dot-list overrides applied on top of the "
                             "sweep config, e.g. globals.training_seed=7 optimizer.lr=1.0e-4")
    args, _ = parser.parse_known_args()

    if args.sweep_id is None:
        main_hydra()
    else:
        overrides = list(args.override)
        if 'WANDB_SWEEP_CONFIG' not in os.environ:
            wandb.agent(sweep_id=args.sweep_id,
                        function=lambda: wandb_sweep_main(overrides),
                        count=1)
        else:
            wandb_sweep_main(overrides)
