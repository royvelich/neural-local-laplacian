# standard library
import os
import argparse
import yaml
# os.environ["WANDB_DIR"] = "F:/wandb/WANDB_DIR"
# os.environ["WANDB_ARTIFACT_DIR"] = "F:/wandb/WANDB_ARTIFACT_DIR"
# os.environ["WANDB_CACHE_DIR"] = "F:/wandb/WANDB_CACHE_DIR"
# os.environ["WANDB_CONFIG_DIR"] = "F:/wandb/WANDB_CONFIG_DIR"
# os.environ["WANDB_DATA_DIR"] = "F:/wandb/WANDB_DATA_DIR"

# Must be set BEFORE importing torch (cuBLAS workspace requirement)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # or ":16:8"
os.environ["PYTHONHASHSEED"] = "0"

# hydra
import hydra

# omegaconf
from omegaconf import OmegaConf, DictConfig

# torch
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

# wandb
import wandb

# lightning
import pytorch_lightning as pl


# =============================================================================
# Core training logic (no decorators — callable from Hydra or W&B sweep)
# =============================================================================

def main(cfg: DictConfig) -> None:
    pl.seed_everything(seed=cfg.globals.training_seed, workers=True)





    import random, numpy as np, torch
    random.seed(cfg.globals.training_seed)
    np.random.seed(cfg.globals.training_seed)
    torch.manual_seed(cfg.globals.training_seed)  # seeds CPU + all CUDA devices
    torch.cuda.manual_seed_all(cfg.globals.training_seed)  # explicit, in case you fork

    # # Force deterministic algorithms; raises if a non-deterministic op is used
    # torch.use_deterministic_algorithms(True, warn_only=False)
    #
    # # cuDNN: disable autotuner + force deterministic conv kernels
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True





    torch.set_float32_matmul_precision(precision='medium')
    data_module = hydra.utils.instantiate(config=cfg.data_module.module)
    model = hydra.utils.instantiate(
        config=cfg.model.module,
        optimizer_cfg=cfg.optimizer,
        scheduler_cfg=cfg.scheduler if 'scheduler' in cfg else None,
    )
    trainer = hydra.utils.instantiate(config=cfg.trainer)
    trainer.cfg = cfg
    trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.globals.ckpt_path)


# =============================================================================
# W&B sweep entry point
# =============================================================================

def wandb_sweep_main(overrides=None):
    """Entry point for ``wandb.agent``.

    On the first (rank-0) process ``wandb.init()`` pulls the sweep config.
    For DDP subprocesses the config is passed via the ``WANDB_SWEEP_CONFIG``
    environment variable so they don't re-initialise a W&B run.

    Args:
        overrides: Optional list of OmegaConf dot-list overrides
                   (e.g. ["globals.training_seed=7", "optimizer.lr=1.0e-4"]).
    """
    if 'WANDB_SWEEP_CONFIG' not in os.environ:
        wandb.init()
        config = OmegaConf.create(dict(wandb.config))
        config = OmegaConf.to_container(config, resolve=True)
        config = OmegaConf.create(config)
        os.environ['WANDB_SWEEP_CONFIG'] = OmegaConf.to_yaml(config)
    else:
        config = OmegaConf.create(yaml.safe_load(os.environ['WANDB_SWEEP_CONFIG']))

    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))

    main(cfg=config)


# =============================================================================
# Hydra entry point (normal CLI usage)
# =============================================================================

@hydra.main(version_base="1.2", config_path="training_config")
def main_hydra(cfg: DictConfig) -> None:
    main(cfg=cfg)


# =============================================================================
# Dispatcher
# =============================================================================

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
