#!/usr/bin/env python3
"""Fine-tune (or train from scratch) the neural Laplacian for functional map correspondence.

Usage examples:

  # SMAL — single GPU, fine-tune from pretrained checkpoint (Hydra)
  python train_fmaps.py \\
      globals.checkpoint_path=/path/to/model.ckpt \\
      data_module=smal \\
      data_module.module.pair_generator_cfg.smal.model_path=/path/to/smal_CVPR2017.pkl \\
      data_module.module.pair_generator_cfg.smal.data_path=/path/to/smal_CVPR2017_data.pkl

  # SMAL — 4-GPU DDP
  python train_fmaps.py \\
      globals.checkpoint_path=/path/to/model.ckpt \\
      data_module=smal \\
      data_module.module.pair_generator_cfg.smal.model_path=/path/to/smal_CVPR2017.pkl \\
      data_module.module.pair_generator_cfg.smal.data_path=/path/to/smal_CVPR2017_data.pkl \\
      trainer.devices=4 trainer.strategy=ddp

  # DT4D — random init, custom val categories
  python train_fmaps.py \\
      globals.checkpoint_path=/path/to/model.ckpt \\
      data_module=dt4d \\
      data_module.module.pair_generator_cfg.root=/path/to/DeformingThings4DMatching \\
      model.module.random_init=true \\
      model.module.keep_areas_head=true \\
      data_module.module.pair_generator_cfg.val_categories="[crypto,zombie]"

  # LoRA fine-tuning
  python train_fmaps.py \\
      globals.checkpoint_path=/path/to/model.ckpt \\
      data_module=smal \\
      data_module.module.pair_generator_cfg.smal.model_path=... \\
      data_module.module.pair_generator_cfg.smal.data_path=... \\
      model.module.use_lora=true \\
      model.module.lora_rank=16

  # Override loss type to DCL-W and increase landmarks
  python train_fmaps.py \\
      globals.checkpoint_path=/path/to/model.ckpt \\
      data_module=smal ... \\
      model.module.loss_fn.loss_type=dclw \\
      model.module.loss_fn.num_landmarks=1024

  # W&B sweep (config supplied by the sweep agent, not Hydra)
  python train_fmaps.py --sweep-id <entity/project/sweep_id>
"""
# standard library
import os
import sys
import io
import argparse
import yaml

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

os.environ.setdefault("WANDB_DIR", "C:/wandb/WANDB_DIR")
os.environ.setdefault("WANDB_ARTIFACT_DIR", "C:/wandb/WANDB_ARTIFACT_DIR")
os.environ.setdefault("WANDB_CACHE_DIR", "C:/wandb/WANDB_CACHE_DIR")
os.environ.setdefault("WANDB_CONFIG_DIR", "C:/wandb/WANDB_CONFIG_DIR")
os.environ.setdefault("WANDB_DATA_DIR", "C:/wandb/WANDB_DATA_DIR")

# hydra
import hydra

# omegaconf
from omegaconf import OmegaConf, DictConfig

# torch
import torch
torch.multiprocessing.set_sharing_strategy("file_system")

# wandb
import wandb

# lightning
import lightning.pytorch as pl


# =============================================================================
# Core training logic (no decorators — callable from Hydra or W&B sweep)
# =============================================================================

def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(seed=cfg.globals.training_seed)

    data_module = hydra.utils.instantiate(config=cfg.data_module.module)

    # Instantiate optimizer and scheduler explicitly as partials before passing
    # to the model. Hydra does not recursively instantiate override kwargs, so
    # cfg.optimizer / cfg.scheduler would arrive as raw DictConfigs otherwise.
    optimizer_cfg = hydra.utils.instantiate(cfg.optimizer)
    scheduler_cfg = hydra.utils.instantiate(cfg.scheduler) if 'scheduler' in cfg else None

    model = hydra.utils.instantiate(
        config=cfg.model.module,
        optimizer_cfg=optimizer_cfg,
        scheduler_cfg=scheduler_cfg,
    )

    trainer = hydra.utils.instantiate(config=cfg.trainer)
    trainer.cfg = cfg  # makes cfg accessible in LaplacianModuleBase.setup() for wandb logging

    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path=cfg.globals.ckpt_path,  # resume a Lightning checkpoint (None = start fresh)
    )


# =============================================================================
# W&B sweep entry point
# =============================================================================

def wandb_sweep_main():
    """Entry point for ``wandb.agent``.

    On the first (rank-0) process ``wandb.init()`` pulls the sweep config.
    For DDP subprocesses the config is passed via the ``WANDB_SWEEP_CONFIG``
    environment variable so they don't re-initialise a W&B run.
    """
    if 'WANDB_SWEEP_CONFIG' not in os.environ:
        wandb.init()
        config = OmegaConf.create(dict(wandb.config))
        config = OmegaConf.to_container(config, resolve=True)
        config = OmegaConf.create(config)
        os.environ['WANDB_SWEEP_CONFIG'] = OmegaConf.to_yaml(config)
    else:
        config = OmegaConf.create(yaml.safe_load(os.environ['WANDB_SWEEP_CONFIG']))

    main(cfg=config)


# =============================================================================
# Hydra entry point (normal CLI usage)
# =============================================================================

@hydra.main(version_base="1.2", config_path="fmaps_config")
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
    args, _ = parser.parse_known_args()

    if args.sweep_id is None:
        main_hydra()
    else:
        if 'WANDB_SWEEP_CONFIG' not in os.environ:
            wandb.agent(sweep_id=args.sweep_id, function=wandb_sweep_main, count=1)
        else:
            wandb_sweep_main()