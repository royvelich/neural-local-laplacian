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

  # W&B sweep with CLI overrides on top of sweep config
  python train_fmaps.py --sweep-id <entity/project/sweep_id> \\
      --override trainer.max_epochs=500 globals.k=20

  # W&B sweep with dataset and model path overrides
  python train_fmaps.py --sweep-id <entity/project/sweep_id> \\
      --data-root /workspace/datasets/DeformingThings4DMatching \\
      --checkpoint-path /workspace/models/model.ckpt
"""
# standard library
import os
import argparse
import yaml

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

def wandb_sweep_main(overrides=None, data_root=None, checkpoint_path=None):
    """Entry point for ``wandb.agent``.

    On the first (rank-0) process ``wandb.init()`` pulls the sweep config.
    For DDP subprocesses the config is passed via the ``WANDB_SWEEP_CONFIG``
    environment variable so they don't re-initialise a W&B run.

    Args:
        overrides: Optional list of OmegaConf dot-list overrides
                   (e.g. ["trainer.max_epochs=500", "globals.k=20"]).
        data_root: Optional path to override pair_generator.root in both
                   train and val dataset specifications.
        checkpoint_path: Optional path to override the pretrained model
                         checkpoint (globals + model block).
    """
    if 'WANDB_SWEEP_CONFIG' not in os.environ:
        wandb.init()
        config = OmegaConf.create(dict(wandb.config))
        config = OmegaConf.to_container(config, resolve=True)
        config = OmegaConf.create(config)
        os.environ['WANDB_SWEEP_CONFIG'] = OmegaConf.to_yaml(config)
    else:
        config = OmegaConf.create(yaml.safe_load(os.environ['WANDB_SWEEP_CONFIG']))

    # Apply CLI overrides on top of sweep config
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))

    # Override dataset root in both train and val pair generators
    if data_root is not None:
        config.data_module.module.train_dataset_specification.dataset.pair_generator.root = data_root
        for val_spec in config.data_module.module.val_dataset_specifications:
            val_spec.dataset.pair_generator.root = data_root

    # Override checkpoint path (must set both globals AND the resolved model
    # field, because resolve=True above flattens ${globals.checkpoint_path}
    # to a literal before overrides are applied)
    if checkpoint_path is not None:
        config.globals.checkpoint_path = checkpoint_path
        config.model.module.checkpoint_path = checkpoint_path

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
    parser.add_argument("--override", nargs="*", default=[],
                        help="OmegaConf dot-list overrides applied on top of the "
                             "sweep config, e.g. globals.k=20 trainer.max_epochs=500")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override the dataset root path (sets both train and val "
                             "pair_generator.root)")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Override globals.checkpoint_path (pretrained model)")
    args, _ = parser.parse_known_args()

    if args.sweep_id is None:
        main_hydra()
    else:
        overrides = list(args.override)

        if 'WANDB_SWEEP_CONFIG' not in os.environ:
            wandb.agent(sweep_id=args.sweep_id,
                        function=lambda: wandb_sweep_main(
                            overrides,
                            data_root=args.data_root,
                            checkpoint_path=args.checkpoint_path),
                        count=1)
        else:
            wandb_sweep_main(overrides,
                             data_root=args.data_root,
                             checkpoint_path=args.checkpoint_path)