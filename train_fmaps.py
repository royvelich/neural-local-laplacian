#!/usr/bin/env python3
"""Fine-tune (or train from scratch) the neural Laplacian for functional map correspondence.

Usage examples:

  # SMAL — single GPU, fine-tune from pretrained checkpoint
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
"""
import torch
import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl

torch.multiprocessing.set_sharing_strategy("file_system")


@hydra.main(version_base="1.2", config_path="fmaps_config")
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


if __name__ == "__main__":
    main()