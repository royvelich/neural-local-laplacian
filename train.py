# standard library
import os
# os.environ["WANDB_DIR"] = "F:/wandb/WANDB_DIR"
# os.environ["WANDB_ARTIFACT_DIR"] = "F:/wandb/WANDB_ARTIFACT_DIR"
# os.environ["WANDB_CACHE_DIR"] = "F:/wandb/WANDB_CACHE_DIR"
# os.environ["WANDB_CONFIG_DIR"] = "F:/wandb/WANDB_CONFIG_DIR"
# os.environ["WANDB_DATA_DIR"] = "F:/wandb/WANDB_DATA_DIR"

# hydra
import hydra

# omegaconf
from omegaconf import DictConfig

# torch
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

# lightning
import pytorch_lightning as pl


@hydra.main(version_base="1.2", config_path="training_config")
def main(cfg: DictConfig) -> None:
    # print(cfg)
    torch.set_float32_matmul_precision(precision='medium')
    pl.seed_everything(seed=cfg.globals.training_seed)
    data_module = hydra.utils.instantiate(config=cfg.data_module.module)
    model = hydra.utils.instantiate(config=cfg.model.module, optimizer_cfg=cfg.optimizer, scheduler_cfg=cfg.scheduler if 'scheduler' in cfg else None)
    trainer = hydra.utils.instantiate(config=cfg.trainer)
    trainer.cfg = cfg
    trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.globals.ckpt_path)


if __name__ == "__main__":
    main()
