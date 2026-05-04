# standard library
import os
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
from omegaconf import DictConfig

# torch
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

# lightning
import pytorch_lightning as pl


@hydra.main(version_base="1.2", config_path="training_config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(seed=cfg.globals.training_seed, workers=True)





    import random, numpy as np, torch
    random.seed(cfg.globals.training_seed)
    np.random.seed(cfg.globals.training_seed)
    torch.manual_seed(cfg.globals.training_seed)  # seeds CPU + all CUDA devices
    torch.cuda.manual_seed_all(cfg.globals.training_seed)  # explicit, in case you fork

    # Force deterministic algorithms; raises if a non-deterministic op is used
    torch.use_deterministic_algorithms(True, warn_only=False)

    # cuDNN: disable autotuner + force deterministic conv kernels
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True





    torch.set_float32_matmul_precision(precision='medium')
    data_module = hydra.utils.instantiate(config=cfg.data_module.module)
    model = hydra.utils.instantiate(config=cfg.model.module, optimizer_cfg=cfg.optimizer, scheduler_cfg=cfg.scheduler if 'scheduler' in cfg else None)
    trainer = hydra.utils.instantiate(config=cfg.trainer)
    trainer.cfg = cfg
    trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.globals.ckpt_path)


if __name__ == "__main__":
    main()
