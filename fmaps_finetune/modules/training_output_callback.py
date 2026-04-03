"""
Lightning Callback that exports training metrics, config, and plots to a local
folder at the end of training.

Usage in train_fmaps.py:
    from training_output_callback import TrainingOutputCallback
    ...
    trainer = hydra.utils.instantiate(config=cfg.trainer)
    trainer.callbacks.append(TrainingOutputCallback(cfg))
"""

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf


class TrainingOutputCallback(pl.Callback):
    """Capture per-epoch metrics and export CSV + plots at end of training.

    Produces a self-contained output folder with:
        config.yaml          – full Hydra config for reproducibility
        epoch_metrics.csv    – one row per epoch, all train + val columns
        summary.txt          – best epoch, final metrics, total time
        figures/             – loss curves, accuracy, lr schedule
    """

    def __init__(
        self,
        cfg: Optional[DictConfig] = None,
        output_root: str = "./training_results",
    ):
        super().__init__()
        self._cfg = cfg  # may be None if added via YAML; resolved in on_fit_start
        self.output_root = Path(output_root)
        self.epoch_metrics: List[Dict[str, Any]] = []
        self.t_start: Optional[float] = None
        self.output_dir: Optional[Path] = None

    @property
    def cfg(self) -> Optional[DictConfig]:
        return self._cfg

    # ------------------------------------------------------------------
    # Folder naming
    # ------------------------------------------------------------------

    def _build_run_name(self, trainer: pl.Trainer) -> str:
        """Build a descriptive folder name from config and run context."""
        parts = []

        # Dataset name from Hydra choices or data_module config
        try:
            from hydra.core.hydra_config import HydraConfig
            ds = HydraConfig.get().runtime.choices.get('data_module', None)
        except Exception:
            ds = None
        if ds is None:
            # Fallback: extract from data_module._target_ or class name
            dm_cfg = self.cfg.get('data_module', {})
            dm_mod = dm_cfg.get('module', {})
            target = dm_mod.get('_target_', '')
            ds = target.split('.')[-1] if target else 'unknown'
        parts.append(ds)

        # k value
        k = self.cfg.get('globals', {}).get('k', None)
        if k is not None:
            parts.append(f'k{k}')

        # Laplacian assembly type
        lap_cfg = self.cfg.get('model', {}).get('module', {}).get('train_laplacian', {})
        if isinstance(lap_cfg, dict) and lap_cfg.get('assembly'):
            asm = lap_cfg['assembly']
            if asm != 'diagonal_gram':
                parts.append(asm)

        # Loss info — try loss_preset first, then individual loss names
        loss_preset = self.cfg.get('globals', {}).get('loss_preset', None)
        if loss_preset:
            parts.append(f'loss{loss_preset}')
        else:
            losses = self.cfg.get('model', {}).get('module', {}).get('losses', [])
            if losses:
                names = []
                for loss in losses:
                    if isinstance(loss, dict):
                        target = loss.get('_target_', '')
                        name = target.split('.')[-1].replace('Loss', '').lower()
                        if name:
                            names.append(name)
                if names:
                    parts.append('_'.join(names[:3]))  # cap at 3 for readability

        # Max vertices
        mv = self.cfg.get('model', {}).get('module', {}).get('max_vertices', None)
        if mv and mv > 0:
            parts.append(f'mv{mv}')

        # W&B run name (if available)
        try:
            import wandb
            if wandb.run is not None and wandb.run.name:
                parts.append(wandb.run.name)
        except ImportError:
            pass

        # Timestamp
        parts.append(datetime.now().strftime('%d%b%Y_%H%M'))

        return '_'.join(str(p) for p in parts)

    # ------------------------------------------------------------------
    # Metric capture
    # ------------------------------------------------------------------

    def _capture_metrics(self, trainer: pl.Trainer):
        """Snapshot all callback_metrics into the current epoch's entry."""
        epoch = trainer.current_epoch

        # Find or create entry for this epoch
        if not self.epoch_metrics or self.epoch_metrics[-1]['epoch'] != epoch:
            self.epoch_metrics.append({'epoch': epoch})
        entry = self.epoch_metrics[-1]

        for k, v in trainer.callback_metrics.items():
            try:
                entry[k] = v.item() if hasattr(v, 'item') else float(v)
            except (ValueError, TypeError):
                pass

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not trainer.is_global_zero:
            return

        self.t_start = time.time()

        # Resolve cfg: prefer constructor arg, fall back to trainer.cfg
        if self._cfg is None:
            self._cfg = getattr(trainer, 'cfg', None)
        if self._cfg is None:
            print("[TrainingOutput] Warning: no config available — skipping output")
            return

        # Skip for eval_only mode
        if getattr(pl_module.hparams, 'eval_only', False):
            return

        # Create output folder
        run_name = self._build_run_name(trainer)
        self.output_dir = self.output_root / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Dump full config
        config_path = self.output_dir / 'config.yaml'
        try:
            with open(config_path, 'w') as f:
                f.write(OmegaConf.to_yaml(self.cfg, resolve=True))
            print(f"\n[TrainingOutput] Output dir: {self.output_dir}")
            print(f"[TrainingOutput] Config saved: {config_path}")
        except Exception as e:
            print(f"[TrainingOutput] Warning: could not save config: {e}")

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Capture baseline summaries computed in FunctionalMapModule.on_fit_start."""
        if not trainer.is_global_zero or self.output_dir is None:
            return

        baselines = getattr(pl_module, '_baseline_summaries', None)
        if baselines:
            entry = {'epoch': -1}  # special "baseline" row before training
            entry.update(baselines)
            self.epoch_metrics.insert(0, entry)
            print(f"[TrainingOutput] Captured {len(baselines)} baseline metrics")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not trainer.is_global_zero or self.output_dir is None:
            return
        self._capture_metrics(trainer)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not trainer.is_global_zero or self.output_dir is None:
            return
        self._capture_metrics(trainer)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not trainer.is_global_zero or self.output_dir is None:
            return
        if not self.epoch_metrics:
            return

        elapsed = time.time() - self.t_start if self.t_start else 0

        # Write CSV
        csv_path = self._write_csv()

        # Write summary
        summary_path = self._write_summary(trainer, elapsed)

        # Generate plots
        figures_dir = self.output_dir / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)
        self._generate_plots(figures_dir)

        print(f"\n[TrainingOutput] Results saved to {self.output_dir}/")
        print(f"  {csv_path.name}")
        print(f"  {summary_path.name}")
        n_figs = len(list(figures_dir.glob('*')))
        if n_figs:
            print(f"  figures/ ({n_figs} plots)")

    # ------------------------------------------------------------------
    # Export: CSV
    # ------------------------------------------------------------------

    def _write_csv(self) -> Path:
        """Write epoch_metrics.csv with one row per epoch."""
        csv_path = self.output_dir / 'epoch_metrics.csv'

        # Collect all keys in order of appearance
        all_keys = []
        seen = set()
        for entry in self.epoch_metrics:
            for k in entry:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
            writer.writeheader()
            for entry in self.epoch_metrics:
                writer.writerow(entry)

        return csv_path

    # ------------------------------------------------------------------
    # Export: Summary
    # ------------------------------------------------------------------

    def _write_summary(self, trainer: pl.Trainer, elapsed: float) -> Path:
        """Write a human-readable summary.txt."""
        summary_path = self.output_dir / 'summary.txt'

        # Find best epoch by val/best_acc (or val/top1)
        best_epoch = None
        best_val = -1
        for entry in self.epoch_metrics:
            val = entry.get('val/best_acc', entry.get('val/top1', None))
            if val is not None and val > best_val:
                best_val = val
                best_epoch = entry['epoch']

        # Final epoch metrics
        final = self.epoch_metrics[-1] if self.epoch_metrics else {}

        with open(summary_path, 'w') as f:
            f.write(f"Training Summary\n")
            f.write(f"{'=' * 60}\n\n")

            f.write(f"Total epochs:    {len(self.epoch_metrics)}\n")
            f.write(f"Total time:      {elapsed:.1f}s ({elapsed/60:.1f}min)\n")
            if best_epoch is not None:
                f.write(f"Best epoch:      {best_epoch} (val_acc={best_val:.4f})\n")
            f.write(f"\n")

            # Final metrics
            f.write(f"Final Epoch ({final.get('epoch', '?')}) Metrics\n")
            f.write(f"{'-' * 60}\n")

            # Group by prefix
            train_keys = sorted(k for k in final if k.startswith('train/'))
            val_keys = sorted(k for k in final if k.startswith('val/'))
            other_keys = sorted(k for k in final
                                if not k.startswith('train/')
                                and not k.startswith('val/')
                                and k != 'epoch')

            for group_name, keys in [('Train', train_keys),
                                     ('Val', val_keys),
                                     ('Other', other_keys)]:
                if not keys:
                    continue
                f.write(f"\n  {group_name}:\n")
                for k in keys:
                    v = final[k]
                    if isinstance(v, float):
                        f.write(f"    {k:<40s} {v:.6f}\n")
                    else:
                        f.write(f"    {k:<40s} {v}\n")

        return summary_path

    # ------------------------------------------------------------------
    # Export: Plots
    # ------------------------------------------------------------------

    def _generate_plots(self, figures_dir: Path):
        """Generate training curve plots."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("[TrainingOutput] matplotlib not available — skipping plots")
            return

        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 12,
            'figure.dpi': 150,
            'savefig.dpi': 200,
            'savefig.bbox': 'tight',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
        })

        epochs = [e['epoch'] for e in self.epoch_metrics]

        self._plot_loss_curves(epochs, figures_dir, plt)
        self._plot_accuracy_curves(epochs, figures_dir, plt)
        self._plot_lr_schedule(epochs, figures_dir, plt)
        self._plot_val_geodesic(epochs, figures_dir, plt)

    def _get_series(self, key: str) -> tuple:
        """Extract (epochs, values) for a metric key, skipping missing entries."""
        epochs, vals = [], []
        for entry in self.epoch_metrics:
            if key in entry and entry[key] is not None:
                epochs.append(entry['epoch'])
                vals.append(entry[key])
        return np.array(epochs), np.array(vals)

    def _plot_loss_curves(self, epochs, figures_dir, plt):
        """Plot total loss + per-component losses."""
        fig, ax = plt.subplots(figsize=(10, 5))
        has_data = False

        # Total loss
        ep, vals = self._get_series('train/loss')
        if len(ep) > 0:
            ax.plot(ep, vals, 'k-', linewidth=2, label='total', zorder=5)
            has_data = True

        # Per-component losses
        loss_keys = sorted(set(
            k for e in self.epoch_metrics for k in e
            if k.startswith('train/loss_') and k != 'train/loss'
        ))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(loss_keys), 1)))
        for k, color in zip(loss_keys, colors):
            ep, vals = self._get_series(k)
            if len(ep) > 0:
                label = k.replace('train/', '')
                ax.plot(ep, vals, '-', color=color, linewidth=1.5,
                        alpha=0.8, label=label)
                has_data = True

        if not has_data:
            plt.close(fig)
            return

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax.set_xlim(left=0)

        for ext in ['pdf', 'png']:
            fig.savefig(figures_dir / f'loss_curves.{ext}')
        plt.close(fig)

    def _plot_accuracy_curves(self, epochs, figures_dir, plt):
        """Plot train/val accuracy curves."""
        fig, ax = plt.subplots(figsize=(10, 5))
        has_data = False

        acc_keys = [
            ('train/top1', 'Train top1', '--', '#2563EB'),
            ('val/top1', 'Val top1', '-', '#2563EB'),
            ('val/sp_top1', 'Val sp_top1', '-', '#DC2626'),
            ('val/best_acc', 'Val best', '-', '#16A34A'),
        ]

        for key, label, ls, color in acc_keys:
            ep, vals = self._get_series(key)
            if len(ep) > 0:
                ax.plot(ep, vals, ls, color=color, linewidth=2, label=label)
                has_data = True

        # Also plot any val/*/accuracy or val/*/geo_at_05pct
        extra_keys = sorted(set(
            k for e in self.epoch_metrics for k in e
            if (k.startswith('val/') and
                ('accuracy' in k or 'geo_at_05pct' in k) and
                k not in ('val/top1', 'val/sp_top1', 'val/best_acc'))
        ))
        extra_colors = plt.cm.Set2(np.linspace(0, 1, max(len(extra_keys), 1)))
        for k, color in zip(extra_keys, extra_colors):
            ep, vals = self._get_series(k)
            if len(ep) > 0:
                label = k.replace('val/', '')
                ax.plot(ep, vals, '-', color=color, linewidth=1.5,
                        alpha=0.7, label=label)
                has_data = True

        if not has_data:
            plt.close(fig)
            return

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy / Metric')
        ax.set_title('Accuracy Curves')
        ax.legend(loc='lower right', fontsize=8, framealpha=0.9,
                  ncol=2 if len(extra_keys) > 4 else 1)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.05)

        for ext in ['pdf', 'png']:
            fig.savefig(figures_dir / f'accuracy_curves.{ext}')
        plt.close(fig)

    def _plot_lr_schedule(self, epochs, figures_dir, plt):
        """Plot learning rate over epochs."""
        ep, vals = self._get_series('train/lr')
        if len(ep) == 0:
            return

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ep, vals, '-', color='#6B7280', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_xlim(left=0)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, -3))

        for ext in ['pdf', 'png']:
            fig.savefig(figures_dir / f'lr_schedule.{ext}')
        plt.close(fig)

    def _plot_val_geodesic(self, epochs, figures_dir, plt):
        """Plot validation geodesic threshold metrics over epochs."""
        # Find all geo_at_* keys
        geo_keys = sorted(set(
            k for e in self.epoch_metrics for k in e
            if 'geo_at_' in k and k.startswith('val/')
        ))

        if not geo_keys:
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, max(len(geo_keys), 1)))

        for k, color in zip(geo_keys, colors):
            ep, vals = self._get_series(k)
            if len(ep) > 0:
                label = k.replace('val/', '')
                ax.plot(ep, vals, '-', color=color, linewidth=1.5,
                        alpha=0.8, label=label)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric Value')
        ax.set_title('Validation Geodesic Metrics')
        ax.legend(loc='lower right', fontsize=7, framealpha=0.9, ncol=2)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.05)

        for ext in ['pdf', 'png']:
            fig.savefig(figures_dir / f'val_geodesic.{ext}')
        plt.close(fig)