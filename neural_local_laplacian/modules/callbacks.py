from lightning.pytorch.callbacks import Callback


class DebugCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"=== EPOCH {trainer.current_epoch} END ===")

        # Check what metrics are available
        if hasattr(trainer, 'logged_metrics'):
            print(f"Logged metrics: {list(trainer.logged_metrics.keys())}")
            print(f"Callback metrics: {trainer.callback_metrics}")

        # Check if ModelCheckpoint exists
        for callback in trainer.callbacks:
            if hasattr(callback, 'monitor'):
                print(f"ModelCheckpoint found - monitoring: {callback.monitor}")
                print(f"ModelCheckpoint best score: {getattr(callback, 'best_model_score', 'None')}")
                print(f"ModelCheckpoint save_top_k: {callback.save_top_k}")