# Standard library imports
from typing import Optional, List, Callable

# Third-party library imports
import lightning
import torch.utils.data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Dataset as PyGDataset

# neural signatures
from neural_local_laplacian.utils import utils


# ---------------------------------------------------------------------------
# Dataset specification
# ---------------------------------------------------------------------------

class DatasetSpecification:
    """Bundles a dataset with its DataLoader settings.

    Args:
        dataset:     The dataset instance (PyG or plain torch).
        batch_size:  Items per batch.
        num_workers: DataLoader worker processes (0 = main process).
        shuffle:     Whether to shuffle each epoch.
        collate_fn:  Optional collate function (plain DataLoader only;
                     ignored by GenericPygDataModule which uses PyG's DataLoader).
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        collate_fn: Optional[Callable] = None,
    ):
        self.dataset     = dataset
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.shuffle     = shuffle
        self.collate_fn  = collate_fn


# ---------------------------------------------------------------------------
# Base: shared __init__ and structure, no DataLoader imports
# ---------------------------------------------------------------------------

class _GenericDataModuleBase(lightning.pytorch.LightningDataModule):
    """Internal base: stores specifications, leaves DataLoader construction
    to subclasses that know which loader to use."""

    def __init__(
        self,
        train_dataset_specification: Optional[DatasetSpecification] = None,
        val_dataset_specifications: Optional[List[DatasetSpecification]] = None,
    ) -> None:
        super().__init__()
        self._train_dataset_specification = train_dataset_specification
        self._val_dataset_specifications  = val_dataset_specifications or []

    def _make_train_loader(self, spec: DatasetSpecification):
        raise NotImplementedError

    def _make_val_loader(self, spec: DatasetSpecification):
        raise NotImplementedError

    def train_dataloader(self):
        if self._train_dataset_specification is None:
            return None
        return self._make_train_loader(self._train_dataset_specification)

    def val_dataloader(self) -> List:
        return [self._make_val_loader(s) for s in self._val_dataset_specifications]


# ---------------------------------------------------------------------------
# PyG DataLoader subclass  (existing behaviour, unchanged)
# ---------------------------------------------------------------------------

class GenericPygDataModule(_GenericDataModuleBase):
    """DataModule backed by PyTorch Geometric's DataLoader.

    Use this when datasets yield ``torch_geometric.data.Data`` objects that
    need PyG's automatic batching (stacking node/edge features into a single
    large disconnected graph).
    """

    def _make_loader(self, spec: DatasetSpecification) -> PyGDataLoader:
        return PyGDataLoader(
            dataset=spec.dataset,
            batch_size=spec.batch_size,
            shuffle=spec.shuffle,
            num_workers=spec.num_workers,
            persistent_workers=spec.num_workers > 0,
        )

    def _make_train_loader(self, spec: DatasetSpecification) -> PyGDataLoader:
        return self._make_loader(spec)

    def _make_val_loader(self, spec: DatasetSpecification) -> PyGDataLoader:
        return self._make_loader(spec)


# ---------------------------------------------------------------------------
# Plain torch DataLoader subclass  (new, for non-PyG datasets)
# ---------------------------------------------------------------------------

class GenericPlainDataModule(_GenericDataModuleBase):
    """DataModule backed by standard ``torch.utils.data.DataLoader``.

    Use this when datasets yield arbitrary Python objects (e.g. plain
    dataclasses, numpy arrays, variable-size meshes) that cannot be
    batched by PyG's collator.  Pass a ``collate_fn`` in the
    ``DatasetSpecification`` to control how items are batched.
    """

    def _make_loader(self, spec: DatasetSpecification) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=spec.dataset,
            batch_size=spec.batch_size,
            shuffle=spec.shuffle,
            num_workers=spec.num_workers,
            persistent_workers=spec.num_workers > 0,
            collate_fn=spec.collate_fn,
        )

    def _make_train_loader(self, spec: DatasetSpecification) -> torch.utils.data.DataLoader:
        return self._make_loader(spec)

    def _make_val_loader(self, spec: DatasetSpecification) -> torch.utils.data.DataLoader:
        return self._make_loader(spec)

# ---------------------------------------------------------------------------
# Mixed DataLoader subclass  (auto-selects PyG vs plain per spec)
# ---------------------------------------------------------------------------

class GenericMixedDataModule(_GenericDataModuleBase):
    """DataModule that auto-selects PyG or plain DataLoader per specification.

    If a ``DatasetSpecification`` has a ``collate_fn``, it uses a standard
    ``torch.utils.data.DataLoader`` (for non-PyG datasets like shape pairs).
    Otherwise, it uses PyG's ``DataLoader`` (for PyG Data/Batch datasets).

    This allows mixing PyG mesh validation datasets with plain pair datasets
    in the same training pipeline.
    """

    def _make_loader(self, spec: DatasetSpecification):
        if spec.collate_fn is not None:
            return torch.utils.data.DataLoader(
                dataset=spec.dataset,
                batch_size=spec.batch_size,
                shuffle=spec.shuffle,
                num_workers=spec.num_workers,
                persistent_workers=spec.num_workers > 0,
                collate_fn=spec.collate_fn,
            )
        else:
            return PyGDataLoader(
                dataset=spec.dataset,
                batch_size=spec.batch_size,
                shuffle=spec.shuffle,
                num_workers=spec.num_workers,
                persistent_workers=spec.num_workers > 0,
            )

    def _make_train_loader(self, spec: DatasetSpecification):
        return self._make_loader(spec)

    def _make_val_loader(self, spec: DatasetSpecification):
        return self._make_loader(spec)
