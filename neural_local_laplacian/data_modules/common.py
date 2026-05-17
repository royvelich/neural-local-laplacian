# Standard library imports
from typing import Optional, List, Callable

# Third-party library imports
import lightning
import numpy as np
import torch.utils.data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Dataset as PyGDataset

# neural signatures
from neural_local_laplacian.utils import utils


# ---------------------------------------------------------------------------
# Worker init: dataset-RNG reseed for reproducibility
# ---------------------------------------------------------------------------

try:
    from lightning.fabric.utilities.seed import (
        pl_worker_init_function as _pl_worker_init_fn,
    )
except ImportError:  # older Lightning
    try:
        from pytorch_lightning.utilities.seed import (
            pl_worker_init_function as _pl_worker_init_fn,
        )
    except ImportError:
        _pl_worker_init_fn = None


def dataset_worker_init_fn(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` that fixes per-worker dataset RNG.

    Datasets like :class:`MongeSurfaceVariationalDataset` keep a private
    ``self._rng = np.random.default_rng(seed)`` set at construction time.
    When the loader uses ``num_workers > 0`` each worker forks its own copy
    of that RNG, and which worker handles which ``get(idx)`` call is
    determined by OS scheduling — so the *content* of batch ``i`` is not
    reproducible across runs and curves diverge between identical configs.

    Lightning's ``seed_everything(workers=True)`` only reseeds the global
    ``numpy``/``torch``/``random`` RNGs in each worker; the dataset's own
    ``_rng`` instance is untouched.  This function:

      1. Delegates to Lightning's worker init (if available) so the global
         RNGs are still seeded per-worker per-epoch.
      2. Reseeds the dataset's own ``_rng`` to
         ``default_rng(dataset._seed + worker_id)``, making the per-worker
         sample sequence reproducible across runs (and across persistent-
         worker epochs).

    Datasets without ``_seed`` / ``_rng`` are silently skipped.
    """
    if _pl_worker_init_fn is not None:
        _pl_worker_init_fn(worker_id)

    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    ds = info.dataset
    if hasattr(ds, '_seed') and hasattr(ds, '_rng'):
        ds._rng = np.random.default_rng(int(ds._seed) + int(worker_id))


# ---------------------------------------------------------------------------
# k-grouped batch sampler (variable-k variational batching)
# ---------------------------------------------------------------------------

class KGroupedBatchSampler(torch.utils.data.Sampler):
    """Yields lists of dataset indices grouped by matching ``k``.

    Cross-surface batching in :class:`_VariationalSurfaceData` requires
    uniform patch size ``k`` across every surface in a batch — the
    ``knn``, ``x``, ``pos`` tensors concatenate on dim 0 and need
    matching trailing dim.  When a dataset draws ``k`` from a range, a
    naive ``shuffle=True`` loader will mix surfaces with different ``k``
    in the same batch and crash at collation.

    This sampler reads each index's ``k`` via ``dataset.k_for_idx(idx)``,
    buckets indices by ``k`` once at construction (assumes the assignment
    is stable for the dataset's lifetime — true for
    :class:`MongeSurfaceVariationalDataset` which pre-samples ``k_per_idx``
    at ``__init__``), and each epoch yields contiguous batches of
    ``batch_size`` drawn from a single bucket.  Every emitted batch is
    therefore ``k``-homogeneous and PyG collation succeeds.

    Args:
        dataset:    Must expose ``__len__`` and ``k_for_idx(idx) -> int``.
        batch_size: Items per batch.
        shuffle:    Shuffle within each ``k`` bucket each epoch and
                    shuffle the order of emitted batches.
        drop_last:  Drop any bucket tail smaller than ``batch_size``.
                    When False (default), the tail is emitted as a
                    smaller (still ``k``-homogeneous) batch.
        seed:       Base seed for the epoch generator; each call to
                    ``__iter__`` advances the epoch counter for
                    reproducible shuffles across runs.
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool = True,
                 drop_last: bool = False, seed: int = 0) -> None:
        if not hasattr(dataset, 'k_for_idx'):
            raise TypeError(
                f"KGroupedBatchSampler requires the dataset to expose "
                f"k_for_idx(idx) -> int. {type(dataset).__name__} does not.")
        if int(batch_size) < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._dataset = dataset
        self._batch_size = int(batch_size)
        self._shuffle = bool(shuffle)
        self._drop_last = bool(drop_last)
        self._seed = int(seed)
        self._epoch = 0

        buckets: dict = {}
        for i in range(len(dataset)):
            k = int(dataset.k_for_idx(i))
            buckets.setdefault(k, []).append(i)
        self._buckets = {k: np.asarray(v, dtype=np.int64)
                         for k, v in buckets.items()}

    def __iter__(self):
        rng = np.random.default_rng(self._seed + self._epoch)
        self._epoch += 1

        all_batches: List[List[int]] = []
        for indices in self._buckets.values():
            arr = indices.copy()
            if self._shuffle:
                rng.shuffle(arr)
            n = arr.shape[0]
            n_full = n // self._batch_size
            for b in range(n_full):
                all_batches.append(
                    arr[b * self._batch_size:(b + 1) * self._batch_size].tolist())
            tail = arr[n_full * self._batch_size:]
            if tail.shape[0] > 0 and not self._drop_last:
                all_batches.append(tail.tolist())

        if self._shuffle:
            order = rng.permutation(len(all_batches))
            all_batches = [all_batches[i] for i in order]
        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        total = 0
        for indices in self._buckets.values():
            n = indices.shape[0]
            if self._drop_last:
                total += n // self._batch_size
            else:
                total += (n + self._batch_size - 1) // self._batch_size
        return total


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
        group_by_k:  If True, wire a :class:`KGroupedBatchSampler` so
                     every batch is k-homogeneous.  Requires the dataset
                     to expose ``k_for_idx(idx) -> int`` (e.g.
                     :class:`MongeSurfaceVariationalDataset`).  Use this
                     when the dataset draws ``k`` from a range and you
                     want ``batch_size > 1`` — without it, mixed-k
                     batches crash at PyG collation.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        collate_fn: Optional[Callable] = None,
        group_by_k: bool = False,
    ):
        self.dataset     = dataset
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.shuffle     = shuffle
        self.collate_fn  = collate_fn
        self.group_by_k  = bool(group_by_k)


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
        wif = dataset_worker_init_fn if spec.num_workers > 0 else None
        if spec.group_by_k:
            sampler = KGroupedBatchSampler(
                dataset=spec.dataset,
                batch_size=spec.batch_size,
                shuffle=spec.shuffle,
                drop_last=False,
                seed=int(getattr(spec.dataset, '_seed', 0)),
            )
            return PyGDataLoader(
                dataset=spec.dataset,
                batch_sampler=sampler,
                num_workers=spec.num_workers,
                persistent_workers=spec.num_workers > 0,
                worker_init_fn=wif,
            )
        return PyGDataLoader(
            dataset=spec.dataset,
            batch_size=spec.batch_size,
            shuffle=spec.shuffle,
            num_workers=spec.num_workers,
            persistent_workers=spec.num_workers > 0,
            worker_init_fn=wif,
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
        wif = dataset_worker_init_fn if spec.num_workers > 0 else None
        if spec.group_by_k:
            sampler = KGroupedBatchSampler(
                dataset=spec.dataset,
                batch_size=spec.batch_size,
                shuffle=spec.shuffle,
                drop_last=False,
                seed=int(getattr(spec.dataset, '_seed', 0)),
            )
            return torch.utils.data.DataLoader(
                dataset=spec.dataset,
                batch_sampler=sampler,
                num_workers=spec.num_workers,
                persistent_workers=spec.num_workers > 0,
                collate_fn=spec.collate_fn,
                worker_init_fn=wif,
            )
        return torch.utils.data.DataLoader(
            dataset=spec.dataset,
            batch_size=spec.batch_size,
            shuffle=spec.shuffle,
            num_workers=spec.num_workers,
            persistent_workers=spec.num_workers > 0,
            collate_fn=spec.collate_fn,
            worker_init_fn=wif,
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
        wif = dataset_worker_init_fn if spec.num_workers > 0 else None
        sampler = None
        if spec.group_by_k:
            sampler = KGroupedBatchSampler(
                dataset=spec.dataset,
                batch_size=spec.batch_size,
                shuffle=spec.shuffle,
                drop_last=False,
                seed=int(getattr(spec.dataset, '_seed', 0)),
            )
        if spec.collate_fn is not None:
            kwargs = dict(
                dataset=spec.dataset,
                num_workers=spec.num_workers,
                persistent_workers=spec.num_workers > 0,
                collate_fn=spec.collate_fn,
                worker_init_fn=wif,
            )
            if sampler is not None:
                kwargs['batch_sampler'] = sampler
            else:
                kwargs.update(batch_size=spec.batch_size, shuffle=spec.shuffle)
            return torch.utils.data.DataLoader(**kwargs)
        else:
            kwargs = dict(
                dataset=spec.dataset,
                num_workers=spec.num_workers,
                persistent_workers=spec.num_workers > 0,
                worker_init_fn=wif,
            )
            if sampler is not None:
                kwargs['batch_sampler'] = sampler
            else:
                kwargs.update(batch_size=spec.batch_size, shuffle=spec.shuffle)
            return PyGDataLoader(**kwargs)

    def _make_train_loader(self, spec: DatasetSpecification):
        return self._make_loader(spec)

    def _make_val_loader(self, spec: DatasetSpecification):
        return self._make_loader(spec)