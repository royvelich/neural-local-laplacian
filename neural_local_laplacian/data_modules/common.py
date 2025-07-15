# Standard library imports
from typing import Optional, List
from dataclasses import dataclass

# Third-party library imports
import lightning
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data

# neural signatures
from neural_local_laplacian.utils import utils


@dataclass
class DatasetSpecification:
    dataset: Dataset
    batch_size: int
    num_workers: int
    shuffle: bool


class GenericDataModule(lightning.pytorch.LightningDataModule):
    """
    PyTorch Lightning DataModule for handling Polynomial Surface datasets.

    This module manages the creation of train and validation dataloaders
    for synthetic polynomial surface data.

    Attributes:
        _train_dataset (SyntheticSurfaceDataset): Dataset for training.
        _val_dataset (SyntheticSurfaceDataset): Dataset for validation.
        _batch_size (int): Number of samples per batch.
        _num_workers (int): Number of subprocesses to use for data loading.
        _shuffle (bool): Whether to shuffle the data during training.
    """

    def __init__(
            self,
            train_dataset_specification: DatasetSpecification,
            val_dataset_specifications: List[DatasetSpecification],
    ) -> None:
        """
        Initialize the PolynomialSurfaceDataModule.

        Args:
            train_synthetic_dataset (SyntheticSurfaceDataset): Dataset for training.
            val_dataset (SyntheticSurfaceDataset): Dataset for validation.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of subprocesses to use for data loading.
            shuffle (bool, optional): Whether to shuffle the data during training. Defaults to True.
        """
        super().__init__()
        self._train_dataset_specification = train_dataset_specification
        self._val_dataset_specifications = val_dataset_specifications

    def train_dataloader(self) -> DataLoader:
        """
        Create and return the train dataloader.

        Returns:
            DataLoader: The dataloader for training data.
        """
        return DataLoader(
            dataset=self._train_dataset_specification.dataset,
            batch_size=self._train_dataset_specification.batch_size,
            shuffle=self._train_dataset_specification.shuffle,
            num_workers=self._train_dataset_specification.num_workers,
            persistent_workers=self._train_dataset_specification.num_workers > 0
        )

    def val_dataloader(self) -> List[DataLoader]:
        """
        Create and return the validation dataloader.

        Returns:
            DataLoader: The dataloader for validation data.
        """
        return [
            DataLoader(
                dataset=val_dataset_specification.dataset,
                batch_size=val_dataset_specification.batch_size,
                shuffle=val_dataset_specification.shuffle,
                num_workers=val_dataset_specification.num_workers,
                persistent_workers=val_dataset_specification.num_workers > 0
            ) for val_dataset_specification in self._val_dataset_specifications
        ]

