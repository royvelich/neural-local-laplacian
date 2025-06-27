# standard library
import importlib
from typing import Type, Callable, List, Tuple, Optional, Union, Optional, Literal, Dict
import inspect

import scipy.sparse
# torch
import torch
import torch.nn.functional as F

# numpy
import numpy as np
from scipy import sparse

# scipy
from scipy.spatial import cKDTree

# open3d
import open3d as o3d

# python-shot
# import handcrafted_descriptor as hd

# sklearn
from sklearn.neighbors import NearestNeighbors

# igl
import igl

# delta-conv
from deltaconv.geometry import build_tangent_basis, build_grad_div, estimate_basis

# pyfm
from pyFM.mesh import TriMesh

# torch geometric
from torch_geometric.data import Batch


def centroid_to_origin(points: np.ndarray) -> np.ndarray:
    centroid = np.mean(points, axis=0, keepdims=True)
    centered_points = points - centroid
    return centered_points


def normalize_to_unit_sphere(points: np.ndarray) -> np.ndarray:
    """
    Rescales a point cloud to fit within a unit sphere centered at the origin.

    Args:
        points (np.ndarray): Point cloud array of shape (K, 3) where K is the number of points

    Returns:
        np.ndarray: Normalized point cloud of shape (K, 3) fitting within a unit sphere

    Raises:
        ValueError: If input array doesn't have shape (K, 3)
    """
    points = centroid_to_origin(points=points)

    # Find the maximum distance from the origin to any point
    distances = np.linalg.norm(points, axis=1)
    max_distance = np.max(distances)

    # Scale the points to fit within a unit sphere
    normalized_points = points / max_distance

    return normalized_points


def normalize_mesh_vertices(vertices: np.ndarray) -> np.ndarray:
    """
    Normalize mesh vertices to be centered at origin and fit within unit sphere.

    This function ensures consistent mesh scaling and positioning across different
    mesh files, making them suitable for comparative analysis and feature extraction.

    Args:
        vertices: Raw mesh vertices of shape (N, 3)

    Returns:
        Normalized vertices of shape (N, 3) where:
        - Center of mass is at origin (0, 0, 0)
        - All vertices fit within unit sphere (max distance = 1.0)

    Raises:
        ValueError: If vertices array is empty or has wrong shape
    """
    if vertices.size == 0:
        raise ValueError("Cannot normalize empty vertices array")

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices shape (N, 3), got {vertices.shape}")

    # Center vertices at origin (center of mass at origin)
    centroid = np.mean(vertices, axis=0)
    centered_vertices = vertices - centroid

    # Scale to fit within unit sphere
    # Find the maximum distance from origin to any vertex
    distances = np.linalg.norm(centered_vertices, axis=1)
    max_distance = np.max(distances)

    if max_distance > 0:
        # Scale so that the farthest vertex is on the unit sphere
        normalized_vertices = centered_vertices / max_distance
    else:
        # Handle degenerate case (all vertices at same point)
        normalized_vertices = centered_vertices

    return normalized_vertices


def split_results_by_nodes(results: torch.Tensor, batch: Batch) -> List[torch.Tensor]:
    return [results[batch.batch == i] for i in range(batch.num_graphs)]


def split_results_by_graphs(results: torch.Tensor, batch: Batch) -> List[torch.Tensor]:
    return [results[i] for i in range(batch.num_graphs)]


def rebuild_batch_from_list(batch: Batch, property_name: str, property_tensor_list: List[torch.Tensor]) -> Batch:
    data_list = batch.to_data_list()
    for data, tensor in zip(data_list, property_tensor_list):
        data[property_name] = tensor
    return Batch.from_data_list(data_list)


def rebuild_batch_from_tensor(batch: Batch, property_name: str, property_tensor: torch.Tensor) -> Batch:
    tensor_list = property_tensor.split(split_size=batch.batch.bincount().tolist(), dim=0)
    return rebuild_batch_from_list(batch=batch, property_name=property_name, property_tensor_list=tensor_list)


def rebuild_batch_from_dictionary_of_lists(batch: Batch, property_dict: Dict[str, List[torch.Tensor]]) -> Batch:
    for property_name, property_tensor_list in property_dict.items():
        batch = rebuild_batch_from_list(batch=batch, property_name=property_name, property_tensor_list=property_tensor_list)
    return batch
