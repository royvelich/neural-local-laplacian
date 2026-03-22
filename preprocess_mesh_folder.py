#!/usr/bin/env python3
"""
Preprocess a mesh dataset: build lookup table + precompute exact geodesics.

Scans mesh folders, applies filters (file size, vertex/face count, components),
builds .mesh_properties_cache.json, then computes FPS source vertices and exact
geodesic distances (MMP algorithm) per mesh.

Must be run BEFORE MeshDataset or quantitative_eval.py, which expect the
lookup table and geodesic cache to exist.

Uses multiprocessing.Pool with maxtasksperchild=1 for crash isolation:
if pygeodesic segfaults on a bad mesh, only that worker dies.

Usage:
    python preprocess_mesh_folder.py dataset=shrec15
    python preprocess_mesh_folder.py dataset=thingi10k num_sources=10 num_workers=16
    python preprocess_mesh_folder.py dataset=shrec15 dataset.folder=/custom/path
"""

import time
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import hydra
from omegaconf import DictConfig

from neural_local_laplacian.utils.utils import (
    normalize_mesh_vertices,
    scan_mesh_folders,
    load_mesh_lookup_table,
    save_mesh_lookup_table,
)
from neural_local_laplacian.utils.geodesic_utils import (
    compute_exact_geodesics,
    select_multiple_geodesic_sources,
    load_cached_geodesics,
    save_geodesics_to_cache,
)


def _compute_one_mesh(args) -> Tuple[str, str, bool, int, int, str]:
    """
    Worker: load mesh, compute exact geodesics, save to cache.

    Runs in a Pool with maxtasksperchild=1 so each mesh gets a fresh
    process. If pygeodesic segfaults, only this worker dies.

    Returns:
        (mesh_file_path, mesh_name, success, n_ok, num_sources, message)
    """
    mesh_file_path, num_sources, seed = args
    mesh_name = Path(mesh_file_path).name

    try:
        import trimesh
        mesh = trimesh.load(str(mesh_file_path), process=True, force='mesh')
        vertices = np.array(mesh.vertices, dtype=np.float64)
        vertices = normalize_mesh_vertices(vertices)
        faces = np.array(mesh.faces, dtype=np.int32)
        N = len(vertices)

        if N == 0 or len(faces) == 0:
            return mesh_file_path, mesh_name, False, 0, num_sources, \
                f"empty mesh (N={N}, F={len(faces)})"

        source_indices = select_multiple_geodesic_sources(
            vertices, num_sources=num_sources,
            method="farthest_point_sampling", seed=seed,
        )

        # Skip if already cached and valid
        cached = load_cached_geodesics(mesh_file_path, N, source_indices)
        if cached is not None:
            n_ok = sum(1 for v in cached.values() if v is not None)
            return mesh_file_path, mesh_name, True, n_ok, num_sources, \
                f"cached ({n_ok}/{num_sources} sources)"

        # Compute exact geodesics
        geodesics = {}
        for src_idx in source_indices:
            try:
                geodesics[int(src_idx)] = compute_exact_geodesics(
                    vertices, faces, int(src_idx),
                )
            except Exception:
                geodesics[int(src_idx)] = None

        n_ok = sum(1 for v in geodesics.values() if v is not None)

        if n_ok > 0:
            save_geodesics_to_cache(mesh_file_path, N, source_indices, geodesics)
            return mesh_file_path, mesh_name, True, n_ok, num_sources, \
                f"computed ({n_ok}/{num_sources} sources, N={N})"
        else:
            return mesh_file_path, mesh_name, False, 0, num_sources, \
                f"all sources failed (N={N}, F={len(faces)})"

    except Exception as e:
        return mesh_file_path, mesh_name, False, 0, num_sources, str(e)


@hydra.main(version_base="1.2", config_path='./mesh_preprocess_config', config_name='config')
def main(cfg: DictConfig) -> None:
    """Preprocess mesh folder: build lookup table + precompute geodesics."""

    ds = cfg.dataset
    folder_paths = [Path(ds.folder)]
    num_sources = getattr(ds, 'num_sources', 10)
    seed = getattr(ds, 'seed', 42)
    num_workers = getattr(ds, 'num_workers', 8)

    # Parse filter ranges from dataset config (None if not set)
    file_size_range = tuple(ds.file_size_range_mb) if getattr(ds, 'file_size_range_mb', None) is not None else None
    vertices_range = tuple(ds.vertices_count_range) if getattr(ds, 'vertices_count_range', None) is not None else None
    faces_range = tuple(ds.faces_count_range) if getattr(ds, 'faces_count_range', None) is not None else None
    components_range = tuple(ds.num_components_range) if getattr(ds, 'num_components_range', None) is not None else None

    print(f"\n{'=' * 70}")
    print(f"PREPROCESS MESH FOLDER")
    print(f"{'=' * 70}")
    print(f"Folder:      {ds.folder}")
    if file_size_range:
        print(f"File size:   {file_size_range[0]:.2f} - {file_size_range[1]:.2f} MB")
    if vertices_range:
        print(f"Vertices:    {vertices_range[0]} - {vertices_range[1]}")
    if faces_range:
        print(f"Faces:       {faces_range[0]} - {faces_range[1]}")
    if components_range:
        print(f"Components:  {components_range[0]} - {components_range[1]}")
    print(f"Sources:     {num_sources}")
    print(f"Seed:        {seed}")
    print(f"Workers:     {num_workers}")
    print(f"{'=' * 70}\n")

    # Step 1: Scan + filter + build lookup table
    print("Step 1: Scanning and filtering meshes...")
    mesh_files = scan_mesh_folders(
        folder_paths=folder_paths,
        file_size_range_mb=file_size_range,
        vertices_count_range=vertices_range,
        faces_count_range=faces_range,
        num_components_range=components_range,
        max_meshes=None,
        shuffle=False,
    )
    print(f"Found {len(mesh_files)} meshes after filtering.\n")

    if len(mesh_files) == 0:
        print("No meshes to process.")
        return

    # Step 2: Precompute exact geodesics
    print(f"Step 2: Precomputing exact geodesics "
          f"({num_sources} sources per mesh)...")
    worker_args = [(str(f), num_sources, seed) for f in mesh_files]
    t_start = time.time()

    if num_workers <= 1:
        results = []
        for i, a in enumerate(worker_args):
            r = _compute_one_mesh(a)
            results.append(r)
            status = "OK" if r[2] else "FAIL"
            print(f"  [{i+1}/{len(worker_args)}] {r[1]:<24s} "
                  f"{status:>4s}  {r[5]}")
    else:
        import multiprocessing as mp
        with mp.Pool(num_workers, maxtasksperchild=1) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(
                _compute_one_mesh, worker_args,
            )):
                results.append(r)
                status = "OK" if r[2] else "FAIL"
                print(f"  [{i+1}/{len(worker_args)}] {r[1]:<24s} "
                      f"{status:>4s}  {r[5]}")

    elapsed = time.time() - t_start

    # Step 3: Update lookup table with geodesic status
    print(f"\nStep 3: Updating lookup table with geodesic status...")
    lookup = load_mesh_lookup_table(folder_paths)
    updated = 0
    for fpath, name, ok, n_ok_i, n_src, msg in results:
        key = str(fpath)
        if key in lookup:
            lookup[key]['geodesics_num_sources'] = n_src
            lookup[key]['geodesics_num_ok'] = n_ok_i
            updated += 1
    save_mesh_lookup_table(lookup, folder_paths)
    print(f"  Updated {updated} entries.")

    n_ok = sum(1 for _, _, ok, _, _, _ in results if ok)
    n_fail = sum(1 for _, _, ok, _, _, _ in results if not ok)

    print(f"\n{'=' * 70}")
    print(f"DONE: {n_ok} succeeded, {n_fail} failed, {elapsed:.1f}s")
    print(f"{'=' * 70}")

    if n_fail > 0:
        print(f"\nFailed meshes:")
        for fpath, name, ok, n_ok_i, n_src, msg in results:
            if not ok:
                print(f"  {name:<24s} {msg}")


if __name__ == '__main__':
    main()