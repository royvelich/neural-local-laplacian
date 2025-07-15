import torch
import numpy as np
import scipy.sparse as sp
from typing import Union, List, Optional, Tuple
from collections import defaultdict


class ARAPDeformer:
    """Public interface for ARAPDeformer class"""

    def __init__(
            self,
            vertices: Union[torch.Tensor, np.ndarray],
            faces: Union[torch.Tensor, np.ndarray, None],
            device: Optional[torch.device] = None,
            precomputed_laplacian: Optional[sp.csr_matrix] = None
    ) -> None:
        """
        Initialize ARAP deformer.

        Args:
            vertices: Vertex positions (N, 3)
            faces: Face indices (F, 3) or None
            device: Torch device to use
            precomputed_laplacian: Optional precomputed Laplacian matrix
        """
        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Convert inputs to tensors and ensure consistent dtype
        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices)
        # Ensure float32 dtype for consistency
        if vertices.dtype == torch.float64:
            vertices = vertices.float()
        elif vertices.dtype not in [torch.float32, torch.float64]:
            vertices = vertices.float()

        self._original_vertices = vertices.to(device)
        self._current_vertices = self._original_vertices.clone()

        if faces is not None:
            if isinstance(faces, np.ndarray):
                faces = torch.from_numpy(faces)
            if faces.dtype != torch.long:
                faces = faces.long()
            self._faces = faces.to(device)
        else:
            self._faces = None

        self.n_vertices = vertices.shape[0]

        # Initialize handle and fixed indices
        self.handle_indices = []
        self.fixed_indices = []

        # Compute one-ring neighbors
        self.one_ring_neighbors = self._compute_one_ring_neighbors()

        # Store precomputed Laplacian if provided
        self.precomputed_laplacian = precomputed_laplacian
        self._laplacian_matrix = None
        self._laplacian_factorized = None

        if precomputed_laplacian is not None:
            # Use precomputed Laplacian - convert to torch tensor if needed
            if isinstance(precomputed_laplacian, sp.csr_matrix):
                # Convert sparse matrix to dense torch tensor
                dense_lap = torch.from_numpy(precomputed_laplacian.toarray()).to(device)
                if dense_lap.dtype != self._original_vertices.dtype:
                    dense_lap = dense_lap.to(dtype=self._original_vertices.dtype)
                self._laplacian_matrix = dense_lap
            else:
                # Assume it's already a torch tensor
                self._laplacian_matrix = precomputed_laplacian.to(device)
                if self._laplacian_matrix.dtype != self._original_vertices.dtype:
                    self._laplacian_matrix = self._laplacian_matrix.to(dtype=self._original_vertices.dtype)

            # For precomputed Laplacian, we need to derive edge weights from the matrix
            self.w_nfmt = self._derive_weights_from_laplacian()
        else:
            # Compute cotangent weights from mesh geometry
            self.w_nfmt = self._compute_cotangent_weights()

        # Precompute indices for efficient computation
        self.ii, self.jj, self.nn = self._produce_idxs()

    def _compute_one_ring_neighbors(self):
        """Compute one-ring neighbors for each vertex"""
        if self._faces is None:
            # If no faces, assume complete graph or return empty
            return {i: [] for i in range(self.n_vertices)}

        neighbors = defaultdict(set)
        faces_np = self._faces.cpu().numpy()

        for face in faces_np:
            for i in range(3):
                v1, v2, v3 = face[i], face[(i + 1) % 3], face[(i + 2) % 3]
                neighbors[v1].add(v2)
                neighbors[v1].add(v3)

        # Convert to lists
        return {k: list(v) for k, v in neighbors.items()}

    def _derive_weights_from_laplacian(self):
        """Derive edge weights from precomputed Laplacian matrix"""
        max_neighbors = max(len(neighbors) for neighbors in self.one_ring_neighbors.values()) if self.one_ring_neighbors else 1
        w_nfmt = torch.zeros((self.n_vertices, max_neighbors), dtype=self._original_vertices.dtype, device=self.device)

        # Extract edge weights from Laplacian matrix
        # For Laplacian L: L_ij = -w_ij (off-diagonal), L_ii = sum_j w_ij (diagonal)
        L = self._laplacian_matrix

        for i in range(self.n_vertices):
            neighbors = self.one_ring_neighbors[i]
            for n, j in enumerate(neighbors):
                # Edge weight is -L_ij
                w_nfmt[i, n] = -L[i, j]

        return w_nfmt

    def _compute_cotangent_weights(self):
        """Compute cotangent weights in nfmt format"""
        if self._faces is None:
            # Return uniform weights if no faces
            max_neighbors = max(len(neighbors) for neighbors in self.one_ring_neighbors.values()) if self.one_ring_neighbors else 1
            return torch.ones((self.n_vertices, max_neighbors), dtype=self._original_vertices.dtype, device=self.device)

        # Compute cotangent weights
        max_neighbors = max(len(neighbors) for neighbors in self.one_ring_neighbors.values())
        w_nfmt = torch.zeros((self.n_vertices, max_neighbors), dtype=self._original_vertices.dtype, device=self.device)

        # Get full cotangent weights matrix
        w_full = self._get_cotangent_weights_full()

        # Convert to nfmt format
        for i in range(self.n_vertices):
            neighbors = self.one_ring_neighbors[i]
            for n, j in enumerate(neighbors):
                w_nfmt[i, n] = w_full[i, j]

        return w_nfmt

    def _get_cotangent_weights_full(self):
        """Compute full cotangent weights matrix"""
        if self._faces is None:
            return torch.eye(self.n_vertices, dtype=self._original_vertices.dtype, device=self.device)

        n_faces = self._faces.shape[0]
        vertices = self._original_vertices
        faces = self._faces

        # Get face vertices
        face_verts = vertices[faces]  # (F, 3, 3)
        v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

        # Compute edge lengths
        A = torch.norm(v1 - v2, dim=1)  # opposite to v0
        B = torch.norm(v0 - v2, dim=1)  # opposite to v1
        C = torch.norm(v0 - v1, dim=1)  # opposite to v2

        # Compute area using Heron's formula
        s = 0.5 * (A + B + C)
        area = torch.sqrt(torch.clamp(s * (s - A) * (s - B) * (s - C), min=1e-12))

        # Compute cotangents
        A2, B2, C2 = A * A, B * B, C * C
        cot_a = (B2 + C2 - A2) / (4.0 * area)  # angle at v0
        cot_b = (A2 + C2 - B2) / (4.0 * area)  # angle at v1
        cot_c = (A2 + B2 - C2) / (4.0 * area)  # angle at v2

        # Build sparse matrix
        W = torch.zeros((self.n_vertices, self.n_vertices), dtype=self._original_vertices.dtype, device=self.device)

        # Add cotangent weights
        for f in range(n_faces):
            i, j, k = faces[f]
            W[j, k] += 0.5 * cot_a[f]  # weight for edge jk from angle at i
            W[k, j] += 0.5 * cot_a[f]
            W[i, k] += 0.5 * cot_b[f]  # weight for edge ik from angle at j
            W[k, i] += 0.5 * cot_b[f]
            W[i, j] += 0.5 * cot_c[f]  # weight for edge ij from angle at k
            W[j, i] += 0.5 * cot_c[f]

        return W

    def _produce_idxs(self):
        """Produce flattened index arrays for efficient computation"""
        ii, jj, nn = [], [], []

        for i in range(self.n_vertices):
            neighbors = self.one_ring_neighbors[i]
            for n, j in enumerate(neighbors):
                ii.append(i)
                jj.append(j)
                nn.append(n)

        ii = torch.tensor(ii, dtype=torch.long, device=self.device)
        jj = torch.tensor(jj, dtype=torch.long, device=self.device)
        nn = torch.tensor(nn, dtype=torch.long, device=self.device)

        return ii, jj, nn

    def _produce_edge_matrix_nfmt(self, vertices):
        """Produce edge matrix in nfmt format"""
        max_neighbors = self.w_nfmt.shape[1]
        # Ensure edge matrix has same dtype as vertices
        edge_matrix = torch.zeros((self.n_vertices, max_neighbors, 3),
                                  dtype=vertices.dtype, device=self.device)

        if len(self.ii) > 0:
            edge_matrix[self.ii, self.nn] = vertices[self.ii] - vertices[self.jj]

        return edge_matrix

    def _compute_rotations(self, p_original, p_deformed):
        """Compute optimal rotations for each vertex"""
        # Compute edge matrices
        P = self._produce_edge_matrix_nfmt(p_original)
        P_prime = self._produce_edge_matrix_nfmt(p_deformed)

        # Compute covariance matrices
        D = torch.diag_embed(self.w_nfmt, dim1=1, dim2=2)
        S = torch.bmm(P.permute(0, 2, 1), torch.bmm(D, P_prime))

        # Handle unchanged vertices
        unchanged_verts = torch.unique(torch.where((P == P_prime).all(dim=2).all(dim=1))[0])
        if len(unchanged_verts) > 0:
            S[unchanged_verts] = 0

        # SVD decomposition
        try:
            U, sig, Vh = torch.linalg.svd(S)
            V = Vh.transpose(-2, -1)
        except:
            # Fallback to standard SVD
            U, sig, V = torch.svd(S)

        R = torch.bmm(V, U.transpose(-2, -1))

        # Ensure proper rotation (det(R) > 0)
        det_R = torch.det(R)
        flip_mask = det_R <= 0

        if flip_mask.any():
            U_mod = U.clone()
            # Flip the column corresponding to smallest singular value
            min_sig_idx = torch.argmin(sig[flip_mask], dim=1)
            U_mod[flip_mask, :, min_sig_idx] *= -1
            R[flip_mask] = torch.bmm(V[flip_mask], U_mod[flip_mask].transpose(-2, -1))

        return R

    def _build_laplacian_system(self, rotations):
        """Build the Laplacian system for position update"""
        # Build Laplacian matrix if not cached
        if self._laplacian_matrix is None:
            self._build_laplacian_matrix()

        # Compute right-hand side
        max_neighbors = self.w_nfmt.shape[1]
        b = torch.zeros((self.n_vertices, 3), dtype=self._original_vertices.dtype, device=self.device)

        # Compute Rsum (Ri + Rj) for each edge
        Rsum_shape = (self.n_vertices, max_neighbors, 3, 3)
        Rsum = torch.zeros(Rsum_shape, dtype=self._original_vertices.dtype, device=self.device)

        if len(self.ii) > 0:
            Rsum[self.ii, self.nn] = rotations[self.ii] + rotations[self.jj]

        # Compute edge vectors from original mesh
        P = self._produce_edge_matrix_nfmt(self._original_vertices)

        # Compute right-hand side: 0.5 * sum_j w_ij * (Ri + Rj) * (pi - pj)
        Rsum_batch = Rsum.view(-1, 3, 3)
        P_batch = P.view(-1, 3).unsqueeze(-1)

        weighted_edges = 0.5 * (self.w_nfmt[..., None] *
                                torch.bmm(Rsum_batch, P_batch).squeeze(-1).reshape(self.n_vertices, max_neighbors, 3))
        b = weighted_edges.sum(dim=1)

        return b

    def _build_laplacian_matrix(self):
        """Build the cotangent Laplacian matrix"""
        if self.precomputed_laplacian is not None:
            # Use precomputed Laplacian - should already be set in __init__
            return

        # Get cotangent weights
        w_full = self._get_cotangent_weights_full()

        # Build Laplacian: L_ii = sum(w_i), L_ij = -w_ij
        L = torch.diag(torch.sum(w_full, dim=1)) - w_full

        self._laplacian_matrix = L

    def _solve_positions(self, b, rotations):
        """Solve for new positions given RHS and constraints"""
        # Get unknown vertices (not fixed or handles)
        all_constrained = set(self.handle_indices + self.fixed_indices)
        unknown_indices = [i for i in range(self.n_vertices) if i not in all_constrained]

        if len(unknown_indices) == 0:
            # All vertices are constrained
            return self._current_vertices.clone()

        # Extract reduced system
        L = self._laplacian_matrix
        L_reduced = L[unknown_indices][:, unknown_indices]
        b_reduced = b[unknown_indices]

        # Subtract contribution from known vertices
        for i in self.handle_indices:
            b_reduced -= L[unknown_indices][:, [i]] @ self._current_vertices[[i]]
        for i in self.fixed_indices:
            b_reduced -= L[unknown_indices][:, [i]] @ self._original_vertices[[i]]

        # Solve reduced system
        try:
            # Use Cholesky if possible
            L_chol = torch.linalg.cholesky(L_reduced)
            x_reduced = torch.cholesky_solve(b_reduced, L_chol)
        except:
            # Fallback to least squares
            x_reduced = torch.linalg.lstsq(L_reduced, b_reduced).solution

        # Reconstruct full solution
        new_vertices = self._current_vertices.clone()
        new_vertices[unknown_indices] = x_reduced

        return new_vertices

    def set_handles(
            self,
            handle_indices: Union[torch.Tensor, np.ndarray, List[int]]
    ) -> None:
        """
        Set handle vertices that will be moved during deformation.
        Args:
            handle_indices: Indices of vertices to use as handles
        """
        if isinstance(handle_indices, (torch.Tensor, np.ndarray)):
            handle_indices = handle_indices.tolist()
        self.handle_indices = list(handle_indices)

    def set_fixed(
            self,
            fixed_indices: Union[torch.Tensor, np.ndarray, List[int]]
    ) -> None:
        """
        Set fixed vertices that will remain at their original positions.
        Args:
            fixed_indices: Indices of vertices to keep fixed
        """
        if isinstance(fixed_indices, (torch.Tensor, np.ndarray)):
            fixed_indices = fixed_indices.tolist()
        self.fixed_indices = list(fixed_indices)

    def deform(
            self,
            target_positions: Union[torch.Tensor, np.ndarray],
            max_iterations: int = 10,
            tolerance: float = 1e-6
    ) -> Tuple[torch.Tensor, int]:
        """
        Perform ARAP deformation using alternating optimization.
        Args:
            target_positions: Target positions for handle vertices (len(handle_indices), 3)
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance for energy
        Returns:
            Tuple of (deformed_vertices, num_iterations)
        """
        if len(self.handle_indices) == 0:
            return self._current_vertices.clone(), 0

        # Convert target positions to tensor with consistent dtype
        if isinstance(target_positions, np.ndarray):
            target_positions = torch.from_numpy(target_positions)
        # Ensure same dtype as vertices
        if target_positions.dtype != self._original_vertices.dtype:
            target_positions = target_positions.to(dtype=self._original_vertices.dtype)
        target_positions = target_positions.to(self.device)

        if target_positions.shape[0] != len(self.handle_indices):
            raise ValueError(f"Expected {len(self.handle_indices)} target positions, got {target_positions.shape[0]}")

        # Set handle positions
        self._current_vertices = self._current_vertices.clone()
        for i, handle_idx in enumerate(self.handle_indices):
            self._current_vertices[handle_idx] = target_positions[i]

        # Initialize with naive Laplacian solution if needed
        if self._laplacian_matrix is None:
            self._build_laplacian_matrix()

        prev_energy = float('inf')

        for iteration in range(max_iterations):
            # Step 1: Compute optimal rotations
            rotations = self._compute_rotations(self._original_vertices, self._current_vertices)

            # Step 2: Solve for new positions
            b = self._build_laplacian_system(rotations)
            new_vertices = self._solve_positions(b, rotations)

            # Update handle positions (they should stay fixed to targets)
            for i, handle_idx in enumerate(self.handle_indices):
                new_vertices[handle_idx] = target_positions[i]

            # Update fixed positions
            for fixed_idx in self.fixed_indices:
                new_vertices[fixed_idx] = self._original_vertices[fixed_idx]

            # Check convergence
            energy = torch.norm(new_vertices - self._current_vertices).item()
            if abs(prev_energy - energy) < tolerance:
                self._current_vertices = new_vertices
                return self._current_vertices.clone(), iteration + 1

            self._current_vertices = new_vertices
            prev_energy = energy

        return self._current_vertices.clone(), max_iterations

    @property
    def vertices(self) -> torch.Tensor:
        """Get current deformed vertices."""
        return self._current_vertices.clone()

    @property
    def faces(self) -> torch.Tensor:
        """Get face indices."""
        return self._faces.clone() if self._faces is not None else None

    @property
    def original_vertices(self) -> torch.Tensor:
        """Get original undeformed vertices."""
        return self._original_vertices.clone()