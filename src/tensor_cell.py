"""
TensorCell — Geometry-aware GW strain processing cell with CJPT integration.

Enhancements:
- Causal-bound alignment for dual-field projection
- Integration with CausalEnforcer for deviation tracking
"""

import uuid
import socket
import time
import numpy as np
import sys
sys.path.append('/app/utils')
from quaternion import (
    generate_geometric_projections,
    quaternion_to_matrix,
    random_unit_quaternion,
)

try:
    import vibetensor as vbt
    HAS_VBT = True
except ImportError:
    HAS_VBT = False
    print("Warning: vibetensor not found, using numpy fallback")

class TensorCell:
    def __init__(self, genome: dict, causal_enforcer=None):
        self.genome = genome
        self.fitness = 0.0
        self._fitness_ema_alpha = genome.get("fitness_ema_alpha", 0.1)
        self.uuid = str(uuid.uuid4())[:8]
        self.pod_name = socket.gethostname()
        self.step = 0
        self.causal_enforcer = causal_enforcer
        self.causal_projection_active = False

        # Initialize Geometric System
        self._setup_geometric_operations()

    def _setup_geometric_operations(self):
        """Configure geometry-aware processing based on genome."""
        seed_val = int(self.uuid, 16) % (2**31)

        # 1. Rotation Configuration
        if self.genome.get("apply_rotation", False):
            q = self.genome.get("rotation_quaternion")
            if q is None:
                q = random_unit_quaternion(seed=seed_val)
            self.rotation_matrix = quaternion_to_matrix(q)
        else:
            self.rotation_matrix = np.eye(3)

        # 2. Projection System
        if self.genome.get("use_geometric_projections", False):
            self.projections = generate_geometric_projections(
                n_proj=self.genome.get("num_projections", 3),
                rank=self.genome.get("projection_rank", 2),
                seed=seed_val,
            )
        else:
            self.projections = None
        
        # 3. Causal projection placeholder (computed dynamically)
        self.causal_projection_matrix = None

    def compute_causal_projection(self, delta_kk, J_bound, eta=0.8):
        """
        Compute projection matrix aligned with causal boundary.
        
        This is the key CJPT innovation: the projection operator is derived
        from the causal deviation covariance, not hardcoded geometry.
        
        Parameters
        ----------
        delta_kk : float
            Current causal deviation
        J_bound : float
            Jacobi amplitude bound
        eta : float
            Boundary scaling factor
        """
        delta_c = eta * J_bound
        
        # Activate causal projection when deviation reaches boundary
        if delta_kk >= 0.8 * J_bound:
            self.causal_projection_active = True
            
            # For now, use a simple deviation-weighted projection
            # In full implementation, this would be eigenvectors of C_Delta
            alpha = min(delta_kk / delta_c, 1.5)
            
            # Generate deviation-aligned projection
            # This is a placeholder - full implementation needs covariance matrix
            theta = alpha * np.pi / 4
            c, s = np.cos(theta), np.sin(theta)
            
            self.causal_projection_matrix = np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])
        else:
            self.causal_projection_active = False
            self.causal_projection_matrix = None

    def solve_physics(self, inputs: dict) -> dict:
        """
        Run comb/parity kernel on (possibly enhanced) strain data.

        Parameters
        ----------
        inputs : dict
            Must contain 'strain' and 'comb_mask' (numpy or dlpack capsules)

        Returns
        -------
        dict with score, elapsed_ms, partial_fitness, step, diagnostics.
        """
        start = time.perf_counter()

        # Handle both numpy arrays and DLPack
        if 'strain_dlpack' in inputs:
            if HAS_VBT:
                strain = vbt.from_dlpack(inputs["strain_dlpack"])
                comb = vbt.from_dlpack(inputs["comb_mask_dlpack"])
            else:
                raise ValueError("DLPack input requires vibetensor")
        else:
            strain = inputs.get('strain')
            comb = inputs.get('comb_mask')
            if strain is None or comb is None:
                raise ValueError("Must provide either strain/comb_mask or strain_dlpack/comb_mask_dlpack")

        # Apply geometric transformations
        processed_strain, geo_info = self._apply_geometric_enhancements(strain)

        # Simple score computation (placeholder for actual physics kernel)
        # Handle shape mismatch: processed_strain may be (B,3,T), comb is (B,T)
        if processed_strain.ndim == 3 and comb.ndim == 2:
            # Take first channel or flatten
            if HAS_VBT:
                score = vbt.sum(processed_strain[:, 0, :] * comb)
            else:
                score = np.sum(processed_strain[:, 0, :] * comb)
        else:
            if HAS_VBT:
                score = vbt.sum(processed_strain * comb)
            else:
                score = np.sum(processed_strain * comb)

        elapsed_ms = (time.perf_counter() - start) * 1000
        speed_reward = 1000.0 / max(elapsed_ms, 1.0)

        # EMA fitness
        self.fitness = (
            self._fitness_ema_alpha * speed_reward
            + (1.0 - self._fitness_ema_alpha) * self.fitness
        )
        self.step += 1

        return {
            "score": float(score.item() if hasattr(score, "item") else score),
            "elapsed_ms": elapsed_ms,
            "partial_fitness": self.fitness,
            "step": self.step,
            "geo_info": geo_info,
            "causal_projection_active": self.causal_projection_active
        }

    def _apply_geometric_enhancements(self, tensor, causal_projection=None):
        """
        Apply geometry-aware processing to GW strain.
        Now with CJPT causal-bound alignment.

        Accepts:
          (Batch, Time)       — single-detector raw strain
          (Batch, Time, 1)    — single-detector with trailing dim
          (Batch, 3, Time)    — multi-detector (H1, L1, V1)

        Returns:
          (tensor, info_dict) — processed tensor + diagnostic metadata
        """
        info = {"embedding": "none", "operator": "none", "projection_idx": None}

        # Convert to numpy if needed
        is_vbt = HAS_VBT and hasattr(tensor, '__module__') and 'vibetensor' in tensor.__module__
        if is_vbt:
            tensor_np = tensor.numpy() if hasattr(tensor, 'numpy') else np.array(tensor)
        else:
            tensor_np = np.array(tensor)

        # 1. Hilbert Embedding for 1D / single-detector data
        if tensor_np.ndim == 2 or (tensor_np.ndim == 3 and tensor_np.shape[1] > 3):
            # Robust squeeze for (Batch, Time, 1)
            if tensor_np.ndim == 3 and tensor_np.shape[-1] == 1:
                tensor_np = tensor_np.squeeze(-1)
            elif tensor_np.ndim == 3:
                b, c, t = tensor_np.shape
                tensor_np = tensor_np.reshape(b, c * t)

            # Analytic signal via Hilbert transform
            from scipy.signal import hilbert
            analytic = hilbert(tensor_np, axis=-1)

            real_part = analytic.real[:, np.newaxis, :]
            imag_part = analytic.imag[:, np.newaxis, :]

            # Instantaneous frequency
            phase = np.angle(analytic)
            inst_freq = np.diff(phase, axis=-1)
            inst_freq = np.concatenate([inst_freq, inst_freq[:, -1:]], axis=-1)
            inst_freq = inst_freq[:, np.newaxis, :]

            tensor_3d = np.concatenate([real_part, imag_part, inst_freq], axis=1)
            info["embedding"] = "hilbert_3ch"

            # Recursively apply rotation/projection
            tensor_out, geo_info = self._apply_geometric_enhancements(tensor_3d, causal_projection)
            geo_info["embedding"] = info["embedding"]
            return tensor_out, geo_info

        # 2. Geometric operations on (Batch, 3, Time) data
        if tensor_np.ndim == 3 and tensor_np.shape[1] == 3:
            op_matrix = None

            # CJPT Enhancement: Use causal projection if active
            if causal_projection is not None:
                op_matrix = causal_projection
                info["operator"] = "causal_projection"
            elif self.causal_projection_active and self.causal_projection_matrix is not None:
                op_matrix = self.causal_projection_matrix
                info["operator"] = "causal_projection_auto"
            elif self.projections:
                idx = self.genome.get(
                    "active_projection_idx", self.step
                ) % len(self.projections)
                op_matrix = self.projections[idx]
                info["operator"] = "projection"
                info["projection_idx"] = int(idx)
            elif self.genome.get("apply_rotation", False):
                op_matrix = self.rotation_matrix
                info["operator"] = "rotation"

            if op_matrix is not None:
                # Channel mixing: rotate/project dimension 1
                # (B, 3, T) -> (B, T, 3) @ (3, 3)^T -> (B, T, 3) -> (B, 3, T)
                tensor_np = np.einsum('btc,dc->btd', 
                                      tensor_np.transpose(0, 2, 1), 
                                      op_matrix).transpose(0, 2, 1)

        # Convert back to original type
        if is_vbt and HAS_VBT:
            tensor = vbt.tensor(tensor_np)
        else:
            tensor = tensor_np

        return tensor, info

    def get_geometric_params(self) -> dict:
        """Export current geometric configuration."""
        params = {
            "rotation_matrix": self.rotation_matrix.tolist(),
            "has_projections": self.projections is not None,
            "causal_projection_active": self.causal_projection_active,
        }
        if self.projections:
            params["projections"] = [p.tolist() for p in self.projections]
            params["active_projection_idx"] = self.genome.get(
                "active_projection_idx", 0
            )
        if self.causal_projection_matrix is not None:
            params["causal_projection_matrix"] = self.causal_projection_matrix.tolist()
        return params

    def update_from_drifting_output(self, generated_params: dict):
        """Accept parameters from Drifting Model."""
        if "rotation_quaternion" in generated_params:
            q = np.asarray(generated_params["rotation_quaternion"], dtype=np.float64)
            self.rotation_matrix = quaternion_to_matrix(q)
            self.genome["rotation_quaternion"] = q.tolist()

        if "projection_rank" in generated_params:
            self.genome["projection_rank"] = int(generated_params["projection_rank"])
            self._setup_geometric_operations()

    @staticmethod
    def _ensure_vbt_tensor(array, reference_tensor):
        """Coerce numpy array to vibetensor."""
        if not HAS_VBT:
            return array
        if isinstance(array, np.ndarray):
            return vbt.tensor(
                array,
                device=reference_tensor.device if hasattr(reference_tensor, 'device') else 'cpu',
                dtype=reference_tensor.dtype if hasattr(reference_tensor, 'dtype') else np.float64,
            )
        return array

    def __repr__(self):
        proj_info = f", projections={len(self.projections)}" if self.projections else ""
        return (
            f"TensorCell(uuid={self.uuid}, fitness={self.fitness:.4f}, "
            f"step={self.step}{proj_info}, causal_active={self.causal_projection_active})"
        )
