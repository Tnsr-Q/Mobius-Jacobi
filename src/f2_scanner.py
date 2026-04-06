"""
f2 Scanner - Scans f2 parameter space and plots causal deviation contours.

Implements step 4 of the execution protocol:
- Run f2 scan from [5×10⁻⁹, 2×10⁻⁸]
- Plot Δ_KK(ω) contours
- Extract emergent bound shape B(ω; f2)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List, Tuple, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class F2Scanner:
    """Scans f2 parameter space for causal bound emergence."""
    
    def __init__(self, cjpt_system, causal_enforcer, output_dir="/app/outputs"):
        self.cjpt = cjpt_system
        self.enforcer = causal_enforcer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.scan_results = []
    
    def scan_f2_range(self, f2_min: float = 5e-9, f2_max: float = 2e-8, 
                     n_points: int = 20, omega_range: Tuple[float, float] = (1e-4, 1e2),
                     n_omega: int = 1000) -> Dict:
        """
        Scan f2 parameter space and compute causal deviation.
        
        Parameters
        ----------
        f2_min, f2_max : float
            f2 scan range
        n_points : int
            Number of f2 sample points
        omega_range : tuple
            Angular frequency range (rad/s)
        n_omega : int
            Frequency resolution
            
        Returns
        -------
        dict
            Scan results with f2 values, omega grid, delta_kk grid
        """
        f2_values = np.logspace(np.log10(f2_min), np.log10(f2_max), n_points)
        omega = np.logspace(np.log10(omega_range[0]), np.log10(omega_range[1]), n_omega)
        
        # Storage for causal deviation at each (f2, omega)
        delta_kk_grid = np.zeros((n_points, n_omega))
        J_bound_values = np.zeros(n_points)
        M2_values = np.zeros(n_points)
        
        logger.info(f"Starting f2 scan: {n_points} points from {f2_min:.2e} to {f2_max:.2e}")
        
        for i, f2 in enumerate(f2_values):
            # Compute derived parameters
            M2 = self.cjpt.compute_M2(f2)
            H = 1e13  # Typical inflation Hubble scale in GeV
            J_bound = self.cjpt.compute_J_bound(f2, H)
            
            M2_values[i] = M2
            J_bound_values[i] = J_bound
            
            # Generate synthetic spectral response for this f2
            # In real implementation, this comes from solving Jacobi equation
            R_s = self._synthetic_spectral_response(omega, f2, H, M2)
            
            # Compute causal deviation at each frequency
            for j, om in enumerate(omega):
                # Sample window around this frequency
                idx_window = max(0, j-50), min(n_omega, j+50)
                omega_window = omega[idx_window[0]:idx_window[1]]
                R_s_window = R_s[idx_window[0]:idx_window[1]]
                
                # Compute deviation
                if len(omega_window) > 10:
                    delta_kk = self.enforcer.compute_causal_deviation(omega_window, R_s_window)
                    delta_kk_grid[i, j] = delta_kk
                else:
                    delta_kk_grid[i, j] = 0.0
            
            logger.info(f"  f2={f2:.2e} | M2={M2:.2e} GeV | J_bound={J_bound:.2e} | max(Δ_KK)={np.max(delta_kk_grid[i]):.3e}")
        
        results = {
            'f2_values': f2_values,
            'omega': omega,
            'delta_kk_grid': delta_kk_grid,
            'J_bound_values': J_bound_values,
            'M2_values': M2_values,
        }
        
        self.scan_results = results
        return results
    
    def _synthetic_spectral_response(self, omega: np.ndarray, f2: float, 
                                    H: float, M2: float) -> np.ndarray:
        """
        Generate synthetic spectral response with fakeon poles.
        
        In real implementation, this comes from solving the Jacobi deviation equation.
        Here we model the key features:
        - Minimum-phase baseline
        - Fakeon pole contribution at omega ~ Gamma_fakeon
        - Gaussian envelope artifact
        """
        # Fakeon correlation length
        xi_H = self.cjpt.xi_H
        Gamma_fakeon = np.sqrt(xi_H * H**2 / self.cjpt.M_Pl**2)
        
        # Baseline minimum-phase response (causal)
        R_baseline = 1.0 / (1.0 + (omega / H)**2)
        
        # Fakeon pole contribution (non-minimum-phase)
        # Creates deviation from causality
        fakeon_contrib = (Gamma_fakeon**2) / ((omega**2 + Gamma_fakeon**2) + 1e-10)
        
        # Gaussian envelope (the "artifact" that's actually the Green's function)
        sigma_env = 1.0 / Gamma_fakeon * np.sqrt(self.cjpt.M_Pl**2 / (xi_H * H**2))
        envelope = np.exp(-(omega - H)**2 / (2 * sigma_env**2))
        
        # Combined response
        magnitude = R_baseline + 0.1 * f2 * fakeon_contrib * envelope
        
        # Add non-minimum-phase component
        phase_offset = 0.5 * f2 * envelope * np.sin(omega / H)
        
        R_s = magnitude * np.exp(1j * phase_offset)
        
        return R_s
    
    def plot_contours_2d(self, results: Dict = None, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot 2D contour map of Δ_KK(f2, ω).
        
        Creates a heatmap showing causal deviation across f2 and frequency space.
        The emergent bound shape B(ω; f2) appears as contour lines.
        """
        if results is None:
            results = self.scan_results
        
        if not results:
            logger.warning("No scan results available. Run scan_f2_range() first.")
            return
        
        f2_values = results['f2_values']
        omega = results['omega']
        delta_kk_grid = results['delta_kk_grid']
        J_bound_values = results['J_bound_values']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create meshgrid for contour plot
        F2, Omega = np.meshgrid(f2_values, omega, indexing='ij')
        
        # Contour plot
        levels = np.logspace(-4, -1, 20)
        contour = ax.contourf(F2, Omega, delta_kk_grid, levels=levels, 
                             cmap='viridis', norm=plt.matplotlib.colors.LogNorm())
        
        # Overlay J_bound contours (the "critical" boundary)
        for i, (f2, J_bound) in enumerate(zip(f2_values[::3], J_bound_values[::3])):
            # Find omega where delta_kk ≈ 0.8 * J_bound
            delta_c = 0.8 * J_bound
            if np.any(delta_kk_grid[i*3] > delta_c):
                ax.axvline(f2, color='red', alpha=0.3, linestyle='--', linewidth=0.5)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('f₂ (dimensionless)', fontsize=14)
        ax.set_ylabel('ω (rad/s)', fontsize=14)
        ax.set_title('Causal Deviation Δ_KK(f₂, ω) - 2D Contour Map', fontsize=16)
        
        cbar = plt.colorbar(contour, ax=ax, label='Δ_KK')
        cbar.set_label('Causal Deviation Δ_KK', fontsize=12)
        
        plt.tight_layout()
        output_path = self.output_dir / 'cjpt_contour_2d.png'
        plt.savefig(output_path, dpi=150)
        logger.info(f"Saved 2D contour plot to {output_path}")
        plt.close()
        
        return fig
    
    def plot_contours_3d(self, results: Dict = None, figsize: Tuple[int, int] = (14, 10)):
        """
        Plot 3D surface of Δ_KK(f2, ω).
        
        Shows the causal boundary as a 3D surface, making the emergence of the
        bound shape visually striking.
        """
        if results is None:
            results = self.scan_results
        
        if not results:
            logger.warning("No scan results available. Run scan_f2_range() first.")
            return
        
        f2_values = results['f2_values']
        omega = results['omega']
        delta_kk_grid = results['delta_kk_grid']
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid
        F2, Omega = np.meshgrid(f2_values, omega, indexing='ij')
        
        # Surface plot
        surf = ax.plot_surface(np.log10(F2), np.log10(Omega), np.log10(delta_kk_grid + 1e-10),
                              cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.8)
        
        ax.set_xlabel('log₁₀(f₂)', fontsize=12)
        ax.set_ylabel('log₁₀(ω)', fontsize=12)
        ax.set_zlabel('log₁₀(Δ_KK)', fontsize=12)
        ax.set_title('3D Causal Deviation Surface', fontsize=16)
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='log₁₀(Δ_KK)')
        
        plt.tight_layout()
        output_path = self.output_dir / 'cjpt_contour_3d.png'
        plt.savefig(output_path, dpi=150)
        logger.info(f"Saved 3D surface plot to {output_path}")
        plt.close()
        
        return fig
    
    def extract_bound_shape(self, f2_target: float = 1e-8, eta: float = 0.8) -> Dict:
        """
        Extract the bound shape B(ω; f2) at a specific f2 value.
        
        The bound shape is defined by the level set:
        B = {ω | Δ_KK(ω) = η * J_bound(f2)}
        
        Parameters
        ----------
        f2_target : float
            Target f2 value
        eta : float
            Anselmi-Piva scaling factor
            
        Returns
        -------
        dict
            Bound shape data: omega values where boundary is crossed
        """
        results = self.scan_results
        if not results:
            logger.warning("No scan results available.")
            return {}
        
        # Find closest f2 value
        idx_f2 = np.argmin(np.abs(results['f2_values'] - f2_target))
        f2_actual = results['f2_values'][idx_f2]
        
        omega = results['omega']
        delta_kk = results['delta_kk_grid'][idx_f2]
        J_bound = results['J_bound_values'][idx_f2]
        
        # Find boundary crossings
        delta_c = eta * J_bound
        crossing_mask = np.abs(delta_kk - delta_c) < 0.1 * delta_c
        
        omega_boundary = omega[crossing_mask]
        
        logger.info(f"Bound shape at f2={f2_actual:.2e}: {len(omega_boundary)} boundary points")
        if len(omega_boundary) > 0:
            logger.info(f"  Ω_boundary = [{omega_boundary.min():.2e}, {omega_boundary.max():.2e}] rad/s")
        else:
            logger.info(f"  No boundary crossings found at current resolution")
        
        return {
            'f2': f2_actual,
            'omega_boundary': omega_boundary,
            'J_bound': J_bound,
            'delta_c': delta_c,
        }
    
    def verify_dual_field_emergence(self, g_trap_range: Tuple[float, float] = (1.0, 1.5),
                                   alpha_c_target: float = 3.18e-4,
                                   gamma_target: float = 8.1) -> Dict:
        """
        Verify dual-field emergence conditions (step 5 of protocol).
        
        Checks:
        1. G_trap in [1.0, 1.5]
        2. σ_env ≈ σ_crit
        3. α_c locks to 3.18×10⁻⁴
        4. Γ → 8.1 without tuning
        
        Returns
        -------
        dict
            Verification results
        """
        # This is a placeholder - full implementation requires running the PPO loop
        logger.info("Verifying dual-field emergence conditions...")
        logger.info(f"  Target G_trap range: {g_trap_range}")
        logger.info(f"  Target α_c: {alpha_c_target:.3e}")
        logger.info(f"  Target Γ: {gamma_target:.2f}")
        
        # In full implementation:
        # 1. Run PPO with f2 in convergence range
        # 2. Monitor g_trap, sigma_env evolution
        # 3. Check if alpha_c and gamma converge to targets
        
        return {
            'g_trap_in_range': None,  # To be computed
            'sigma_env_critical': None,
            'alpha_c_locked': None,
            'gamma_converged': None,
        }
