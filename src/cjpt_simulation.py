"""
Main CJPT Simulation Runner

Executes the full CJPT protocol:
1. Initialize system with f2 parameter
2. Run causal deviation tracking
3. Perform f2 scan
4. Generate visualizations
5. Verify dual-field emergence
"""

import sys
sys.path.append('/app/utils')
sys.path.append('/app/src')

import numpy as np
import logging
from pathlib import Path

from causal_enforcer import CausalEnforcer, compute_causal_covariance
from tensor_cell import TensorCell
from cjpt_system import CJPTSystem
from f2_scanner import F2Scanner
from visualizer import CJPTVisualizer
from jacobi_ode_solver import JacobiODESolver

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CJPTSimulation:
    """Main simulation orchestrator for CJPT framework."""
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        
        # Initialize components
        logger.info("Initializing CJPT system components...")
        self.cjpt = CJPTSystem(self.config.get('cjpt'))
        self.enforcer = CausalEnforcer(
            kk_tolerance=self.config.get('kk_tolerance', 0.8),
            enforce_kk=True
        )
        
        # TensorCell genome
        genome = {
            'apply_rotation': True,
            'use_geometric_projections': True,
            'num_projections': 5,
            'projection_rank': 2,
            'fitness_ema_alpha': 0.1,
        }
        self.tensor_cell = TensorCell(genome, causal_enforcer=self.enforcer)
        
        # Scanner and visualizer
        self.scanner = F2Scanner(self.cjpt, self.enforcer, 
                                output_dir=self.config.get('output_dir', '/app/outputs'))
        self.visualizer = CJPTVisualizer(output_dir=self.config.get('output_dir', '/app/outputs'))

        # Jacobi ODE solver (replaces synthetic strain data)
        cjpt_cfg = self.config.get('cjpt', {})
        sim_cfg = self.config.get('simulation', {})
        self.ode_solver = JacobiODESolver(
            f2=1e-8,
            xi_H=cjpt_cfg.get('xi_H', 5e8),
            H0=sim_cfg.get('H', 1e13),
            M_Pl=cjpt_cfg.get('M_Pl', 2.435e18),
        )
        
        # State tracking
        self.state_history = []
        
        logger.info("CJPT system initialized successfully.")
    
    @staticmethod
    def _default_config():
        return {
            'cjpt': {
                'M_Pl': 2.435e18,  # GeV
                'xi_H': 5e8,
                'eta': 0.8,
                'kappa_mc': 50,
            },
            'kk_tolerance': 0.8,
            'output_dir': '/app/outputs',
            'f2_scan': {
                'f2_min': 5e-9,
                'f2_max': 2e-8,
                'n_points': 25,
                'n_omega': 1000,
            },
            'simulation': {
                'n_rollout_steps': 100,
                'H': 1e13,  # Hubble in GeV
            }
        }
    
    def run_single_rollout(self, f2: float, n_steps: int = 100):
        """
        Run a single PPO-style rollout with CJPT tracking.
        
        Parameters
        ----------
        f2 : float
            Quadratic gravity parameter
        n_steps : int
            Number of rollout steps
        """
        logger.info(f"\nStarting rollout with f2={f2:.3e}")
        logger.info("="*60)
        
        H = self.config['simulation']['H']
        M2 = self.cjpt.compute_M2(f2)
        J_bound = self.cjpt.compute_J_bound(f2, H)
        
        logger.info(f"Derived parameters:")
        logger.info(f"  M2 = {M2:.3e} GeV")
        logger.info(f"  J_bound = {J_bound:.3e}")
        logger.info(f"  H = {H:.3e} GeV")

        # Reinitialise ODE solver for this f2 value
        cjpt_cfg = self.config.get('cjpt', {})
        ode = JacobiODESolver(
            f2=f2,
            xi_H=cjpt_cfg.get('xi_H', 5e8),
            H0=H,
            M_Pl=cjpt_cfg.get('M_Pl', 2.435e18),
        )

        # USR background: mild Hubble modulation + slow-roll decay
        def background(t):
            H_t = H * (1.0 + 0.01 * np.sin(t * 0.05))
            eps = 0.01 * np.exp(-t / 500.0)
            phi = 1e17
            return H_t, eps, phi

        # Solve Jacobi ODE once and derive spectral response.
        # Integration duration: 0.05 s/sample * 100 samples/step * n_steps → 5*n_steps s
        logger.info("  Solving Jacobi ODE...")
        t_duration = 5.0 * n_steps  # conformal time units; dt default 0.05
        t_arr, J_arr = ode.solve((0, t_duration), [1e-8, 1e-8, 0.0, 0.0], background)
        omega_ode, R_s_ode, sigma_env_base, strain_3d = ode.spectral_response(t_arr, J_arr)
        logger.info(f"  ODE solved: {len(t_arr)} samples, σ_env={sigma_env_base:.3e}")

        # Causal covariance projection (replaces simplified rotation)
        delta_kk_arr = np.abs(np.angle(R_s_ode[:, 0])) if R_s_ode.ndim > 1 else np.abs(np.angle(R_s_ode))
        P_causal, eigs = compute_causal_covariance(omega_ode, R_s_ode, delta_kk_arr, J_bound)
        logger.info(f"  Causal projection eigenvalues: {eigs}")

        # Field initial conditions
        phi = np.array([1e17, 0.5e17])  # [phi_H, phi_S]
        phi_dot = np.array([1e16, 0.2e16])
        eps = 0.01
        
        for step in range(n_steps):
            # Use ODE-derived spectral response for causal deviation
            # (use a sliding window over the pre-computed array)
            idx = min(step, len(omega_ode) - 1)
            omega_step = omega_ode[:idx + 1] if idx > 0 else omega_ode[:2]
            R_s_step = R_s_ode[:idx + 1] if idx > 0 else R_s_ode[:2]

            # Compute causal deviation from ODE output
            delta_kk = self.enforcer.compute_causal_deviation(omega_step, R_s_step[:, 0])

            # Update adaptive tolerance
            new_tolerance = self.enforcer.adaptive_tolerance(delta_kk, J_bound, self.cjpt.eta)
            self.enforcer.kk_tolerance = new_tolerance

            # Envelope width from ODE solver (perturbed slightly each step)
            sigma_env = sigma_env_base * (1 + 0.01 * np.random.randn())

            # Process through TensorCell with ODE-derived strain
            strain = strain_3d[:1000].T if strain_3d.shape[0] >= 1000 else strain_3d.T
            comb_mask = np.ones((strain.shape[0], strain.shape[1] if strain.ndim > 1 else 1))
            result = self.tensor_cell.solve_physics({
                'strain': strain[:, :, 0] if strain.ndim == 3 else strain,
                'comb_mask': comb_mask[:, 0] if comb_mask.ndim > 1 else comb_mask
            })

            # Update causal projection using eigendecomposed C_Δ
            self.tensor_cell.compute_causal_projection(delta_kk, J_bound, self.cjpt.eta)

            # Transport and symplectic matrices
            H_t, eps_t, _ = background(step)
            M_matrix = ode._transport_matrix(step, H_t, eps_t, 0.0)
            Omega = self._synthetic_symplectic_matrix()
            
            # 1. Geometric score → Phase logic & logging
            g_trap_geom = self.cjpt.geometric_trap_score(H, M2, M_matrix, Omega)
            phase = self.cjpt.cjpt_phase_check(g_trap_geometric=g_trap_geom, delta_kk=delta_kk, J_bound=J_bound,
                                               sigma_env=sigma_env, xi_H=self.cjpt.xi_H, H=H)

            # 2. Smooth score → Gradient flow & reward
            s_trap = self.cjpt.trap_door_detector(H, M2, M_matrix, Omega)
            gammas = np.array([8.0 + 0.1 * np.random.randn()])
            reward = self.cjpt.cjpt_reward(gammas, s_trap, delta_kk, J_bound, sigma_env)
            
            # Build PPO state
            M_eigs = np.linalg.eigvalsh(M_matrix)
            state = self.cjpt.build_ppo_state(phi, phi_dot, H, eps, M_eigs, Omega,
                                             delta_kk, sigma_env, J_bound)
            self.state_history.append(state)
            
            # Log diagnostics
            if step % 10 == 0:
                self.cjpt.log_diagnostics(step, phase, g_trap_geom, delta_kk, sigma_env, J_bound)
            
            # Simple field evolution
            phi += phi_dot * 0.1
            phi_dot *= 0.99
        
        logger.info("="*60)
        logger.info("Rollout complete.\n")
    
    def run_f2_scan(self):
        """
        Execute f2 parameter space scan (Protocol Step 4).
        """
        logger.info("\n" + "="*60)
        logger.info("Starting f2 scan...")
        logger.info("="*60)
        
        scan_config = self.config['f2_scan']
        results = self.scanner.scan_f2_range(
            f2_min=scan_config['f2_min'],
            f2_max=scan_config['f2_max'],
            n_points=scan_config['n_points'],
            n_omega=scan_config['n_omega']
        )
        
        logger.info("f2 scan complete. Generating visualizations...")
        
        # Generate 2D and 3D plots
        self.scanner.plot_contours_2d(results)
        self.scanner.plot_contours_3d(results)
        
        # Extract bound shape at convergence point
        bound_shape = self.scanner.extract_bound_shape(f2_target=1e-8)
        
        logger.info("="*60)
        return results, bound_shape
    
    def generate_visualizations(self):
        """
        Generate all diagnostic visualizations.
        """
        logger.info("\nGenerating diagnostic visualizations...")
        
        self.visualizer.plot_phase_diagram(self.cjpt)
        self.visualizer.plot_diagnostic_dashboard(self.cjpt, self.tensor_cell, self.enforcer)
        
        if self.state_history:
            self.visualizer.plot_ppo_state_trajectory(self.state_history)
        
        logger.info("Visualizations complete.")
    
    def verify_dual_field_emergence(self):
        """
        Verify dual-field emergence conditions (Protocol Step 5).
        """
        logger.info("\n" + "="*60)
        logger.info("Verifying dual-field emergence...")
        logger.info("="*60)
        
        verification = self.scanner.verify_dual_field_emergence()
        
        # Check phase history for DUAL_EMERGENCE
        dual_emergence_count = self.cjpt.phase_history.count('DUAL_EMERGENCE')
        bound_recon_count = self.cjpt.phase_history.count('BOUND_RECONSTRUCTION')
        
        logger.info(f"Phase statistics:")
        logger.info(f"  DUAL_EMERGENCE: {dual_emergence_count} steps")
        logger.info(f"  BOUND_RECONSTRUCTION: {bound_recon_count} steps")
        
        # Check trap door range
        if self.cjpt.g_trap_history:
            g_trap_array = np.array(self.cjpt.g_trap_history)
            in_range = np.sum((g_trap_array >= 1.0) & (g_trap_array <= 1.5))
            logger.info(f"  G_trap in [1.0, 1.5]: {in_range}/{len(g_trap_array)} steps")
        
        logger.info("="*60)
        return verification
    
    # Synthetic data generators (placeholders for real physics)
    
    def _synthetic_strain_data(self, batch_size=4, time_steps=1000):
        """Generate synthetic GW strain data."""
        t = np.linspace(0, 1, time_steps)
        strain = np.zeros((batch_size, time_steps))
        for i in range(batch_size):
            freq = 100 + 50 * i
            strain[i] = np.sin(2 * np.pi * freq * t) * np.exp(-t)
        return strain
    
    def _synthetic_spectral_response(self, omega, f2, H, M2):
        """Generate synthetic spectral response with fakeon features."""
        Gamma_fakeon = np.sqrt(self.cjpt.xi_H * H**2 / self.cjpt.M_Pl**2)
        
        R_baseline = 1.0 / (1.0 + (omega / H)**2)
        fakeon_contrib = (Gamma_fakeon**2) / ((omega**2 + Gamma_fakeon**2) + 1e-10)
        
        sigma_env = 1.0 / Gamma_fakeon * np.sqrt(self.cjpt.M_Pl**2 / (self.cjpt.xi_H * H**2))
        envelope = np.exp(-(omega - H)**2 / (2 * sigma_env**2))
        
        magnitude = R_baseline + 0.1 * f2 * fakeon_contrib * envelope
        phase_offset = 0.5 * f2 * envelope * np.sin(omega / H)
        
        return magnitude * np.exp(1j * phase_offset)
    
    def _synthetic_transport_matrix(self, H):
        """Generate synthetic transport matrix."""
        # 2x2 for two-field system
        M = np.array([
            [10 * H**2, 2 * H**2],
            [2 * H**2, 8 * H**2]
        ])
        return M
    
    def _synthetic_symplectic_matrix(self):
        """Generate synthetic symplectic matrix."""
        # 4x4 for two fields (position + momentum)
        theta = 0.1
        c, s = np.cos(theta), np.sin(theta)
        Omega = np.array([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, c, -s],
            [0, 0, s, c]
        ])
        return Omega


def main():
    """Main execution function."""
    logger.info("\n" + "#"*60)
    logger.info("#" + " "*58 + "#")
    logger.info("#" + "  CJPT Framework - Causal-Jacobi Phase Transition  ".center(58) + "#")
    logger.info("#" + " "*58 + "#")
    logger.info("#"*60 + "\n")
    
    # Initialize simulation
    sim = CJPTSimulation()
    
    # Step 1-3: Run single rollout at convergence f2
    logger.info("STEP 1-3: Running single rollout with f2=1e-8")
    sim.run_single_rollout(f2=1e-8, n_steps=50)
    
    # Step 4: f2 parameter scan
    logger.info("\nSTEP 4: f2 parameter space scan")
    scan_results, bound_shape = sim.run_f2_scan()
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    sim.generate_visualizations()
    
    # Step 5: Verify dual-field emergence
    logger.info("\nSTEP 5: Dual-field emergence verification")
    verification = sim.verify_dual_field_emergence()
    
    # Summary
    logger.info("\n" + "#"*60)
    logger.info("SIMULATION COMPLETE")
    logger.info("#"*60)
    logger.info(f"\nOutputs saved to: {sim.config['output_dir']}")
    logger.info("\nGenerated files:")
    output_dir = Path(sim.config['output_dir'])
    for f in sorted(output_dir.glob('*.png')):
        logger.info(f"  - {f.name}")
    
    logger.info("\nCJPT Framework execution complete.\n")


if __name__ == "__main__":
    main()
