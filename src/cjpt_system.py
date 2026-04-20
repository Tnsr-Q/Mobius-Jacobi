"""
CJPT System - Causal-Jacobi Phase Transition Framework

Implements:
1. Phase transition detection
2. f2 parameter scanning
3. Dual-field emergence verification
4. PPO state vector construction
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CJPTSystem:
    """Main CJPT orchestrator."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self._ensure_config_complete()
        self.M_Pl = self.config['M_Pl']  # Planck mass in GeV
        self.xi_H = self.config['xi_H']  # Higgs non-minimal coupling
        self.eta = self.config['eta']     # Anselmi-Piva scaling
        self.kappa_mc = self.config['kappa_mc']  # Microcausality margin
        
        # Phase history
        self.phase_history = []
        self.delta_kk_history = []
        self.sigma_env_history = []
        self.g_trap_history = []
    
    @staticmethod
    def _default_config() -> Dict:
        return {
            'M_Pl': 2.435e18,  # GeV
            'xi_H': 5e8,
            'eta': 0.8,
            'kappa_mc': 50,
            'epsilon_H': 0.01,
            'Lambda_crit_factor': 100,
            'delta_symp': 1e-6,
        }
    
    def _ensure_config_complete(self):
        """Ensure all required config keys exist."""
        defaults = self._default_config()
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def compute_M2(self, f2: float) -> float:
        """Fakeon mass from f2 parameter."""
        return self.M_Pl / np.sqrt(f2)
    
    def compute_J_bound(self, f2: float, H: float) -> float:
        """Jacobi amplitude bound from microcausality."""
        return self.kappa_mc * np.sqrt(f2) * (H / self.M_Pl)
    
    def cjpt_phase_check(self, g_trap_geometric: float, delta_kk: float,
                         J_bound: float, sigma_env: float,
                         xi_H: float, H: float) -> str:
        """
        Determine CJPT phase using GEOMETRIC crossing scalar.
        DO NOT pass smooth trap_door_detector output here.

        Parameters
        ----------
        g_trap_geometric : float
            Geometric trap door score from geometric_trap_score() in [0, ∞).
            Thresholds 1.0 and 1.5 are defined for this metric (Rule 2).
            Must NOT be the smooth trap_door_detector() score (range [0, 3]).
        delta_kk : float
            Causal deviation
        J_bound : float
            Jacobi amplitude bound
        sigma_env : float
            Gaussian envelope width
        xi_H : float
            Higgs coupling
        H : float
            Hubble parameter

        Returns
        -------
        str
            Phase label
        """
        sigma_crit = self._compute_sigma_crit(g_trap_geometric, xi_H, H)
        order_param = sigma_env / max(sigma_crit, 1e-12)

        # Phase boundaries use G_trap (geometric) thresholds: 1.0 and 1.5
        if g_trap_geometric < 1.0 and delta_kk < 0.8 * J_bound:
            phase = "MINIMAL_PHASE"
        elif 1.0 <= g_trap_geometric <= 1.5 and 0.8*J_bound <= delta_kk <= 1.2*J_bound:
            phase = "BOUND_RECONSTRUCTION"
        elif g_trap_geometric > 1.5 and delta_kk > 1.2 * J_bound:
            phase = "DUAL_EMERGENCE"
        else:
            phase = "TRANSITION"

        logger.info(f"Phase: {phase} | G_trap={g_trap_geometric:.3f} | Δ_KK={delta_kk:.3e} | σ_param={order_param:.3e}")

        self.phase_history.append(phase)
        return phase
    
    def _compute_sigma_crit(self, g_trap_geometric: float, xi_H: float, H: float) -> float:
        """Critical Gaussian envelope width."""
        return (1.0 / max(g_trap_geometric, 1e-6)) * np.sqrt(self.M_Pl**2 / (xi_H * H**2))
    
    def trap_door_detector(self, H: float, M2: float, M_matrix: np.ndarray,
                          Omega: np.ndarray, kappa: Tuple[float, float, float] = (50, 30, 40)) -> float:
        """
        Smooth (differentiable) trap door score for gradient flow and rewards.

        Returns the sum of three sigmoid veto terms in [0, 3].
        Use this score for PPO rewards and gradient-based updates only.
        DO NOT pass this score to cjpt_phase_check — use geometric_trap_score instead.

        Parameters
        ----------
        H : float
            Hubble parameter
        M2 : float
            Fakeon mass
        M_matrix : np.ndarray
            Transport matrix
        Omega : np.ndarray
            Symplectic matrix
        kappa : tuple
            Sensitivity parameters

        Returns
        -------
        float
            Smooth trap door score in [0, 3]
        """
        # 1. Ghost unitarity bound
        ghost_ratio = H / M2
        ghost_veto = 1.0 / (1.0 + np.exp(-kappa[0] * (ghost_ratio - self.config['epsilon_H'])))
        
        # 2. Tachyonic instability monitor
        lambda_min = np.min(np.linalg.eigvalsh(M_matrix))
        Lambda_crit = self.config['Lambda_crit_factor'] * H**2
        tachyon_veto = 1.0 / (1.0 + np.exp(-kappa[1] * (-lambda_min - Lambda_crit)))
        
        # 3. Symplectic integrity check
        symp_drift = np.abs(np.log(np.linalg.det(Omega)))
        symplectic_veto = 1.0 / (1.0 + np.exp(-kappa[2] * (symp_drift - self.config['delta_symp'])))
        
        total = ghost_veto + tachyon_veto + symplectic_veto
        return total
    
    def geometric_trap_score(self, H: float, M2: float, M_matrix: np.ndarray,
                            Omega: np.ndarray) -> float:
        """
        Geometric crossing scalar (interpretability metric).

        Returns max ratio of actual to threshold values.
        Range: [0, ∞). Use for phase logic and thresholds (Rule 2).
        DO NOT use this score for gradient flow or rewards — use trap_door_detector instead.
        """
        ratio_ghost = (H / M2) / self.config['epsilon_H']
        lambda_min = np.min(np.linalg.eigvalsh(M_matrix))
        Lambda_crit = self.config['Lambda_crit_factor'] * H**2
        ratio_tachyon = (-lambda_min) / Lambda_crit if lambda_min < 0 else 0.0

        symp_drift = np.abs(np.log(np.linalg.det(Omega)))
        ratio_symp = symp_drift / self.config['delta_symp']

        g_trap_geometric = max(ratio_ghost, ratio_tachyon, ratio_symp)
        self.g_trap_history.append(g_trap_geometric)
        return g_trap_geometric
    
    def cjpt_reward(self, gammas: np.ndarray, trap_score: float,
                   delta_kk: float, J_bound: float, sigma_env: float,
                   gamma_target: float = 8.1, alpha_trap: float = 0.5) -> float:
        """
        PPO reward with CJPT phase transition shaping.

        Parameters
        ----------
        gammas : np.ndarray
            Floquet eigenvalue trajectory
        trap_score : float
            Smooth trap door score from trap_door_detector() in [0, 3].
            DO NOT pass geometric_trap_score output here.
        delta_kk : float
            Causal deviation
        J_bound : float
            Jacobi bound
        sigma_env : float
            Gaussian envelope width
        gamma_target : float
            Target Floquet eigenvalue
        alpha_trap : float
            Trap penalty weight

        Returns
        -------
        float
            Shaped reward
        """
        base = -np.abs(gammas[-1] - gamma_target)
        
        # Reward causal deviation UP TO the bound, then penalize
        causal_reward = -np.maximum(0, delta_kk - 1.2*J_bound)**2
        
        # Trap door becomes smooth phase indicator
        phase_penalty = trap_score * alpha_trap * np.exp(-sigma_env / (1e-8 + 1e-12))
        
        return base + 0.5*causal_reward - phase_penalty
    
    def build_ppo_state(self, phi: np.ndarray, phi_dot: np.ndarray, H: float, 
                       eps: float, M_eigs: np.ndarray, Omega: np.ndarray,
                       delta_kk: float, sigma_env: float, J_bound: float) -> np.ndarray:
        """
        Build augmented PPO state vector with transport and symplectic features.
        
        State vector structure:
        - Kinematics: [phi_H, phi_S, phi_dot_H, phi_dot_S, H, eps]
        - Transport: [lam_min_norm, lam_max_norm, trace_norm]
        - Symplectic: [log_det_symp, antisym_norm, condition_num]
        - Causal: [delta_kk, sigma_env, J_bound]
        
        Total: 15 dimensions
        
        Parameters
        ----------
        phi : np.ndarray
            Field values [phi_H, phi_S]
        phi_dot : np.ndarray
            Field velocities
        H : float
            Hubble parameter
        eps : float
            Slow-roll parameter
        M_eigs : np.ndarray
            Transport matrix eigenvalues
        Omega : np.ndarray
            Symplectic matrix
        delta_kk : float
            Causal deviation
        sigma_env : float
            Gaussian envelope width
        J_bound : float
            Jacobi bound
            
        Returns
        -------
        np.ndarray
            Normalized state vector (15,)
        """
        STATE_NORM = {
            "phi": 1e17, "phi_dot": 1e17, "H": 1e13, "eps": 1.0,
            "lam_norm": 100.0, "trace_norm": 100.0,
            "log_det_symp": 1.0, "antisym_norm": 1.0, "condition_num": 10.0,
            "delta_kk": 0.1, "sigma_env": 1e-4, "J_bound": 1e-8
        }
        
        # Kinematics
        s_kin = np.concatenate([phi / STATE_NORM["phi"], 
                               phi_dot / STATE_NORM["phi_dot"],
                               [H / STATE_NORM["H"], eps / STATE_NORM["eps"]]])
        
        # Transport spectrum
        lam_min = np.min(M_eigs) / (H**2)
        lam_max = np.max(M_eigs) / (H**2)
        trace = np.sum(M_eigs) / (H**2)
        s_trans = np.array([lam_min, lam_max, trace]) / STATE_NORM["lam_norm"]
        
        # Symplectic structure
        det_O = np.log(np.abs(np.linalg.det(Omega)))
        anti_O = np.linalg.norm(Omega - np.linalg.inv(Omega).T, ord='fro')
        cond_O = np.linalg.cond(Omega)
        s_symp = np.array([det_O, anti_O, cond_O]) / np.array([
            STATE_NORM["log_det_symp"], 
            STATE_NORM["antisym_norm"], 
            STATE_NORM["condition_num"]
        ])
        
        # Causal order parameters
        s_causal = np.array([delta_kk, sigma_env, J_bound]) / np.array([
            STATE_NORM["delta_kk"],
            STATE_NORM["sigma_env"],
            STATE_NORM["J_bound"]
        ])
        
        state = np.concatenate([s_kin, s_trans, s_symp, s_causal])
        self.delta_kk_history.append(delta_kk)
        self.sigma_env_history.append(sigma_env)
        return state
    
    def log_diagnostics(self, step: int, phase: str, g_trap: float, 
                       delta_kk: float, sigma_env: float, J_bound: float):
        """Log key diagnostics at each rollout step."""
        logger.info(
            f"Step {step:04d} | Phase: {phase:20s} | "
            f"G_trap: {g_trap:6.3f} | "
            f"Δ_KK: {delta_kk:10.3e} | "
            f"σ_env: {sigma_env:10.3e} | "
            f"J_bound: {J_bound:10.3e}"
        )
