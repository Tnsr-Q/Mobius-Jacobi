"""
AletheiaEnv — Gymnasium PPO wrapper for the CJPT framework.

Exposes the 15-dimensional CJPT state vector, a continuous 2-D action space,
and the CJPT-shaped reward function.  Fully compatible with stable_baselines3.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AletheiaEnv(gym.Env):
    """
    Gymnasium environment that wires the CJPT system and Jacobi ODE solver
    into a reinforcement-learning loop.

    Observation (15-D)
    ------------------
    As defined by ``CJPTSystem.build_ppo_state``:
    - Kinematics   : [phi_H, phi_S, phi_dot_H, phi_dot_S, H, eps]       (6)
    - Transport    : [lam_min_norm, lam_max_norm, trace_norm]            (3)
    - Symplectic   : [log_det_symp, antisym_norm, condition_num]         (3)
    - Causal       : [delta_kk, sigma_env, J_bound]                      (3)

    Action (2-D continuous)
    -----------------------
    - delta_phi_init  : perturbation to the initial Jacobi amplitude
    - trajectory_bend : fractional bend of the slow-roll trajectory
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, cjpt_system, ode_solver, max_steps: int = 2000):
        """
        Parameters
        ----------
        cjpt_system : CJPTSystem
            Initialised CJPT orchestrator.
        ode_solver : JacobiODESolver
            Initialised Jacobi ODE solver.
        max_steps : int
            Maximum episode length before truncation.
        """
        super().__init__()
        self.cjpt = cjpt_system
        self.ode = ode_solver
        self.max_steps = max_steps

        # 15-D state vector (see CJPTSystem.build_ppo_state)
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(15,), dtype=np.float32
        )

        # Action: [delta_phi_init, trajectory_bend_angle]
        self.action_space = spaces.Box(
            low=np.array([-0.1, -0.5], dtype=np.float32),
            high=np.array([0.1, 0.5], dtype=np.float32),
            dtype=np.float32,
        )

        self.t = 0
        self.current_state = None
        self._omega = None
        self._R_s = None
        self._sigma_env = None
        self._background = None
        self._trap_history = []
        self._reward_history = []

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """
        Reset the environment and return the initial observation.

        Parameters
        ----------
        seed : int or None
        options : dict or None
            Optional keys:
            - ``delta_phi`` (float): initial field perturbation (default 0).
            - ``action`` (array): 2-D action applied at reset (default zeros).

        Returns
        -------
        obs : np.ndarray, shape (15,)
        info : dict
        """
        super().reset(seed=seed)
        self.t = 0
        options = options or {}

        delta_phi = options.get("delta_phi", 0.0)
        action = np.asarray(options.get("action", np.zeros(2)), dtype=float)

        # USR background trajectory (mild Hubble modulation + USR decay)
        def background(t):
            H = self.ode.H0 * (1.0 + 0.01 * np.sin(t * 0.05))
            eps = 1e-3 * np.exp(-t / 500.0)
            phi = 1e17 + delta_phi * 1e16
            return H, eps, phi

        # Initial Jacobi conditions perturbed by action
        y0 = [1e-8, 1e-8 + float(action[0]), 0.0, 0.0]

        t, J = self.ode.solve((0, 500), y0, background)
        omega, R_s, sigma_env, strain_3d = self.ode.spectral_response(t, J)

        self._omega = omega
        self._R_s = R_s
        self._sigma_env = sigma_env
        self._background = background
        self._trap_history = []
        self._reward_history = []

        # Compute diagnostics at t=0
        H_t, eps_t, _ = background(0)
        J_bound = self.cjpt.compute_J_bound(self.ode.f2, H_t)
        # reset(): use a small fixed fraction of J_bound as the initial deviation
        # (well inside the causal region).  In step() this drifts toward J_bound
        # to simulate a realistic causal trajectory.  Replace both with
        # CausalEnforcer.compute_causal_deviation() once a full rollout is wired.
        delta_kk = 0.05 * J_bound

        M_matrix = self.ode._transport_matrix(0, H_t, eps_t, 0.0)
        M_eigs = np.linalg.eigvalsh(M_matrix)
        Omega = np.eye(4)  # Placeholder symplectic; replace with Magnus integrator

        self.current_state = self.cjpt.build_ppo_state(
            np.array([y0[0], y0[1]]),
            np.array([y0[2], y0[3]]),
            H_t, eps_t, M_eigs, Omega, delta_kk, sigma_env, J_bound,
        )

        return self.current_state.astype(np.float32), {}

    def step(self, action):
        """
        Advance the environment by one step.

        Parameters
        ----------
        action : array-like, shape (2,)
            [delta_phi_init, trajectory_bend_angle]

        Returns
        -------
        obs : np.ndarray, shape (15,)
        reward : float
        terminated : bool
        truncated : bool
        info : dict
        """
        self.t += 1
        delta_phi, bend = float(action[0]), float(action[1])

        H, eps, phi = self._background(self.t)
        H += 1e7 * delta_phi
        eps *= 1.0 + 0.01 * bend

        # Recompute transport matrix and diagnostics at the current env time
        M_matrix = self.ode._transport_matrix(self.t, H, eps, phi)
        M_eigs = np.linalg.eigvalsh(M_matrix)
        J_bound = self.cjpt.compute_J_bound(self.ode.f2, H)
        sigma_env = self._sigma_env * (1.0 + 0.001 * delta_phi)

        # Simulated causal drift: delta_kk drifts toward J_bound over episode
        delta_kk = J_bound * (0.8 + 0.4 * self.t / self.max_steps)

        Omega = np.eye(4)  # Placeholder symplectic

        g_trap = self.cjpt.geometric_trap_score(H, self.ode.M2, M_matrix, Omega)
        s_trap = self.cjpt.trap_door_detector(H, self.ode.M2, M_matrix, Omega)
        phase = self.cjpt.cjpt_phase_check(
            g_trap, delta_kk, J_bound, sigma_env, self.ode.xi_H, H
        )

        # Reward: single Floquet track placeholder
        gammas = np.array([8.0])
        reward = float(self.cjpt.cjpt_reward(
            gammas, s_trap, delta_kk, J_bound, sigma_env
        ))

        # Update 15-D state
        self.current_state = self.cjpt.build_ppo_state(
            np.array([phi, phi + 1e-8]),
            np.array([0.0, 0.0]),
            H, eps, M_eigs, Omega, delta_kk, sigma_env, J_bound,
        )

        self._trap_history.append(g_trap)
        self._reward_history.append(reward)

        terminated = bool(g_trap > 1.5)
        truncated = bool(self.t >= self.max_steps and not terminated)

        return (
            self.current_state.astype(np.float32),
            reward,
            terminated,
            truncated,
            {"phase": phase, "g_trap": g_trap, "sigma_env": sigma_env},
        )

    def render(self):
        """Print a brief summary of the current state (human render mode)."""
        if self.current_state is not None:
            print(
                f"Step {self.t:04d} | "
                f"state_norm={np.linalg.norm(self.current_state):.4f} | "
                f"σ_env={self._sigma_env:.2e}"
            )
