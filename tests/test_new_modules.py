"""
Tests for the three new CJPT modules:
  - JacobiODESolver
  - compute_causal_covariance
  - AletheiaEnv
"""

import sys
import os
import numpy as np
import pytest

# Allow direct imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from jacobi_ode_solver import JacobiODESolver
from causal_enforcer import compute_causal_covariance
from cjpt_system import CJPTSystem
from aletheia_env import AletheiaEnv


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

F2 = 1e-8
XI_H = 5e8
H0 = 1e9
T_SPAN = (0, 10)  # Short span for fast tests
DT = 0.5


def _background(t):
    H = H0 * (1.0 + 0.01 * np.sin(t * 0.05))
    eps = 1e-3 * np.exp(-t / 500.0)
    phi = 1e17
    return H, eps, phi


@pytest.fixture(scope="module")
def ode_solver():
    return JacobiODESolver(f2=F2, xi_H=XI_H, H0=H0)


@pytest.fixture(scope="module")
def solved(ode_solver):
    t, J = ode_solver.solve(T_SPAN, [1e-8, 1e-8, 0.0, 0.0], _background, dt=DT)
    return t, J


@pytest.fixture(scope="module")
def spectral(ode_solver, solved):
    t, J = solved
    return ode_solver.spectral_response(t, J)


# ---------------------------------------------------------------------------
# JacobiODESolver tests
# ---------------------------------------------------------------------------

class TestJacobiODESolver:
    def test_solve_returns_correct_shapes(self, solved):
        t, J = solved
        assert t.ndim == 1
        assert J.ndim == 2
        assert J.shape[1] == 2, "J should have 2 channels (H and S)"
        assert len(t) == J.shape[0]

    def test_solve_at_least_four_samples(self, solved):
        t, J = solved
        assert len(t) >= 4, "ODE solver must return ≥4 samples"

    def test_transport_matrix_symmetric(self, ode_solver):
        M = ode_solver._transport_matrix(0, H0, 1e-3, 1e17)
        assert M.shape == (2, 2)
        np.testing.assert_allclose(M, M.T, rtol=1e-10)

    def test_spectral_response_shapes(self, solved, spectral):
        t, J = solved
        omega, R_s, sigma_env, strain_3d = spectral
        N = len(t)
        assert omega.shape == (N // 2,)
        assert R_s.shape[0] == N // 2
        assert strain_3d.shape == (N, 2, 3)

    def test_spectral_sigma_env_positive(self, spectral):
        _, _, sigma_env, _ = spectral
        assert sigma_env > 0

    def test_spectral_response_raises_on_short_array(self, ode_solver):
        t_short = np.linspace(0, 1, 3)
        J_short = np.ones((3, 2))
        with pytest.raises(ValueError, match="at least 4 time samples"):
            ode_solver.spectral_response(t_short, J_short)

    def test_fallback_on_diverging_ode(self, ode_solver):
        """ODE with extreme initial conditions should still return ≥4 points."""
        t_bad, J_bad = ode_solver.solve(
            (0, 10), [1e30, 1e30, 0.0, 0.0], _background, dt=0.5
        )
        assert len(t_bad) >= 4


# ---------------------------------------------------------------------------
# compute_causal_covariance tests
# ---------------------------------------------------------------------------

class TestComputeCausalCovariance:
    def test_returns_identity_when_no_mask(self, spectral):
        omega, R_s, _, _ = spectral
        # Use a delta_kk array entirely outside the causal window
        delta_kk_outside = np.zeros(len(omega))  # all zero, far from J_bound=1e30
        P, eigs = compute_causal_covariance(omega, R_s, delta_kk_outside, J_bound=1e30)
        np.testing.assert_array_equal(P, np.eye(3))
        np.testing.assert_array_equal(eigs, np.zeros(3))

    def test_projection_matrix_shape_and_symmetric(self, spectral):
        omega, R_s, _, _ = spectral
        delta_kk = np.abs(np.angle(R_s[:, 0]))
        J_bound = float(np.median(delta_kk)) + 1e-30
        P, eigs = compute_causal_covariance(omega, R_s, delta_kk, J_bound)
        assert P.ndim == 2
        assert P.shape[0] == P.shape[1]
        np.testing.assert_allclose(P, P.T, atol=1e-10)

    def test_projection_idempotent(self, spectral):
        """P_causal @ P_causal ≈ P_causal (idempotent projection)."""
        omega, R_s, _, _ = spectral
        delta_kk = np.abs(np.angle(R_s[:, 0]))
        J_bound = float(np.median(delta_kk)) + 1e-30
        P, _ = compute_causal_covariance(omega, R_s, delta_kk, J_bound)
        d = P.shape[0]
        if not np.allclose(P, np.eye(d)):  # Skip identity fallback
            np.testing.assert_allclose(P @ P, P, atol=1e-8)

    def test_eigenvalues_descending(self, spectral):
        omega, R_s, _, _ = spectral
        delta_kk = np.linspace(0.8e-5, 1.2e-5, len(omega))
        J_bound = 1e-5
        P, eigs = compute_causal_covariance(omega, R_s, delta_kk, J_bound)
        if len(eigs) > 1 and not np.allclose(eigs, 0):
            assert eigs[0] >= eigs[-1], "eigenvalues must be in descending order"


# ---------------------------------------------------------------------------
# AletheiaEnv tests
# ---------------------------------------------------------------------------

class TestAletheiaEnv:
    @pytest.fixture
    def env(self, ode_solver):
        cjpt = CJPTSystem({"f2": F2, "xi_H": XI_H, "M_Pl": 2.435e18})
        return AletheiaEnv(cjpt_system=cjpt, ode_solver=ode_solver, max_steps=5)

    def test_observation_space(self, env):
        assert env.observation_space.shape == (15,)

    def test_action_space(self, env):
        assert env.action_space.shape == (2,)

    def test_reset_returns_correct_obs_shape(self, env):
        obs, info = env.reset()
        assert obs.shape == (15,)
        assert obs.dtype == np.float32

    def test_step_returns_correct_types(self, env):
        env.reset()
        action = np.array([0.01, 0.1], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (15,)
        assert obs.dtype == np.float32
        assert isinstance(float(reward), float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "phase" in info
        assert "g_trap" in info
        assert "sigma_env" in info

    def test_terminates_within_max_steps(self, env):
        env.reset()
        for _ in range(20):  # More than max_steps=5
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break
        else:
            pytest.fail("Episode did not terminate within 20 steps")

    def test_obs_within_bounds(self, env):
        """Observations should be finite (though may exceed [-10, 10])."""
        obs, _ = env.reset()
        assert np.all(np.isfinite(obs)), "reset observation must be finite"
        obs2, _, _, _, _ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs2)), "step observation must be finite"
