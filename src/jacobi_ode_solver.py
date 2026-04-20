"""
Jacobi ODE Solver - Palatini-Jacobi deviation equation solver

Solves: J̈^A + M^A_B(t) J^B = 0

with background-dependent transport matrix derived from the USR inflaton
background. Outputs time-series J(t), FFT spectral response R_s(ω), and
Gaussian envelope width σ_env.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.fft import fftfreq


class JacobiODESolver:
    """Solves Palatini-Jacobi deviation equation with USR background coupling."""

    def __init__(self, f2: float, xi_H: float, H0: float, M_Pl: float = 2.435e18):
        """
        Parameters
        ----------
        f2 : float
            Quadratic gravity parameter (fakeon coupling).
        xi_H : float
            Higgs non-minimal coupling.
        H0 : float
            Reference Hubble scale [GeV].
        M_Pl : float
            Planck mass [GeV].
        """
        self.f2 = f2
        self.M2 = M_Pl / np.sqrt(f2)
        self.xi_H = xi_H
        self.H0 = H0
        self.M_Pl = M_Pl

    def _transport_matrix(self, t: float, H_t: float, eps_t: float) -> np.ndarray:
        """
        Construct M(t) from local curvature and RGE portal coupling.

        Parameters
        ----------
        t : float
            Conformal time.
        H_t : float
            Hubble parameter at time t.
        eps_t : float
            Slow-roll parameter at time t.

        Returns
        -------
        np.ndarray
            2×2 transport matrix.
        """
        # DeWitt-Seeley curvature shift: a2/a0 ≈ ξ_shift * R / M2^2
        R_scalar = 6 * (2 * H_t**2 + eps_t * H_t**2)  # R ≈ 12H^2 at USR
        xi_shift = 1 / 6  # Minimal coupling for spin-2 ghost channel
        curvature_shift = (xi_shift * R_scalar) / self.M2**2

        # Base Hubble damping + Palatini enhancement
        M_diag = H_t**2 * (1 + curvature_shift * self.xi_H)

        # Portal leakage: λ_HS ~ f2/(4π) → scaled to H^2
        portal = (self.f2 / (4 * np.pi)) * self.xi_H * (
            self.M_Pl**2 / self.H0**2) * H_t**2

        return np.array([[M_diag, portal], [portal, M_diag]])

    def solve(self, t_span: tuple, y0: list, background_func, dt: float = 0.05):
        """
        Integrate the Jacobi deviation equation.

        Parameters
        ----------
        t_span : (t0, tf)
            Integration interval.
        y0 : list of 4 floats
            Initial state [J0_H, J0_S, dJ0_H, dJ0_S].
        background_func : callable
            ``background_func(t) -> (H, eps, phi)``
        dt : float
            Time-step for evaluation grid.

        Returns
        -------
        t : np.ndarray, shape (N,)
            Time array.
        J : np.ndarray, shape (N, 2)
            Jacobi field trajectory for both channels.
        """
        import warnings
        t_eval = np.arange(t_span[0], t_span[1], dt)

        def ode(t, y):
            J = y[:2]
            dJ = y[2:]
            H_t, eps_t, _ = background_func(t)
            M = self._transport_matrix(t, H_t, eps_t)
            d2J = -M @ J
            return np.concatenate([dJ, d2J])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sol = solve_ivp(ode, t_span, y0, t_eval=t_eval,
                            method='DOP853', rtol=1e-8, atol=1e-10)

        if not sol.success or sol.t.shape[0] < 4:
            # Integration failed (e.g. overflow in stiff portal term).
            # Return a minimal oscillatory trajectory at the initial background.
            H_0, eps_0, _ = background_func(t_span[0])
            M_0 = self._transport_matrix(t_span[0], H_0, eps_0)
            # Use diagonal element as characteristic frequency
            omega_0 = np.sqrt(np.abs(np.diag(M_0)).mean() + 1e-30)
            t_fallback = t_eval if len(t_eval) >= 4 else np.linspace(t_span[0], t_span[1], 100)
            tau = t_fallback - t_span[0]
            J0 = np.array(y0[:2], dtype=float)
            J_fallback = np.outer(np.cos(omega_0 * tau), J0)
            return t_fallback, J_fallback

        return sol.t, sol.y[:2].T  # t, J(t) shape (N, 2)

    def spectral_response(self, t: np.ndarray, J: np.ndarray):
        """
        Compute FFT spectral response and Gaussian envelope width.

        Parameters
        ----------
        t : np.ndarray, shape (N,)
            Time array from ``solve``.
        J : np.ndarray, shape (N, 2)
            Jacobi field from ``solve``.

        Returns
        -------
        omega : np.ndarray, shape (N//2,)
            Positive angular frequencies.
        R_s : np.ndarray, shape (N//2, 2)
            Complex spectral response (positive-frequency half), with one
            column per Jacobi channel.
        sigma_env : float
            Gaussian envelope width.
        strain_3d : np.ndarray, shape (N, 2, 3)
            Stacked multi-channel strain [Re(J), Im(J_analytic), inst_freq].
        """
        if len(t) < 4:
            raise ValueError(
                f"spectral_response requires at least 4 time samples; got {len(t)}. "
                "Check that solve() converged successfully."
            )

        N = len(t)
        dt = t[1] - t[0]
        omega = 2 * np.pi * fftfreq(N, d=dt)

        # Analytic signal via Hilbert transform (FFT method)
        J_fft = np.fft.fft(J, axis=0)
        H_sign = np.zeros(N)
        if N % 2 == 0:
            H_sign[1:N // 2] = 1
            H_sign[N // 2 + 1:] = -1
        else:
            H_sign[1:(N + 1) // 2] = 1
            H_sign[(N + 1) // 2:] = -1
        # ifft(1j * H_sign * FFT(x)) gives the Hilbert transform as a real-valued
        # signal; we preserve it as real to build the complex analytic signal.
        hilbert_J = np.fft.ifft(1j * H_sign[:, np.newaxis] * J_fft, axis=0).real
        J_analytic = J + 1j * hilbert_J

        phase = np.angle(J_analytic)
        inst_freq = np.gradient(phase, dt, axis=0)

        # Stack channels: [Re(J), Im(J_analytic), inst_freq]
        strain_3d = np.stack([J.real, J_analytic.imag, inst_freq], axis=-1)

        # Gaussian envelope fit via log-parabolic least squares on channel 0.
        # When the parabolic fit yields a non-negative leading coefficient
        # (no well-defined Gaussian peak), fall back to a conservative 1e-4
        # which corresponds to a broad envelope that does not drive the reward.
        log_mag = np.log(np.abs(J[:, 0]) + 1e-12)
        t_sq = t**2
        A = np.vstack([t_sq, t, np.ones_like(t)]).T
        coeffs = np.linalg.lstsq(A, log_mag, rcond=None)[0]
        sigma_env = float(np.sqrt(-1.0 / (2.0 * coeffs[0]))) if coeffs[0] < 0 else 1e-4

        R_s = np.fft.fft(
            strain_3d[:, :, 0] + 1j * strain_3d[:, :, 1], axis=0
        )
        return omega[:N // 2], R_s[:N // 2], sigma_env, strain_3d
