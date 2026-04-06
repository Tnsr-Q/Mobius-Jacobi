import numpy as np
try:
    import vibetensor as vbt
    HAS_VBT = True
except ImportError:
    HAS_VBT = False
    print("Warning: vibetensor not found, using numpy fallback")


def compute_causal_covariance(omega: np.ndarray, R_s: np.ndarray,
                               delta_kk: np.ndarray, J_bound: float,
                               eta: float = 0.8):
    """
    Compute C_Δ eigendecomposition over the causal bound region.

    Replaces simplified geometric rotation with eigendecomposition of the
    causal deviation covariance matrix.  The resulting projection matrices
    are aligned to the causal bound B(ω).

    Parameters
    ----------
    omega : np.ndarray, shape (M,)
        Angular frequency array (positive half from FFT).
    R_s : np.ndarray, shape (M, …)
        Complex spectral response array; at least one frequency dimension.
    delta_kk : np.ndarray, shape (M,)
        Per-frequency causal deviation array.  Values should be of the same
        order of magnitude as ``J_bound`` for the window mask to be non-empty.
    J_bound : float
        Jacobi amplitude bound from microcausality.
    eta : float
        Causal-window boundary factor.  The window is
        ``[eta * J_bound, (2 - eta) * J_bound]``, which is symmetric around
        ``J_bound``.  With the default ``eta=0.8`` this gives
        ``[0.8, 1.2] * J_bound``.

    Returns
    -------
    P_causal : np.ndarray, shape (d, d)
        Rank-2 projection matrix onto the dominant causal subspace.
        Falls back to identity(3) when the bound window is empty.
    eigvals : np.ndarray, shape (d,)
        Eigenvalues of C_Δ sorted in descending order.
    """
    lo = eta * J_bound
    hi = (2.0 - eta) * J_bound  # symmetric window around J_bound
    mask = (delta_kk >= lo) & (delta_kk <= hi)

    if not np.any(mask):
        return np.eye(3), np.zeros(3)

    # Select rows corresponding to the causal bound window
    R_window = R_s[mask]
    weights = delta_kk[mask] / (hi + 1e-30)  # Normalize to (0, 1]

    # Flatten to 2-D real matrix for covariance (use real part of R_s)
    R_real = R_window.real
    if R_real.ndim == 1:
        R_real = R_real[:, np.newaxis]
    # Collapse any extra dims to single feature vector per frequency
    R_real = R_real.reshape(R_real.shape[0], -1)

    # Outer-product integral: C_Δ = ∫ w(ω) Δ(ω) Δ†(ω) dω
    # Approximated as weighted sample covariance
    if R_real.shape[1] < 2:
        # Not enough features for covariance — return identity
        return np.eye(3), np.zeros(3)

    # Use at most 3 features (match default projection rank)
    n_feat = min(R_real.shape[1], 3)
    R_feat = R_real[:, :n_feat]

    C_delta = np.cov(R_feat.T, aweights=weights)
    if C_delta.ndim == 0:
        C_delta = np.array([[float(C_delta)]])

    # Eigendecomposition — Möbius alignment basis
    eigvals, eigvecs = np.linalg.eigh(C_delta)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Top-2 projection matrix (rank-2 subspace)
    d = eigvecs.shape[0]
    rank = min(2, d)
    P_causal = eigvecs[:, :rank] @ eigvecs[:, :rank].T

    return P_causal, eigvals


class CausalEnforcer:
    def __init__(self, kk_tolerance: float = 0.8, enforce_kk: bool = True):
        self.kk_tolerance = kk_tolerance
        self.enforce_kk = enforce_kk
        self.causal_deviation_history = []

    def compute_causal_deviation(self, omega, R_s) -> float:
        """
        Compute normalized causal deviation from KK minimum-phase constraint.
        
        Delta_KK(ω) = ||arg[R_s(ω)] - H[ln|R_s(ω)|]||_2 / ||R_s(ω)||_2
        
        Parameters
        ----------
        omega : array
            Angular frequency array
        R_s : array
            Complex spectral response
            
        Returns
        -------
        float
            Normalized causal deviation
        """
        if HAS_VBT:
            magnitude = vbt.abs(R_s)
            phase = vbt.angle(R_s)
            
            # Log-magnitude (clipped to avoid log(0))
            log_magnitude = vbt.log(vbt.clip(magnitude, 1e-10, None))
            
            # Hilbert transform of log-magnitude
            H_sign = self._hilbert_sign_mask(len(omega), rfft=False)
            H_logR = vbt.real(vbt.ifft(1j * H_sign * vbt.fft(log_magnitude)))
            
            # KK-derived phase (minimum phase)
            phase_kk = H_logR
            
            # Compute deviation
            deviation_num = vbt.norm(phase - phase_kk, p=2)
            deviation_denom = vbt.norm(R_s, p=2)
        else:
            # Numpy fallback
            magnitude = np.abs(R_s)
            phase = np.angle(R_s)
            
            # Log-magnitude (clipped)
            log_magnitude = np.log(np.clip(magnitude, 1e-10, None))
            
            # Hilbert transform via FFT
            H_sign = self._hilbert_sign_mask_np(len(omega), rfft=False)
            H_logR = np.real(np.fft.ifft(1j * H_sign * np.fft.fft(log_magnitude)))
            
            phase_kk = H_logR
            
            deviation_num = np.linalg.norm(phase - phase_kk)
            deviation_denom = np.linalg.norm(R_s)
        
        delta_kk = float(deviation_num / max(deviation_denom, 1e-12))
        self.causal_deviation_history.append(delta_kk)
        return delta_kk

    def adaptive_tolerance(self, delta_kk: float, J_bound: float, eta: float = 0.8) -> float:
        """
        Dynamically set kk_tolerance to track the causal boundary.
        
        Parameters
        ----------
        delta_kk : float
            Current causal deviation
        J_bound : float
            Jacobi bound from fakeon mass
        eta : float
            Anselmi-Piva contour scaling factor (~0.8)
            
        Returns
        -------
        float
            Adaptive tolerance value
        """
        # Tolerance tracks the bound shape: kk_tol = f(delta_kk, J_bound)
        # When delta_kk approaches eta * J_bound, increase tolerance
        delta_c = eta * J_bound
        
        if delta_kk < 0.5 * delta_c:
            # Well within causal region - strict enforcement
            return 0.9
        elif delta_kk < delta_c:
            # Approaching boundary - moderate enforcement
            return 0.7
        elif delta_kk < 1.2 * delta_c:
            # At boundary - adaptive tracking
            return 0.5
        else:
            # Beyond boundary - dual field regime
            return 0.3

    def apply_kramers_kronig(self, omega, R_s):
        """
        GPU-accelerated KK causality enforcement on a complex spectral response.

        Parameters
        ----------
        omega : array
            Angular frequency array (rad/s), full-spectrum (N bins from fft, NOT rfft).
        R_s : array
            Complex spectral response to enforce causality on.

        Returns
        -------
        array
            Causality-enforced complex spectral response.
        """
        if not self.enforce_kk:
            return R_s

        if HAS_VBT:
            # VibeTensor implementation
            domega = omega[1] - omega[0]
            if not vbt.allclose(vbt.diff(omega), domega):
                print("Warning: Uneven omega spacing – skipping KK enforcement")
                return R_s

            magnitude = vbt.abs(R_s)
            phase = vbt.angle(R_s)

            log_magnitude = vbt.log(vbt.clip(magnitude, 1e-10, None))
            H_sign = self._hilbert_sign_mask(len(omega), rfft=False)
            H_logR = vbt.real(vbt.ifft(1j * H_sign * vbt.fft(log_magnitude)))

            phase_kk = H_logR
            phase_corrected = (1 - self.kk_tolerance) * phase + self.kk_tolerance * phase_kk

            return magnitude * (vbt.cos(phase_corrected) + 1j * vbt.sin(phase_corrected))
        else:
            # Numpy fallback
            domega = omega[1] - omega[0]
            if not np.allclose(np.diff(omega), domega):
                print("Warning: Uneven omega spacing – skipping KK enforcement")
                return R_s

            magnitude = np.abs(R_s)
            phase = np.angle(R_s)

            log_magnitude = np.log(np.clip(magnitude, 1e-10, None))
            H_sign = self._hilbert_sign_mask_np(len(omega), rfft=False)
            H_logR = np.real(np.fft.ifft(1j * H_sign * np.fft.fft(log_magnitude)))

            phase_kk = H_logR
            phase_corrected = (1 - self.kk_tolerance) * phase + self.kk_tolerance * phase_kk

            return magnitude * (np.cos(phase_corrected) + 1j * np.sin(phase_corrected))

    @staticmethod
    def _hilbert_sign_mask(N: int, rfft: bool = False):
        """VibeTensor version of Hilbert sign mask"""
        if rfft:
            H = vbt.ones(N)
            H[0] = 0
            if N % 2 == 0:
                H[-1] = 0
            return H

        H = vbt.zeros(N)
        if N % 2 == 0:
            H[1:N // 2] = 1
            H[N // 2 + 1:] = -1
        else:
            H[1:(N + 1) // 2] = 1
            H[(N + 1) // 2:] = -1
        return H

    @staticmethod
    def _hilbert_sign_mask_np(N: int, rfft: bool = False):
        """Numpy version of Hilbert sign mask"""
        if rfft:
            H = np.ones(N)
            H[0] = 0
            if N % 2 == 0:
                H[-1] = 0
            return H

        H = np.zeros(N)
        if N % 2 == 0:
            H[1:N // 2] = 1
            H[N // 2 + 1:] = -1
        else:
            H[1:(N + 1) // 2] = 1
            H[(N + 1) // 2:] = -1
        return H

    def get_params(self) -> dict:
        return {"kk_tolerance": self.kk_tolerance, "enforce_kk": self.enforce_kk}

    def set_params(self, params: dict):
        if "kk_tolerance" in params:
            self.kk_tolerance = params["kk_tolerance"]
        if "enforce_kk" in params:
            self.enforce_kk = params["enforce_kk"]

    def __repr__(self):
        return (
            f"CausalEnforcer(kk_tolerance={self.kk_tolerance}, "
            f"enforce_kk={self.enforce_kk})"
        )
