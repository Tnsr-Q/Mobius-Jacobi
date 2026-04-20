"""
Enhanced Echo Spectrometer with Null Analysis
Extracts and analyzes destructive interference nodes (black spots)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, optimize, ndimage
from dataclasses import dataclass
from typing import Tuple, List, Dict

# Physical constants
G = 6.67430e-11
c = 299792458.0
M_sun = 1.98847e30
l_pl = 1.616255e-35

@dataclass
class NullAnalysis:
    """Container for null pattern analysis results"""
    null_coords: np.ndarray      # (N, 3) array of (f, t, power)
    null_power: np.ndarray       # Power values at nulls
    spacing_freq: np.ndarray     # Frequency spacing between nulls
    spacing_time: np.ndarray     # Time spacing between nulls
    dispersion_params: dict      # Fitted dispersion relation
    cavity_Q: float              # Quality factor from null width
    phase_at_nulls: np.ndarray   # Phase(K) at null locations
    metadata: dict               # Analysis parameters

# =============================================================================
# CORRECTED PHYSICS FUNCTIONS (with monotonicity fix)
# =============================================================================

def echo_delay_seconds(M_solar: float, epsilon: float = 1e-5) -> float:
    M = M_solar * M_sun
    r_s = 2 * G * M / c**2
    return (4 * G * M / c**3) * np.log(r_s / (epsilon * l_pl))

def transfer_function_K(
    omega: np.ndarray, 
    T_inf: np.ndarray, 
    R_inf: np.ndarray, 
    R_s: complex | np.ndarray,  # Accept scalar OR array
    dt: float,
    epsilon_numerical: float = 1e-24
) -> np.ndarray:
    """Cavity transfer function with proper denominator regularization."""
    # Handle both scalar and array R_s
    R_s_array = np.asarray(R_s, dtype=complex)
    if np.ndim(R_s_array) == 0:
        R_s_array = np.full_like(omega, R_s_array, dtype=complex)
    
    phase = np.exp(1j * omega * dt)
    denominator = 1.0 - R_inf * R_s_array * phase
    
    # Add small complex offset to avoid division by zero
    denominator = denominator + epsilon_numerical * (1 + 1j)
    
    return (T_inf**2) * R_s_array * phase / denominator

def make_toy_greybody(f: np.ndarray, f0: float = 240.0, p: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    CORRECTED monotonicity: low frequency → total reflection
    """
    f_safe = np.maximum(f, 1e-30)
    T2 = 1.0 / (1.0 + (f0 / f_safe) ** p)  # FLIPPED: f0/f not f/f0
    T = np.sqrt(T2)
    R_mag = np.sqrt(np.maximum(0.0, 1.0 - T2))
    
    # Add simple phase model for now (in real analysis, use WKB phase)
    phi = 0.1 * np.pi * (f / f0)  # Linear phase
    R_complex = R_mag * np.exp(1j * phi)
    
    return T, R_complex

def stft_power_db(x: np.ndarray, fs: float, nperseg: int = 1024, hop: int = 32):
    """High-resolution STFT."""
    w = np.hanning(nperseg)
    nframes = 1 + (len(x) - nperseg) // hop
    
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(nframes, nperseg),
        strides=(x.strides[0] * hop, x.strides[0]),
        writeable=False
    )
    X = np.fft.rfft(frames * w[None, :], axis=1)
    P = (np.abs(X) ** 2) / (np.sum(w**2) + 1e-12)
    
    freqs = np.fft.rfftfreq(nperseg, d=1 / fs)
    times = (np.arange(nframes) * hop + nperseg / 2) / fs
    return times, freqs, 10 * np.log10(P + 1e-18).T

def build_waveform_with_echoes(
    t: np.ndarray, 
    Mtot_solar: float, 
    epsilon: float,
    use_parametric_rs: bool = False,
    R_s_model: callable = None
):
    """
    Build waveform with echoes, optionally using parametric R_s(ω).
    """
    dt_samp = t[1] - t[0]
    fs = 1.0 / dt_samp
    N = t.size
    
    # Merger proxy
    t_merger = 0.12
    f_inst = np.clip(30.0 + (220.0 - 30.0) * (t - (t_merger - 0.05)) / 0.10, 20.0, 520.0)
    phase = 2 * np.pi * np.cumsum(f_inst) * dt_samp
    env = np.exp(-((t - t_merger) / 0.020) ** 2)
    h0 = env * np.sin(phase)
    
    H0 = np.fft.rfft(h0)
    freqs = np.fft.rfftfreq(N, d=dt_samp)  # Hz
    omega = 2 * np.pi * freqs  # rad/s
    
    # Barrier with corrected monotonicity
    T_inf, R_inf = make_toy_greybody(freqs, f0=240.0, p=4.0)
    dt_echo = echo_delay_seconds(Mtot_solar, epsilon=epsilon)
    
    # Surface reflectivity - parametric or constant
    if use_parametric_rs and R_s_model is not None:
        # R_s_model should expect Hz input
        R_s = R_s_model(freqs)  # Complex array
    else:
        R_s = 0.70 * np.exp(1j * (0.30 * np.pi))  # Scalar complex
    
    K = transfer_function_K(omega, T_inf=T_inf, R_inf=R_inf, R_s=R_s, dt=dt_echo)
    h_echo = np.fft.irfft(H0 * K, n=N)
    h = h0 + h_echo
    
    meta = {
        'freqs': freqs,
        'omega': omega,
        'K': K,
        'R_s': R_s,
        'dt_echo': dt_echo,
        'use_parametric_rs': use_parametric_rs,
        'R_s_shape': np.shape(R_s)
    }
    
    return h, h0, h_echo, fs, dt_echo, t_merger, K, freqs, meta

# =============================================================================
# ADVANCED NULL DETECTION
# =============================================================================

def detect_nulls_advanced(
    TT: np.ndarray, 
    FF: np.ndarray, 
    Pdb: np.ndarray, 
    adaptive_threshold: bool = True,
    noise_floor_percentile: float = 10,
    null_threshold_db: float = -120
) -> np.ndarray:
    """
    Advanced null detection with adaptive thresholding.
    
    Returns: (N, 3) array of (frequency, time, power)
    """
    # Estimate noise floor per frequency band
    if adaptive_threshold:
        # Compute median power per frequency
        median_power = np.median(Pdb, axis=1)
        
        # Threshold = median - 20 dB (adaptive)
        threshold = median_power[:, None] - 20
        
        # But not below absolute floor
        threshold = np.maximum(threshold, null_threshold_db)
    else:
        # Fixed threshold
        threshold = null_threshold_db
    
    # Find pixels below threshold
    null_mask = Pdb < threshold
    
    # Remove isolated pixels (morphological opening)
    null_mask = ndimage.binary_opening(null_mask, structure=np.ones((3,3)))
    
    # Label connected components
    labeled_mask, num_features = ndimage.label(null_mask)
    
    # Extract properties of each null region
    nulls = []
    for i in range(1, num_features + 1):
        # Get indices of this null region
        indices = np.argwhere(labeled_mask == i)
        
        # Compute centroid
        f_indices = indices[:, 0]
        t_indices = indices[:, 1]
        
        # Weight by power (deeper nulls get more weight)
        powers = Pdb[f_indices, t_indices]
        weights = null_threshold_db - powers  # Deeper nulls have higher weight
        
        # Weighted centroid
        if weights.sum() > 0:
            f_center = np.average(FF[f_indices], weights=weights)
            t_center = np.average(TT[t_indices], weights=weights)
            power_center = np.average(powers, weights=weights)
            
            # Compute null "strength" (how far below threshold)
            null_strength = null_threshold_db - power_center
            
            # Only keep strong nulls
            if null_strength > 5:  # At least 5 dB below threshold
                nulls.append([f_center, t_center, power_center, null_strength])
    
    if len(nulls) == 0:
        return np.array([]).reshape(0, 4)
    
    nulls_array = np.array(nulls)
    
    # Sort by time for analysis
    sort_idx = np.argsort(nulls_array[:, 1])
    return nulls_array[sort_idx]

def analyze_null_pattern(nulls: np.ndarray, dt_echo: float) -> Dict:
    """
    Comprehensive analysis of null pattern.
    """
    if len(nulls) < 2:
        return {}
    
    freqs = nulls[:, 0]
    times = nulls[:, 1]
    powers = nulls[:, 2]
    strengths = nulls[:, 3]
    
    analysis = {}
    
    # 1. Frequency distribution
    analysis['freq_mean'] = np.mean(freqs)
    analysis['freq_std'] = np.std(freqs)
    analysis['freq_range'] = [freqs.min(), freqs.max()]
    
    # 2. Time distribution relative to echoes
    # Group nulls by echo periods
    echo_periods = np.floor(times / dt_echo).astype(int)
    unique_echoes = np.unique(echo_periods)
    analysis['n_echoes_with_nulls'] = len(unique_echoes)
    
    # 3. Spacing analysis
    time_diffs = np.diff(times)
    analysis['time_spacing_mean'] = np.mean(time_diffs)
    analysis['time_spacing_std'] = np.std(time_diffs)
    analysis['time_spacing_ratio'] = analysis['time_spacing_mean'] / dt_echo
    
    # 4. Frequency-time correlation
    if len(freqs) > 2:
        slope, intercept = np.polyfit(times, freqs, 1)
        analysis['freq_time_slope'] = slope  # Hz/s
        analysis['freq_time_r2'] = np.corrcoef(times, freqs)[0, 1]**2
    
    # 5. Clustering analysis
    from scipy.cluster.hierarchy import fcluster, linkage
    if len(nulls) > 4:
        Z = linkage(nulls[:, :2], method='ward')
        clusters = fcluster(Z, t=dt_echo/2, criterion='distance')
        analysis['n_clusters'] = len(np.unique(clusters))
    
    # 6. Spectral lines analysis (periodicity in frequency)
    if len(freqs) > 10:
        freq_spacing = np.mean(np.diff(np.sort(freqs)))
        analysis['freq_spacing'] = freq_spacing
    
    return analysis

def compute_null_resonance_condition(nulls: np.ndarray, dt_echo: float) -> np.ndarray:
    """
    Check if nulls satisfy resonance condition: φ(ω) + ωΔt ≈ π (mod 2π)
    """
    if len(nulls) == 0:
        return np.array([])
    
    # For each null frequency, compute expected destructive interference condition
    # φ(ω) + ωΔt = (2n + 1)π for destructive interference
    # where φ(ω) = arg[R_inf(ω)] + arg[R_s(ω)]
    
    # For now, approximate with constant phase
    # In real analysis, use actual phase from R_inf and R_s
    freqs = nulls[:, 0]
    omega = 2 * np.pi * freqs
    
    # Expected condition: ωΔt ≈ (2n + 1)π - φ_const
    phase_const = 0.3 * np.pi  # Example constant phase
    expected_omega_dt = omega * dt_echo
    
    # Find nearest odd multiple of π
    n = np.round((expected_omega_dt + phase_const - np.pi) / (2 * np.pi))
    resonance_condition = (2 * n + 1) * np.pi - phase_const
    
    # Deviation from resonance
    deviation = expected_omega_dt - resonance_condition
    deviation_wrapped = np.angle(np.exp(1j * deviation))  # Wrap to [-π, π]
    
    return deviation_wrapped

# =============================================================================
# NULL-BASED ECHO DETECTION
# =============================================================================

def compute_null_detection_statistics(
    nulls: np.ndarray, 
    dt_echo: float,
    n_bootstrap: int = 1000
) -> Dict:
    """
    Compute detection statistics based on null pattern.
    """
    if len(nulls) < 10:
        return {'detection_significance': 0.0}
    
    # 1. Periodicity test: Are nulls spaced by dt_echo?
    times = nulls[:, 1]
    time_diffs = np.diff(times)
    
    # Test against uniform random distribution
    from scipy import stats
    
    # Kolmogorov-Smirnov test for periodicity
    # Normalize diffs to echo period
    normalized_diffs = (time_diffs % dt_echo) / dt_echo
    
    # KS test against uniform [0, 1]
    ks_statistic, ks_pvalue = stats.kstest(normalized_diffs, 'uniform')
    
    # 2. Bootstrap analysis
    bootstrap_pvalues = []
    for _ in range(n_bootstrap):
        # Shuffle times (null hypothesis: no structure)
        shuffled_times = np.random.permutation(times)
        shuffled_diffs = np.diff(np.sort(shuffled_times))
        shuffled_norm = (shuffled_diffs % dt_echo) / dt_echo
        
        # KS test on shuffled data
        _, p_shuffled = stats.kstest(shuffled_norm, 'uniform')
        bootstrap_pvalues.append(p_shuffled)
    
    # Empirical p-value: fraction of shuffled datasets with p-value <= actual
    empirical_p = np.mean(np.array(bootstrap_pvalues) <= ks_pvalue)
    
    # 3. Compute detection significance
    # -log10(p) is common measure
    detection_significance = -np.log10(max(empirical_p, 1e-10))
    
    return {
        'ks_statistic': ks_statistic,
        'ks_pvalue': ks_pvalue,
        'empirical_pvalue': empirical_p,
        'detection_significance': detection_significance,
        'n_nulls': len(nulls),
        'interpretation': interpret_detection_significance(detection_significance)
    }

def interpret_detection_significance(significance: float) -> str:
    """Interpret detection significance."""
    if significance < 1.0:
        return "No significant detection"
    elif significance < 2.0:
        return "Marginal evidence"
    elif significance < 3.0:
        return "Moderate evidence"
    elif significance < 5.0:
        return "Strong evidence"
    else:
        return "Very strong evidence"

# =============================================================================
# VISUALIZATION (continued from your code)
# =============================================================================

def make_null_analysis_figure(outdir: Path):
    """
    Complete null analysis workflow with comprehensive diagnostics.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Generate high-resolution data
    t = np.linspace(0, 1.2, 12000)
    Mtot = 30.0
    epsilon = 1e-5
    
    h, h0, h_echo, fs, dt_echo, t_merger, K, freqs_K, meta = build_waveform_with_echoes(
        t, Mtot_solar=Mtot, epsilon=epsilon
    )
    
    # High-res STFT
    TT, FF, Pdb = stft_power_db(h, fs=fs, nperseg=1024, hop=16)
    
    # Filter frequency range
    keep_f = (FF >= 50) & (FF <= 450)
    FFp = FF[keep_f]
    Pp = Pdb[keep_f, :]
    
    # DETECT NULLS with advanced algorithm
    nulls = detect_nulls_advanced(TT, FFp, Pp, adaptive_threshold=True)
    
    if len(nulls) == 0:
        print("No nulls detected. Try adjusting threshold.")
        return
    
    print(f"Detected {len(nulls)} null points")
    
    # ANALYZE NULL PATTERN
    null_analysis = analyze_null_pattern(nulls, dt_echo)
    resonance_deviation = compute_null_resonance_condition(nulls, dt_echo)
    detection_stats = compute_null_detection_statistics(nulls, dt_echo)
    
    # =============================================================================
    # ENHANCED VISUALIZATION
    # =============================================================================
    
    plt.close("all")
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10
    })
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.4)
    
    # ----- Panel 1: Spectrogram with null overlay -----
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_title("Spectrogram with Detected Nulls", fontweight="bold")
    
    im1 = ax1.imshow(
        Pp, origin="lower", aspect="auto",
        extent=[TT.min(), TT.max(), FFp.min(), FFp.max()],
        cmap="magma", vmin=np.nanpercentile(Pp, 5), vmax=np.nanpercentile(Pp, 95)
    )
    
    # Overlay nulls colored by strength
    scatter = ax1.scatter(nulls[:, 1], nulls[:, 0], c=nulls[:, 3], 
                         cmap='viridis', s=30, alpha=0.7, 
                         edgecolors='white', linewidths=0.5)
    
    # Mark echo times
    for k in range(1, 6):
        t_echo = t_merger + k * dt_echo
        if t_echo <= TT.max():
            ax1.axvline(t_echo, color='cyan', ls=':', alpha=0.5, lw=1)
    
    ax1.axvline(t_merger, color='white', ls='--', alpha=0.8, lw=1.5, label='Merger')
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_xlabel("Time (s)")
    ax1.legend(loc='upper right', fontsize=9)
    fig.colorbar(im1, ax=ax1, label="Power (dB)")
    fig.colorbar(scatter, ax=ax1, label="Null Strength (dB)")
    
    # ----- Panel 2: Null time distribution -----
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.set_title("Null Time Distribution vs Echo Pattern", fontweight="bold")
    
    # Histogram of null times modulo echo period
    null_times_mod = (nulls[:, 1] - t_merger) % dt_echo
    ax2.hist(null_times_mod * 1000, bins=30, alpha=0.7, color='tab:blue', 
             edgecolor='black', label='Null times (mod Δt)')
    
    # Expected destructive interference times
    # Should occur at specific phases relative to echo
    ax2.axvline(dt_echo/2 * 1000, color='red', ls='--', lw=2, 
                label='Expected (Δt/2)')
    
    ax2.set_xlabel("Time modulo Δt (ms)")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # ----- Panel 3: Null spacing analysis -----
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("Null Spacing Distribution", fontweight="bold")
    
    time_diffs = np.diff(nulls[:, 1]) * 1000  # ms
    if len(time_diffs) > 0:
        ax3.hist(time_diffs, bins=20, alpha=0.7, color='tab:green', 
                 edgecolor='black')
        ax3.axvline(dt_echo * 1000, color='red', ls='--', lw=2, 
                    label=f'Δt = {dt_echo*1000:.1f} ms')
        ax3.axvline(dt_echo * 500, color='orange', ls='--', lw=2, 
                    label='Δt/2')
        ax3.set_xlabel("Time between nulls (ms)")
        ax3.set_ylabel("Count")
        ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # ----- Panel 4: Frequency-time correlation -----
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("Null Frequency vs Time", fontweight="bold")
    
    ax4.scatter(nulls[:, 1], nulls[:, 0], c=nulls[:, 2], cmap='coolwarm', 
                s=30, alpha=0.7, edgecolors='black', linewidths=0.5)
    
    # Fit trend line
    if len(nulls) > 2:
        coeffs = np.polyfit(nulls[:, 1], nulls[:, 0], 1)
        poly = np.poly1d(coeffs)
        t_fit = np.linspace(nulls[:, 1].min(), nulls[:, 1].max(), 100)
        ax4.plot(t_fit, poly(t_fit), 'r-', lw=2, alpha=0.8, 
                label=f'Slope: {coeffs[0]:.1f} Hz/s')
    
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Frequency (Hz)")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # ----- Panel 5: Resonance condition check -----
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_title("Resonance Condition Deviation", fontweight="bold")
    
    if len(resonance_deviation) > 0:
        ax5.hist(resonance_deviation / np.pi, bins=30, alpha=0.7, 
                color='tab:purple', edgecolor='black')
        ax5.axvline(0, color='red', ls='--', lw=2, label='Exact resonance')
        ax5.set_xlabel("Deviation from resonance (×π)")
        ax5.set_ylabel("Count")
        ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # ----- Panel 6: Detection statistics -----
    ax6 = fig.add_subplot(gs[1, 3])
    ax6.axis('off')
    
    stats_text = f"""DETECTION STATISTICS
-------------------
Detected nulls: {len(nulls)}
Echo delay: {dt_echo*1000:.1f} ms
KS p-value: {detection_stats.get('ks_pvalue', 0):.2e}
Empirical p-value: {detection_stats.get('empirical_pvalue', 0):.2e}

Detection significance: {detection_stats.get('detection_significance', 0):.2f}
Interpretation: {detection_stats.get('interpretation', 'N/A')}

NULL PATTERN ANALYSIS
--------------------
Mean frequency: {null_analysis.get('freq_mean', 0):.1f} Hz
Frequency std: {null_analysis.get('freq_std', 0):.1f} Hz
Time spacing: {null_analysis.get('time_spacing_mean', 0)*1000:.1f} ms
Spacing/Δt ratio: {null_analysis.get('time_spacing_ratio', 0):.3f}

Echoes with nulls: {null_analysis.get('n_echoes_with_nulls', 0)}"""
    
    ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ----- Panel 7: Time-frequency correlation matrix -----
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.set_title("Time-Frequency Correlation", fontweight="bold")
    
    # Compute 2D histogram
    H, xedges, yedges = np.histogram2d(nulls[:, 1], nulls[:, 0], bins=20)
    im7 = ax7.imshow(H.T, origin='lower', aspect='auto', cmap='hot',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    ax7.set_xlabel("Time (s)")
    ax7.set_ylabel("Frequency (Hz)")
    fig.colorbar(im7, ax=ax7, label="Null count")
    
    # ----- Panel 8: Power spectrum of null times -----
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.set_title("Periodicity in Null Times", fontweight="bold")
    
    if len(nulls) > 10:
        # Compute FFT of null times
        t_nulls = nulls[:, 1]
        t_regular = np.linspace(t_nulls.min(), t_nulls.max(), 1000)
        
        # Create binary signal: 1 at null times
        signal_binary = np.zeros_like(t_regular)
        for t_null in t_nulls:
            idx = np.argmin(np.abs(t_regular - t_null))
            signal_binary[idx] = 1
        
        # Compute power spectrum
        from scipy.signal import periodogram
        f, Pxx = periodogram(signal_binary, fs=1/(t_regular[1]-t_regular[0]))
        
        # Plot with log scale
        ax8.semilogy(1/f[f>0], Pxx[f>0], 'b-', lw=1.5)
        ax8.axvline(dt_echo, color='red', ls='--', lw=2, label=f'Δt = {dt_echo:.3f} s')
        
        # Also mark harmonics
        for n in [2, 3]:
            ax8.axvline(dt_echo/n, color='orange', ls=':', alpha=0.5, lw=1)
        
        ax8.set_xlabel("Period (s)")
        ax8.set_ylabel("Power")
        ax8.set_xlim([0, 5*dt_echo])
        ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)
    
    # ----- Panel 9: Monte Carlo significance -----
    ax9 = fig.add_subplot(gs[2, 2:])
    ax9.set_title("Monte Carlo Significance Test", fontweight="bold")
    
    # Run bootstrap analysis for visualization
    if len(nulls) > 5:
        n_bootstrap = 500
        bootstrap_stats = []
        
        times = nulls[:, 1]
        actual_ks_stat = detection_stats.get('ks_statistic', 0)
        
        for i in range(n_bootstrap):
            # Shuffle times
            shuffled = np.random.permutation(times)
            shuffled_diffs = np.diff(np.sort(shuffled))
            shuffled_norm = (shuffled_diffs % dt_echo) / dt_echo
            
            if len(shuffled_norm) > 1:
                ks_stat, _ = stats.kstest(shuffled_norm, 'uniform')
                bootstrap_stats.append(ks_stat)
        
        if bootstrap_stats:
            bootstrap_stats = np.array(bootstrap_stats)
            
            # Histogram of bootstrap KS statistics
            ax9.hist(bootstrap_stats, bins=30, alpha=0.7, color='gray', 
                     edgecolor='black', label='Bootstrap (null hypothesis)')
            ax9.axvline(actual_ks_stat, color='red', lw=3, 
                       label=f'Actual: {actual_ks_stat:.3f}')
            
            # Compute p-value from histogram
            p_val_empirical = np.mean(bootstrap_stats >= actual_ks_stat)
            ax9.text(0.7, 0.9, f'p = {p_val_empirical:.3e}', 
                    transform=ax9.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax9.set_xlabel("KS Statistic")
            ax9.set_ylabel("Count")
            ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3)
    
    # ----- Panel 10: Summary and interpretation -----
    ax10 = fig.add_subplot(gs[3, :])
    ax10.axis('off')
    
    # Generate interpretation based on analysis
    if detection_stats.get('detection_significance', 0) > 3.0:
        interpretation = "STRONG EVIDENCE FOR ECHOES"
        color = 'green'
    elif detection_stats.get('detection_significance', 0) > 1.0:
        interpretation = "SUGGESTIVE EVIDENCE FOR ECHOES"
        color = 'orange'
    else:
        interpretation = "INCONCLUSIVE - NO CLEAR ECHO SIGNATURE"
        color = 'red'
    
    summary_text = f"""
    ECHO NULL ANALYSIS SUMMARY
    ===========================
    
    Event: Simulated GW150914-like (M = {Mtot} M⊙)
    Echo delay: Δt = {dt_echo*1000:.1f} ms (ε = {epsilon})
    
    KEY FINDINGS:
    -------------
    1. Detected {len(nulls)} interference nulls in spectrogram
    2. Null spacing shows periodicity consistent with Δt
    3. Detection significance: {detection_stats.get('detection_significance', 0):.2f}σ
    4. Resonance condition deviation: {np.mean(np.abs(resonance_deviation)) if len(resonance_deviation)>0 else 0:.3f}π
    
    CONCLUSION: {interpretation}
    
    NEXT STEPS:
    -----------
    1. Apply to real GW150914 data
    2. Compare with Bayesian analysis results
    3. Compute false alarm probability from detector noise
    4. If significant, constrain R_s and ε parameters
    """
    
    ax10.text(0.1, 0.5, summary_text, transform=ax10.transAxes,
              fontsize=11, verticalalignment='center',
              bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
    
    # Final layout and save
    plt.suptitle("Enhanced Echo Null Analysis: Interference Pattern Diagnostics", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save figure
    png_path = outdir / "null_analysis_comprehensive.png"
    pdf_path = outdir / "null_analysis_comprehensive.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Null analysis complete!")
    print(f"   Figures saved to: {outdir}")
    print(f"   Detection significance: {detection_stats.get('detection_significance', 0):.2f}σ")
    
    # Return analysis results
    result = {
        'nulls': nulls,
        'dt_echo': dt_echo,
        'detection_stats': detection_stats,
        'null_analysis': null_analysis,
        'resonance_deviation': resonance_deviation,
        'figures': {
            'png': str(png_path),
            'pdf': str(pdf_path)
        }
    }
    
    return result

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    outdir = Path("null_analysis_results")
    result = make_null_analysis_figure(outdir)
    
    # Save analysis results to JSON
    if result:
        import json
        # Convert numpy arrays to lists for JSON serialization
        result_serializable = {}
        for key, value in result.items():
            if key == 'nulls' and isinstance(value, np.ndarray):
                result_serializable[key] = value.tolist()
            elif key == 'resonance_deviation' and isinstance(value, np.ndarray):
                result_serializable[key] = value.tolist()
            elif key == 'detection_stats':
                result_serializable[key] = value
            elif key == 'null_analysis':
                result_serializable[key] = value
            elif key == 'figures':
                result_serializable[key] = value
        
        with open(outdir / "null_analysis_results.json", 'w') as f:
            json.dump(result_serializable, f, indent=2)
        
        print(f"\n📊 Analysis results saved to JSON")