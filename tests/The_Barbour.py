
# ═══════════════════════════════════════════════════════════════
# PART 2: CORRECTED PIPELINE
# ═══════════════════════════════════════════════════════════════

The key changes:
1. WHITEN before computing any statistic
2. Use ON-COMB vs OFF-COMB differential statistic
3. Background from frequency-shifted combs (not shuffles)
4. Require H1-L1 phase consistency
5. Injection-recovery validation built in
""")
═══════════════════════════════════════════════════════════════
# DEMONSTRATION: Correct vs Incorrect Methods on Synthetic Data
# ═══════════════════════════════════════════════════════════════

def simulate_ligo_data(n_freq=8192, f_min=10, f_max=1000, 
                       inject_comb=False, comb_snr=0.0,
                       M_det=63.0, seed=None):
    """Generate realistic LIGO-like data with optional comb injection"""
    if seed is not None:
        np.random.seed(seed)
    
    freqs = np.linspace(f_min, f_max, n_freq)
    df = freqs[1] - freqs[0]
    
    # Realistic aLIGO PSD shape
    def aligo_psd(f):
        return 1e-46 * ((20/f)**10 + 1.0 + (f/500)**2)
    
    psd = aligo_psd(freqs)
    
    # Generate noise
    h1_noise = np.sqrt(psd) * (np.random.randn(n_freq) + 1j * np.random.randn(n_freq)) / np.sqrt(2)
    l1_noise = np.sqrt(psd) * (np.random.randn(n_freq) + 1j * np.random.randn(n_freq)) / np.sqrt(2)
    
    # Comb parameters
    delta_f_comb = 549.5 / M_det
    
    # Inject comb signal if requested
    signal = np.zeros(n_freq, dtype=complex)
    if inject_comb and comb_snr > 0:
        tooth_freqs = np.arange(f_min, f_max, delta_f_comb)
        for n, f_tooth in enumerate(tooth_freqs):
            idx = np.argmin(np.abs(freqs - f_tooth))
            if 0 <= idx < n_freq:
                # Parity-odd: alternating sign
                sign = (-1)**n
                amplitude = comb_snr * np.sqrt(psd[idx]) * sign
                phase = np.random.uniform(0, 2*np.pi)
                signal[idx] = amplitude * np.exp(1j * phase)
    
    h1 = h1_noise + signal
    l1 = l1_noise + signal  # Same signal in both detectors
    
    return freqs, h1, l1, psd, signal, delta_f_comb


def method_WRONG_raw_psd_correlation(freqs, h1, l1, psd, delta_f_comb, f_min=10, f_max=1000):
    """
    THE WRONG METHOD: Pearson correlation of raw |h|² at comb teeth.
    This is what merlin_correct_analysis.py does.
    """
    # Get comb tooth indices
    tooth_freqs = np.arange(f_min, f_max, delta_f_comb)
    tooth_idx = [np.argmin(np.abs(freqs - tf)) for tf in tooth_freqs]
    
    h1_power = np.abs(h1[tooth_idx])**2
    l1_power = np.abs(l1[tooth_idx])**2
    
    rho, _ = stats.pearsonr(h1_power, l1_power)
    
    # Shuffle null
    null_rhos = []
    for _ in range(500):
        h1_s = h1_power.copy()
        np.random.shuffle(h1_s)
        r, _ = stats.pearsonr(h1_s, l1_power)
        null_rhos.append(r)
    
    null_mean = np.mean(null_rhos)
    null_std = np.std(null_rhos)
    sigma = (rho - null_mean) / null_std if null_std > 0 else 0
    
    return rho, sigma, 'RAW PSD + shuffle null'


def method_CORRECT_whitened_differential(freqs, h1, l1, psd, delta_f_comb, 
                                          f_min=10, f_max=1000):
    """
    THE CORRECT METHOD:
    1. Whiten (divide by PSD)
    2. Compute cross-spectral density at comb teeth
    3. Compare to cross-spectral density at off-comb frequencies
    4. Background from frequency-shifted combs
    """
    df = freqs[1] - freqs[0]
    
    # Step 1: Whiten
    h1_white = h1 / np.sqrt(psd)
    l1_white = l1 / np.sqrt(psd)
    
    # Step 2: Cross-spectral density
    csd = h1_white * np.conj(l1_white)
    
    # Step 3: On-comb statistic
    tooth_freqs = np.arange(f_min, f_max, delta_f_comb)
    tooth_idx = [np.argmin(np.abs(freqs - tf)) for tf in tooth_freqs]
    tooth_idx = [i for i in tooth_idx if 0 <= i < len(freqs)]
    
    # Use a few bins around each tooth for robustness
    half_width = max(1, int(0.5 / df))  # ±0.5 Hz
    
    on_comb_csd = []
    for idx in tooth_idx:
        lo = max(0, idx - half_width)
        hi = min(len(freqs), idx + half_width + 1)
        on_comb_csd.append(np.mean(np.real(csd[lo:hi])))
    
    on_comb_stat = np.mean(on_comb_csd)
    
    # Step 4: Off-comb statistic (shifted by Δf/2)
    off_tooth_freqs = np.arange(f_min + delta_f_comb/2, f_max, delta_f_comb)
    off_tooth_idx = [np.argmin(np.abs(freqs - tf)) for tf in off_tooth_freqs]
    off_tooth_idx = [i for i in off_tooth_idx if 0 <= i < len(freqs)]
    
    off_comb_csd = []
    for idx in off_tooth_idx:
        lo = max(0, idx - half_width)
        hi = min(len(freqs), idx + half_width + 1)
        off_comb_csd.append(np.mean(np.real(csd[lo:hi])))
    
    off_comb_stat = np.mean(off_comb_csd)
    
    # DIFFERENTIAL statistic
    diff_stat = on_comb_stat - off_comb_stat
    
    # Step 5: Background from frequency-shifted combs
    n_shifts = 200
    shift_stats = []
    for _ in range(n_shifts):
        # Random frequency offset
        offset = np.random.uniform(0, delta_f_comb)
        shifted_freqs = np.arange(f_min + offset, f_max, delta_f_comb)
        shifted_idx = [np.argmin(np.abs(freqs - tf)) for tf in shifted_freqs]
        shifted_idx = [i for i in shifted_idx if 0 <= i < len(freqs)]
        
        shifted_csd = []
        for idx in shifted_idx:
            lo = max(0, idx - half_width)
            hi = min(len(freqs), idx + half_width + 1)
            shifted_csd.append(np.mean(np.real(csd[lo:hi])))
        
        # Also compute off-comb for this shift
        off_shifted_freqs = np.arange(f_min + offset + delta_f_comb/2, f_max, delta_f_comb)
        off_shifted_idx = [np.argmin(np.abs(freqs - tf)) for tf in off_shifted_freqs]
        off_shifted_idx = [i for i in off_shifted_idx if 0 <= i < len(freqs)]
        
        off_shifted_csd = []
        for idx in off_shifted_idx:
            lo = max(0, idx - half_width)
            hi = min(len(freqs), idx + half_width + 1)
            off_shifted_csd.append(np.mean(np.real(csd[lo:hi])))
        
        if shifted_csd and off_shifted_csd:
            shift_stats.append(np.mean(shifted_csd) - np.mean(off_shifted_csd))
    
    bg_mean = np.mean(shift_stats)
    bg_std = np.std(shift_stats)
    sigma = (diff_stat - bg_mean) / bg_std if bg_std > 0 else 0
    
    return diff_stat, sigma, 'Whitened differential + freq-shift background'


def method_CORRECT_phase_consistency(freqs, h1, l1, psd, delta_f_comb,
                                      f_min=10, f_max=1000):
    """
    PHASE CONSISTENCY CHECK:
    If a real signal exists at comb teeth, the cross-spectral PHASE
    should be consistent between H1 and L1 at those frequencies
    (after accounting for light travel time).
    
    Test: Rayleigh statistic of cross-spectral phase at comb teeth
    vs at random frequencies.
    """
    df = freqs[1] - freqs[0]
    
    # Whiten
    h1_white = h1 / np.sqrt(psd)
    l1_white = l1 / np.sqrt(psd)
    
    # Cross-spectral phase at comb teeth
    csd = h1_white * np.conj(l1_white)
    
    tooth_freqs = np.arange(f_min, f_max, delta_f_comb)
    tooth_idx = [np.argmin(np.abs(freqs - tf)) for tf in tooth_freqs]
    tooth_idx = [i for i in tooth_idx if 0 <= i < len(freqs)]
    
    phases_at_teeth = np.angle(csd[tooth_idx])
    
    # Rayleigh statistic: R = |mean(e^(iφ))|
    # For random phases: E[R] ≈ √(π/4N), for coherent: R → 1
    R_teeth = np.abs(np.mean(np.exp(1j * phases_at_teeth)))
    N = len(tooth_idx)
    
    # Expected R for random phases
    R_expected = np.sqrt(np.pi / (4 * N))
    R_std = np.sqrt((4 - np.pi) / (4 * N))
    
    sigma = (R_teeth - R_expected) / R_std if R_std > 0 else 0
    
    return R_teeth, sigma, f'Phase consistency (Rayleigh, N={N} teeth)'


# ═══════════════════════════════════════════════════════════════
# RUN COMPARISON
# ═══════════════════════════════════════════════════════════════

print("\n" + "─" * 78)
print("COMPARISON: Wrong vs Correct Methods")
print("─" * 78)

# Test scenarios
scenarios = [
    ("No signal (noise only)", False, 0.0),
    ("Weak comb (SNR/tooth=0.3)", True, 0.3),
    ("Medium comb (SNR/tooth=1.0)", True, 1.0),
    ("Strong comb (SNR/tooth=3.0)", True, 3.0),
]

n_trials = 50
M_det = 63.0

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for ax_idx, (scenario_name, inject, snr_tooth) in enumerate(scenarios):
    ax = axes[ax_idx // 2, ax_idx % 2]
    
    wrong_sigmas = []
    correct_diff_sigmas = []
    correct_phase_sigmas = []
    
    for trial in range(n_trials):
        freqs, h1, l1, psd, signal, delta_f = simulate_ligo_data(
            inject_comb=inject, comb_snr=snr_tooth, M_det=M_det, seed=trial)
        
        _, sig_wrong, _ = method_WRONG_raw_psd_correlation(
            freqs, h1, l1, psd, delta_f)
        _, sig_diff, _ = method_CORRECT_whitened_differential(
            freqs, h1, l1, psd, delta_f)
        _, sig_phase, _ = method_CORRECT_phase_consistency(
            freqs, h1, l1, psd, delta_f)
        
        wrong_sigmas.append(sig_wrong)
        correct_diff_sigmas.append(sig_diff)
        correct_phase_sigmas.append(sig_phase)
    
    # Plot distributions
    bins = np.linspace(-5, max(25, np.max(wrong_sigmas) + 2), 40)
    
    ax.hist(wrong_sigmas, bins=bins, alpha=0.4, color='red', 
            label=f'WRONG (μ={np.mean(wrong_sigmas):.1f}σ)', density=True)
    ax.hist(correct_diff_sigmas, bins=bins, alpha=0.4, color='green',
            label=f'Correct-Diff (μ={np.mean(correct_diff_sigmas):.1f}σ)', density=True)
    ax.hist(correct_phase_sigmas, bins=bins, alpha=0.4, color='blue',
            label=f'Correct-Phase (μ={np.mean(correct_phase_sigmas):.1f}σ)', density=True)
    
    ax.axvline(3, color='orange', linestyle='--', alpha=0.7, label='3σ')
    ax.axvline(5, color='red', linestyle='--', alpha=0.7, label='5σ')
    ax.axvline(0, color='black', linewidth=1)
    
    ax.set_xlabel('Significance (σ)')
    ax.set_ylabel('Density')
    ax.set_title(f'{scenario_name}', fontweight='bold', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Print summary
    print(f"\n{scenario_name}:")
    print(f"  WRONG method:    mean = {np.mean(wrong_sigmas):+6.1f}σ  "
          f"(>{3}σ: {100*np.mean(np.array(wrong_sigmas)>3):.0f}%)")
    print(f"  Correct (diff):  mean = {np.mean(correct_diff_sigmas):+6.1f}σ  "
          f"(>{3}σ: {100*np.mean(np.array(correct_diff_sigmas)>3):.0f}%)")
    print(f"  Correct (phase): mean = {np.mean(correct_phase_sigmas):+6.1f}σ  "
          f"(>{3}σ: {100*np.mean(np.array(correct_phase_sigmas)>3):.0f}%)")

fig.suptitle('Method Comparison: Wrong vs Correct Significance Estimation\n'
             f'{n_trials} trials per scenario, M_det = {M_det} M☉',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/method_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved: method_comparison.png")


# ═══════════════════════════════════════════════════════════════
# DETECTION POWER: HOW STRONG MUST THE SIGNAL BE?
# ═══════════════════════════════════════════════════════════════

print("\n" + "─" * 78)
print("DETECTION THRESHOLD: Minimum comb SNR for 3σ/5σ")
print("─" * 78)

snr_values = np.arange(0, 5.1, 0.5)
n_trials_power = 30

mean_sigmas_diff = []
mean_sigmas_phase = []

for snr_tooth in snr_values:
    diff_sigs = []
    phase_sigs = []
    
    for trial in range(n_trials_power):
        freqs, h1, l1, psd, signal, delta_f = simulate_ligo_data(
            inject_comb=(snr_tooth > 0), comb_snr=snr_tooth, M_det=M_det, 
            seed=1000+trial)
        
        _, sig_d, _ = method_CORRECT_whitened_differential(
            freqs, h1, l1, psd, delta_f)
        _, sig_p, _ = method_CORRECT_phase_consistency(
            freqs, h1, l1, psd, delta_f)
        
        diff_sigs.append(sig_d)
        phase_sigs.append(sig_p)
    
    mean_sigmas_diff.append(np.mean(diff_sigs))
    mean_sigmas_phase.append(np.mean(phase_sigs))

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(snr_values, mean_sigmas_diff, 'go-', linewidth=2, markersize=8,
        label='Whitened Differential')
ax.plot(snr_values, mean_sigmas_phase, 'bo-', linewidth=2, markersize=8,
        label='Phase Consistency')
ax.axhline(3, color='orange', linestyle='--', label='3σ threshold')
ax.axhline(5, color='red', linestyle='--', label='5σ threshold')
ax.axhline(0, color='gray', linestyle=':')
ax.set_xlabel('Injected SNR per comb tooth', fontsize=12)
ax.set_ylabel('Recovered significance (σ)', fontsize=12)
ax.set_title('Detection Power Curve\n'
             'How strong must the comb signal be for detection?',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/detection_power_curve.png', dpi=150, bbox_inches='tight')
print("\nSaved: detection_power_curve.png")

# Print threshold
for name, sigmas in [("Differential", mean_sigmas_diff), ("Phase", mean_sigmas_phase)]:
    for threshold in [3, 5]:
        above = np.array(sigmas) >= threshold
        if np.any(above):
            idx = np.where(above)[0][0]
            print(f"  {name}: {threshold}σ requires SNR/tooth ≥ {snr_values[idx]:.1f}")
        else:
            print(f"  {name}: {threshold}σ not reached in tested range")

