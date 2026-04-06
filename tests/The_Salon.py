#!/usr/bin/env python3
"""
Merlin Whitened vs Non-Whitened Comparison
==========================================
Critical systematic check: Compare PSD correlation results using
whitened vs non-whitened data to determine if high correlations
are real signals or instrumental artifacts.

If correlations drop dramatically with whitening → instrumental artifact
If correlations remain high → validates real signal
"""

import numpy as np
from scipy import signal
from scipy.stats import pearsonr
import os
import warnings
import urllib.request
import json
warnings.filterwarnings('ignore')

# Set LIGO data server
os.environ['LIGO_DATAFIND_SERVER'] = 'datafind.ligo.org:443'

from gwpy.timeseries import TimeSeries

# =============================================================================
# CONFIGURATION
# =============================================================================
REFERENCE_MASS = 30.0       # M☉ reference mass
REFERENCE_DELAY = 54.6e-3   # seconds (round-trip time for 30 M☉)
FREQ_MIN = 10.0             # Hz
FREQ_MAX = 1000.0           # Hz
Q_FACTOR = 26
LINEWIDTH = 0.58            # Hz
DATA_DURATION = 32          # seconds around merger
SAMPLE_RATE = 4096          # Hz

# 8 specific events to analyze
EVENTS = [
    {'name': 'GW231028', 'gps': 1382475706.0, 'M_detector': 27.0},
    {'name': 'GW231226', 'gps': 1387839620.0, 'M_detector': 55.0},
    {'name': 'GW250114', 'gps': 1420878141.2365, 'M_detector': 80.0},
    {'name': 'GW231215', 'gps': 1386799755.0, 'M_detector': 63.0},
    {'name': 'GW231123', 'gps': 1384714055.0, 'M_detector': 57.0},
    {'name': 'GW230927', 'gps': 1379721467.0, 'M_detector': 45.0},
    {'name': 'GW231206', 'gps': 1386088806.0, 'M_detector': 52.0},
    {'name': 'GW230627', 'gps': 1371848400.0, 'M_detector': 38.0},
]


def compute_comb_spacing(M_detector):
    """Compute event-specific comb spacing from detector-frame total mass."""
    delta_t = REFERENCE_DELAY * (M_detector / REFERENCE_MASS)
    delta_f = 1.0 / delta_t
    return delta_f, delta_t


def fetch_strain_data(gps_time, detector, duration=DATA_DURATION):
    """Fetch strain data from GWOSC"""
    t_start = gps_time - duration / 2
    t_end = gps_time + duration / 2
    
    try:
        data = TimeSeries.fetch_open_data(detector, t_start, t_end, 
                                          sample_rate=SAMPLE_RATE,
                                          cache=True)
        return data
    except Exception as e:
        print(f"    Warning: Could not fetch {detector} data: {e}")
        return None


def whiten_and_bandpass(data, fmin=FREQ_MIN, fmax=FREQ_MAX, whiten=True):
    """Whiten data and apply bandpass filter"""
    try:
        if whiten:
            processed = data.whiten(fftlength=4, overlap=2)
        else:
            processed = data
        filtered = processed.bandpass(fmin, fmax)
        return filtered
    except Exception as e:
        return None


def compute_fft(data):
    """Compute FFT and return frequencies and complex amplitudes"""
    n = len(data.value)
    dt = 1.0 / data.sample_rate.value
    fft_vals = np.fft.rfft(data.value)
    freqs = np.fft.rfftfreq(n, dt)
    return freqs, fft_vals


def get_comb_frequencies(f_spacing, f_min, f_max):
    """Generate comb frequencies within range"""
    n_min = int(np.ceil(f_min / f_spacing))
    n_max = int(np.floor(f_max / f_spacing))
    comb_freqs = np.array([n * f_spacing for n in range(n_min, n_max + 1)])
    return comb_freqs


def extract_comb_amplitudes(freqs, fft_vals, comb_freqs, linewidth=LINEWIDTH):
    """Extract complex amplitudes at comb frequencies"""
    amplitudes = []
    snrs = []
    
    df = freqs[1] - freqs[0]
    half_width = max(1, int(linewidth / (2 * df)))
    noise_width = max(10, int(5 * linewidth / df))
    
    for f_comb in comb_freqs:
        idx = np.argmin(np.abs(freqs - f_comb))
        
        if idx < noise_width or idx >= len(fft_vals) - noise_width:
            amplitudes.append(np.nan + 1j*np.nan)
            snrs.append(np.nan)
            continue
        
        idx_min = max(0, idx - half_width)
        idx_max = min(len(fft_vals), idx + half_width + 1)
        amp = np.mean(fft_vals[idx_min:idx_max])
        
        noise_left = fft_vals[idx - noise_width:idx - half_width - 1]
        noise_right = fft_vals[idx + half_width + 1:idx + noise_width]
        noise_samples = np.concatenate([noise_left, noise_right])
        noise_std = np.std(np.abs(noise_samples))
        
        snr = np.abs(amp) / noise_std if noise_std > 0 else 0
        
        amplitudes.append(amp)
        snrs.append(snr)
    
    return np.array(amplitudes), np.array(snrs)


def analyze_with_whitening_option(event_info, use_whitening):
    """Analyze event with specified whitening option"""
    event_name = event_info['name']
    gps_time = event_info['gps']
    M_detector = event_info['M_detector']
    
    # Compute event-specific comb spacing
    delta_f, delta_t = compute_comb_spacing(M_detector)
    
    # Fetch data
    h1_data = fetch_strain_data(gps_time, 'H1')
    l1_data = fetch_strain_data(gps_time, 'L1')
    
    if h1_data is None or l1_data is None:
        return None
    
    # Process data with specified whitening
    h1_processed = whiten_and_bandpass(h1_data, whiten=use_whitening)
    l1_processed = whiten_and_bandpass(l1_data, whiten=use_whitening)
    
    if h1_processed is None or l1_processed is None:
        return None
    
    # FFT
    freqs_h1, fft_h1 = compute_fft(h1_processed)
    freqs_l1, fft_l1 = compute_fft(l1_processed)
    
    # Generate comb frequencies
    comb_freqs = get_comb_frequencies(delta_f, FREQ_MIN, FREQ_MAX)
    
    # Extract comb amplitudes
    amps_h1, snrs_h1 = extract_comb_amplitudes(freqs_h1, fft_h1, comb_freqs)
    amps_l1, snrs_l1 = extract_comb_amplitudes(freqs_l1, fft_l1, comb_freqs)
    
    # H1-L1 correlation at comb teeth
    valid = ~(np.isnan(amps_h1) | np.isnan(amps_l1))
    n_valid = np.sum(valid)
    
    if n_valid < 5:
        return None
    
    h1_psd = np.abs(amps_h1[valid])**2
    l1_psd = np.abs(amps_l1[valid])**2
    rho, _ = pearsonr(h1_psd, l1_psd)
    
    # Null distribution (shuffle method)
    null_corrs = []
    for _ in range(500):
        h1_shuffled = h1_psd.copy()
        np.random.shuffle(h1_shuffled)
        r, _ = pearsonr(h1_shuffled, l1_psd)
        null_corrs.append(r)
    
    null_mean = np.mean(null_corrs)
    null_std = np.std(null_corrs)
    
    # Significance
    if null_std > 0:
        sigma = (rho - null_mean) / null_std
    else:
        sigma = np.nan
    
    return {
        'event': event_name,
        'gps': gps_time,
        'M_detector': M_detector,
        'delta_f': delta_f,
        'rho': rho,
        'sigma': sigma,
        'n_teeth': n_valid,
    }


def main():
    print("="*80)
    print("CRITICAL SYSTEMATIC CHECK: Whitened vs Non-Whitened PSD Correlation")
    print("="*80)
    print("\nPurpose: Determine if high H1-L1 correlations are real signals or")
    print("         instrumental artifacts from common noise shape.")
    print("\nExpected outcomes:")
    print("  - If correlations drop dramatically → confirms instrumental artifact")
    print("  - If correlations remain high → validates real signal")
    print("\n" + "="*80)
    
    results_nonwhitened = []
    results_whitened = []
    
    print(f"\nAnalyzing {len(EVENTS)} events...")
    print("-"*80)
    
    for event in EVENTS:
        print(f"\n>>> {event['name']} (M_det={event['M_detector']:.0f} M☉) ...")
        
        # Non-whitened analysis
        print("    [1] Non-whitened analysis...", end=" ", flush=True)
        result_nw = analyze_with_whitening_option(event, use_whitening=False)
        if result_nw:
            results_nonwhitened.append(result_nw)
            print(f"ρ={result_nw['rho']:.4f}, σ={result_nw['sigma']:.2f}")
        else:
            print("FAILED")
        
        # Whitened analysis
        print("    [2] Whitened analysis...", end=" ", flush=True)
        result_w = analyze_with_whitening_option(event, use_whitening=True)
        if result_w:
            results_whitened.append(result_w)
            print(f"ρ={result_w['rho']:.4f}, σ={result_w['sigma']:.2f}")
        else:
            print("FAILED")
    
    # Generate comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE: Non-Whitened vs Whitened PSD Correlation")
    print("="*80)
    print(f"\n{'Event':<12} {'M_det':>6} {'Old ρ':>8} {'Old σ':>8} {'New ρ':>8} {'New σ':>8} {'Change':>10}")
    print("-"*70)
    
    old_sigmas = []
    new_sigmas = []
    
    for nw in results_nonwhitened:
        # Find matching whitened result
        w = next((r for r in results_whitened if r['event'] == nw['event']), None)
        
        if w:
            change = w['sigma'] - nw['sigma']
            change_str = f"{change:+.2f}σ"
            
            print(f"{nw['event']:<12} {nw['M_detector']:>5.0f}  {nw['rho']:>7.4f}  {nw['sigma']:>7.2f}σ  {w['rho']:>7.4f}  {w['sigma']:>7.2f}σ  {change_str:>10}")
            
            old_sigmas.append(nw['sigma'])
            new_sigmas.append(w['sigma'])
        else:
            print(f"{nw['event']:<12} {nw['M_detector']:>5.0f}  {nw['rho']:>7.4f}  {nw['sigma']:>7.2f}σ  {'N/A':>8}  {'N/A':>8}  {'N/A':>10}")
            old_sigmas.append(nw['sigma'])
    
    # Combined significance
    print("\n" + "="*80)
    print("COMBINED SIGNIFICANCE")
    print("="*80)
    
    if old_sigmas:
        old_combined = np.sqrt(np.sum([s**2 for s in old_sigmas if not np.isnan(s)]))
        print(f"\nNon-whitened (OLD): {old_combined:.2f}σ combined ({len(old_sigmas)} events)")
    
    if new_sigmas:
        new_combined = np.sqrt(np.sum([s**2 for s in new_sigmas if not np.isnan(s)]))
        print(f"Whitened (NEW):     {new_combined:.2f}σ combined ({len(new_sigmas)} events)")
    
    if old_sigmas and new_sigmas:
        change_combined = new_combined - old_combined
        print(f"\nChange: {change_combined:+.2f}σ")
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    if new_sigmas and old_sigmas:
        avg_old = np.mean([s for s in old_sigmas if not np.isnan(s)])
        avg_new = np.mean([s for s in new_sigmas if not np.isnan(s)])
        reduction = (avg_old - avg_new) / avg_old * 100 if avg_old > 0 else 0
        
        print(f"\nAverage significance: {avg_old:.2f}σ → {avg_new:.2f}σ ({reduction:.0f}% reduction)")
        
        if new_combined < 3 and old_combined > 5:
            print("\n★★★ CONCLUSION: INSTRUMENTAL ARTIFACT CONFIRMED ★★★")
            print("The high correlations in non-whitened data were due to common")
            print("detector noise shape, not real signal coherence.")
        elif new_combined > 5:
            print("\n★★★ CONCLUSION: REAL SIGNAL VALIDATED ★★★")
            print("Correlations remain significant even after whitening,")
            print("suggesting true H1-L1 coherence at comb frequencies.")
        elif new_combined > 3:
            print("\n★★ CONCLUSION: MARGINAL EVIDENCE ★★")
            print("Some signal remains but greatly reduced.")
        else:
            print("\n★ CONCLUSION: NULL RESULT ★")
            print("No significant correlation with whitened data.")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
