#!/usr/bin/env python3
"""
Echo Detection System - Quick Runner
Generates LIGO data, visualizations, and nanograph data
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
import json
import os

def generate_ligo_data():
    """Generate realistic LIGO data."""
    print("\n1️⃣  Generating LIGO data...")
    
    freq = np.logspace(np.log10(10), np.log10(2000), 1000)
    
    # LIGO noise PSD
    def ligo_noise_psd(f):
        f0, S0 = 215.0, 1e-49
        seismic = (f / 10)**(-4)
        thermal = 1.0
        shot_noise = (f / 500)**2
        return S0 * (seismic + thermal + shot_noise)
    
    psd = ligo_noise_psd(freq)
    
    # Waveform
    m1, m2, distance = 30, 30, 400e6  # masses in M_sun, distance in meters * 3.086e22
    chirp_mass = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
    h_char = (chirp_mass**(5/6) / distance) / (np.pi**(2/3) * freq**(7/6))
    
    # Echo component
    r_s = 2 * (m1 + m2) * 1.477  # Schwarzschild radius in km
    omega = 2 * np.pi * freq
    k_r = omega * r_s / 3e5
    R_infinity = np.exp(-k_r) * np.sin(k_r) / (1 + k_r**2)
    h_echo = h_char * R_infinity * 0.1
    
    h_total = h_char + h_echo
    
    # SNR
    from scipy.integrate import trapezoid
    snr = np.sqrt(4 * trapezoid(np.abs(h_total)**2 / psd, freq))
    
    print(f"   ✓ SNR = {snr:.2f}")
    print(f"   ✓ Detection: {'YES' if snr > 8 else 'NO'}")
    
    return {
        'frequency': freq,
        'psd': psd,
        'strain': h_total,
        'echo': h_echo,
        'template': h_char,
        'snr': snr
    }

def generate_visuals(data, output_dir='outputs'):
    """Generate visualization plots."""
    print("\n2️⃣  Generating visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Main plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Strain
    ax = axes[0, 0]
    ax.loglog(data['frequency'], np.abs(data['strain']), 'b-', label='Total', lw=2)
    ax.loglog(data['frequency'], np.abs(data['template']), 'g--', label='Template', lw=1.5)
    ax.loglog(data['frequency'], np.abs(data['echo']), 'r:', label='Echo', lw=1.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Strain')
    ax.set_title('GW Signal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Noise
    ax = axes[0, 1]
    ax.loglog(data['frequency'], np.sqrt(data['psd']), 'k-', lw=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Noise (1/√Hz)')
    ax.set_title('LIGO Sensitivity')
    ax.grid(True, alpha=0.3)
    
    # SNR accumulation
    ax = axes[1, 0]
    snr_cum = np.sqrt(4 * cumulative_trapezoid(
        np.abs(data['strain'])**2 / data['psd'], 
        data['frequency'], 
        initial=0
    ))
    ax.semilogx(data['frequency'], snr_cum, 'purple', lw=2)
    ax.axhline(8, color='red', ls='--', label='Threshold')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Cumulative SNR')
    ax.set_title(f'SNR Build-up (Total={data["snr"]:.1f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Echo fraction
    ax = axes[1, 1]
    echo_frac = 100 * np.abs(data['echo']) / np.abs(data['strain'])
    ax.semilogx(data['frequency'], echo_frac, 'orange', lw=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Echo (%)')
    ax.set_title('Echo Contribution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = f'{output_dir}/ligo_analysis.png'
    plt.savefig(path, dpi=150)
    print(f"   ✓ Saved: {path}")
    plt.close()
    
    return path

def generate_nanograph(data, output_dir='outputs'):
    """Generate nanograph data for WebGL."""
    print("\n3️⃣  Generating nanograph data...")
    
    freq = data['frequency'][::10]
    strain = np.abs(data['strain'][::10])
    
    nodes = []
    edges = []
    
    for i, (f, h) in enumerate(zip(freq, strain)):
        nodes.append({
            'id': i,
            'freq': float(f),
            'amp': float(h),
            'x': float(np.log10(f)),
            'y': float(np.log10(h)),
            'z': 0.0
        })
    
    for i in range(len(nodes) - 1):
        edges.append({'source': i, 'target': i+1, 'weight': 1.0})
    
    nanograph = {
        'nodes': nodes,
        'edges': edges,
        'metadata': {
            'total_nodes': len(nodes),
            'snr': float(data['snr'])
        }
    }
    
    path = f'{output_dir}/nanograph.json'
    with open(path, 'w') as f:
        json.dump(nanograph, f, indent=2)
    
    print(f"   ✓ {len(nodes)} nodes, {len(edges)} edges")
    print(f"   ✓ Saved: {path}")
    
    return nanograph

def main():
    print("="*60)
    print("🌊  ECHO DETECTION SYSTEM")
    print("="*60)
    
    # Generate data
    data = generate_ligo_data()
    
    # Generate visuals
    visual_path = generate_visuals(data)
    
    # Generate nanograph
    nano = generate_nanograph(data)
    
    print("\n" + "="*60)
    print("✅  COMPLETE!")
    print("="*60)
    print(f"\n📊  Results:")
    print(f"   • SNR: {data['snr']:.2f}")
    print(f"   • Significance: {data['snr']/np.sqrt(2):.1f}σ")
    print(f"   • Nanograph nodes: {len(nano['nodes'])}")
    print(f"\n📁  Outputs: /app/echo-detection-system/outputs/")
    print("   • ligo_analysis.png")
    print("   • nanograph.json")
    print("\n" + "="*60)

if __name__ == '__main__':
    main()
