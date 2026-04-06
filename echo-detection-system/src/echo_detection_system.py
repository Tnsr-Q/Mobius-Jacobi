"""
Echo Detection System - Complete Pipeline
Combines Tensorvibe, CausalEnforcer, and Quaternion for LIGO analysis
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid, solve_ivp
import matplotlib.pyplot as plt
from typing import Dict, Optional
import json

# Import the actual modules
import sys
sys.path.insert(0, '/app/echo-detection-system/src')
from causal_enforcer import CausalEnforcer
from quaternion import quaternion_to_matrix, random_unit_quaternion, generate_geometric_projections

class EchoDetectionSystem:
    """Complete echo detection and visualization system."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.enforcer = CausalEnforcer(
            kk_tolerance=self.config.get('kk_tolerance', 0.8),
            enforce_kk=True
        )
        self.results = {}
        
    @staticmethod
    def _default_config():
        return {
            'kk_tolerance': 0.8,
            'freq_range': (10, 2000),  # Hz
            'n_freq': 1000,
            'mass_range': (5, 100),  # Solar masses
            'distance': 400,  # Mpc
            'snr_threshold': 8.0,
        }
    
    def generate_ligo_data(self):
        """Generate realistic LIGO data with noise curves."""
        freq = np.logspace(
            np.log10(self.config['freq_range'][0]),
            np.log10(self.config['freq_range'][1]),
            self.config['n_freq']
        )
        
        # LIGO noise curve (Advanced LIGO design sensitivity)
        def ligo_noise_psd(f):
            """LIGO noise power spectral density"""
            # Simplified analytical fit
            f0 = 215.0  # Hz
            S0 = 1e-49  # Strain^2/Hz at f0
            
            # Low frequency: seismic noise
            seismic = (f / 10)**(-4)
            
            # Mid frequency: thermal noise
            thermal = 1.0
            
            # High frequency: shot noise
            shot_noise = (f / 500)**2
            
            return S0 * (seismic + thermal + shot_noise)
        
        psd = ligo_noise_psd(freq)
        
        # Generate template waveform
        m1, m2 = 30, 30  # Solar masses
        chirp_mass = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
        
        # Frequency evolution (chirp)
        def chirp_frequency(f):
            # Post-Newtonian chirp
            tau = (5.0 / 256.0) * (np.pi * f)**(-8/3) * chirp_mass**(-5/3)
            return tau
        
        # Strain amplitude
        distance_m = self.config['distance'] * 3.086e22  # Mpc to meters
        h_char = (chirp_mass**(5/6) / distance_m) / (np.pi**(2/3) * freq**(7/6))
        
        # Add echo reflection
        R_infinity = self._schwarzschild_reflection(freq, m1 + m2)
        h_echo = h_char * R_infinity * 0.1  # Echo amplitude
        
        # Total signal
        h_total = h_char + h_echo
        
        # SNR calculation
        snr_integrand = np.abs(h_total)**2 / psd
        snr = np.sqrt(4 * np.trapz(snr_integrand, freq))
        
        self.results['ligo_data'] = {
            'frequency': freq,
            'psd': psd,
            'strain': h_total,
            'echo_component': h_echo,
            'snr': snr,
            'template': h_char
        }
        
        print(f"✓ LIGO data generated: SNR = {snr:.2f}")
        return self.results['ligo_data']
    
    def _schwarzschild_reflection(self, freq, M_total):
        """Calculate Schwarzschild reflection coefficient."""
        # Gravitational wavelength
        r_s = 2 * M_total * 1.477  # km (Schwarzschild radius)
        omega = 2 * np.pi * freq
        
        # Reflection coefficient (simplified)
        k_r = omega * r_s / 3e5  # dimensionless
        R = np.exp(-k_r) * np.sin(k_r) / (1 + k_r**2)
        
        return R
    
    def generate_visuals(self, output_dir='/app/echo-detection-system/outputs'):
        """Generate all visualization outputs."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if 'ligo_data' not in self.results:
            self.generate_ligo_data()
        
        data = self.results['ligo_data']
        
        # 1. SNR Projection Plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Top left: Strain vs Frequency
        ax = axes[0, 0]
        ax.loglog(data['frequency'], np.abs(data['strain']), 'b-', label='Total Signal', linewidth=2)
        ax.loglog(data['frequency'], np.abs(data['template']), 'g--', label='Template', linewidth=1.5)
        ax.loglog(data['frequency'], np.abs(data['echo_component']), 'r:', label='Echo', linewidth=1.5)
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Characteristic Strain', fontsize=12)
        ax.set_title('Gravitational Wave Signal', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Top right: Noise PSD
        ax = axes[0, 1]
        ax.loglog(data['frequency'], np.sqrt(data['psd']), 'k-', linewidth=2)
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Strain Noise (1/√Hz)', fontsize=12)
        ax.set_title('LIGO Noise Curve', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Bottom left: SNR Accumulation
        ax = axes[1, 0]
        snr_cumulative = np.sqrt(4 * cumulative_trapezoid(
            np.abs(data['strain'])**2 / data['psd'], 
            data['frequency'], 
            initial=0
        ))
        ax.semilogx(data['frequency'], snr_cumulative, 'purple', linewidth=2)
        ax.axhline(self.config['snr_threshold'], color='red', linestyle='--', label='Threshold')
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Cumulative SNR', fontsize=12)
        ax.set_title(f'SNR Accumulation (Total={data["snr"]:.2f})', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Bottom right: Echo Contribution
        ax = axes[1, 1]
        echo_fraction = np.abs(data['echo_component']) / np.abs(data['strain'])
        ax.semilogx(data['frequency'], echo_fraction * 100, 'orange', linewidth=2)
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Echo Contribution (%)', fontsize=12)
        ax.set_title('Echo Signal Fraction', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = f'{output_dir}/ligo_snr_projection.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
        
        # 2. Detection Statistics
        self._generate_detection_stats(output_dir)
        
        # 3. Causal Analysis
        self._generate_causal_analysis(output_dir)
        
        return {
            'snr_plot': f'{output_dir}/ligo_snr_projection.png',
            'stats_plot': f'{output_dir}/detection_statistics.png',
            'causal_plot': f'{output_dir}/causal_analysis.png'
        }
    
    def _generate_detection_stats(self, output_dir):
        """Generate detection statistics visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # False alarm rate
        far = np.logspace(-8, -3, 100)
        snr_threshold = np.sqrt(-2 * np.log(far))
        
        ax = axes[0]
        ax.loglog(far, snr_threshold, 'b-', linewidth=2)
        ax.axhline(self.results['ligo_data']['snr'], color='red', linestyle='--', 
                   label=f'Observed SNR={self.results["ligo_data"]["snr"]:.1f}')
        ax.set_xlabel('False Alarm Rate (Hz)', fontsize=12)
        ax.set_ylabel('SNR Threshold', fontsize=12)
        ax.set_title('Detection Threshold vs FAR', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Significance
        sigma = np.linspace(0, 10, 100)
        p_value = 2 * (1 - 0.5 * (1 + np.tanh(sigma / np.sqrt(2))))
        
        ax = axes[1]
        ax.semilogy(sigma, p_value, 'g-', linewidth=2)
        detected_sigma = self.results['ligo_data']['snr'] / np.sqrt(2)
        ax.axvline(detected_sigma, color='red', linestyle='--', 
                   label=f'{detected_sigma:.1f}σ significance')
        ax.set_xlabel('Significance (σ)', fontsize=12)
        ax.set_ylabel('P-value', fontsize=12)
        ax.set_title('Statistical Significance', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/detection_statistics.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/detection_statistics.png")
        plt.close()
    
    def _generate_causal_analysis(self, output_dir):
        """Generate causal deviation analysis."""
        data = self.results['ligo_data']
        omega = 2 * np.pi * data['frequency']
        
        # Compute causal deviation
        R_s = data['strain'] * np.exp(1j * np.angle(data['strain']))
        delta_kk = self.enforcer.compute_causal_deviation(omega, R_s)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.axhline(delta_kk, color='red', linewidth=3, label=f'Δ_KK = {delta_kk:.4f}')
        ax.axhline(0.8, color='orange', linestyle='--', label='Causal Boundary (0.8)')
        ax.axhline(1.2, color='purple', linestyle='--', label='Dual Emergence (1.2)')
        
        ax.set_ylim(0, 2)
        ax.set_xlabel('Measurement', fontsize=12)
        ax.set_ylabel('Causal Deviation', fontsize=12)
        ax.set_title('Kramers-Kronig Causal Analysis', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/causal_analysis.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/causal_analysis.png")
        plt.close()
    
    def generate_nanograph_data(self):
        """Generate nano-scale graph data for WebGL rendering."""
        # Generate geometric projection matrices
        projections = generate_geometric_projections(n_proj=5, rank=2, seed=42)
        
        # Create graph nodes (frequency bins)
        data = self.results.get('ligo_data')
        if not data:
            self.generate_ligo_data()
            data = self.results['ligo_data']
        
        freq = data['frequency']
        strain = np.abs(data['strain'])
        
        # Graph structure for WebGL
        nodes = []
        edges = []
        
        for i, (f, h) in enumerate(zip(freq[::10], strain[::10])):
            nodes.append({
                'id': i,
                'frequency': float(f),
                'amplitude': float(h),
                'x': float(np.log10(f)),
                'y': float(np.log10(h)),
                'z': 0.0
            })
        
        # Connect nodes
        for i in range(len(nodes) - 1):
            edges.append({
                'source': i,
                'target': i + 1,
                'weight': float(abs(nodes[i]['amplitude'] - nodes[i+1]['amplitude']))
            })
        
        nanograph = {
            'nodes': nodes,
            'edges': edges,
            'projections': [p.tolist() for p in projections],
            'metadata': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'frequency_range': [float(freq.min()), float(freq.max())],
                'snr': float(data['snr'])
            }
        }
        
        # Save for WebGL
        output_path = '/app/echo-detection-system/outputs/nanograph_data.json'
        with open(output_path, 'w') as f:
            json.dump(nanograph, f, indent=2)
        
        print(f"✓ Nanograph data generated: {len(nodes)} nodes, {len(edges)} edges")
        print(f"✓ Saved: {output_path}")
        
        self.results['nanograph'] = nanograph
        return nanograph
    
    def print_results(self):
        """Print summary of all results."""
        print("\n" + "="*60)
        print("ECHO DETECTION SYSTEM - RESULTS")
        print("="*60)
        
        if 'ligo_data' in self.results:
            data = self.results['ligo_data']
            print(f"\n📊 LIGO Data Analysis:")
            print(f"  • Frequency range: {data['frequency'].min():.1f} - {data['frequency'].max():.1f} Hz")
            print(f"  • SNR: {data['snr']:.2f}")
            print(f"  • Detection: {'YES ✓' if data['snr'] > self.config['snr_threshold'] else 'NO ✗'}")
            print(f"  • Significance: {data['snr']/np.sqrt(2):.1f}σ")
        
        if 'nanograph' in self.results:
            nano = self.results['nanograph']
            print(f"\n🔬 Nanograph Data:")
            print(f"  • Nodes: {nano['metadata']['total_nodes']}")
            print(f"  • Edges: {nano['metadata']['total_edges']}")
            print(f"  • Projections: {len(nano['projections'])}")
        
        print(f"\n📁 Output Files:")
        import os
        for f in os.listdir('/app/echo-detection-system/outputs'):
            size = os.path.getsize(f'/app/echo-detection-system/outputs/{f}') / 1024
            print(f"  • {f} ({size:.1f} KB)")
        
        print("\n" + "="*60)

def main():
    """Run complete pipeline."""
    print("Starting Echo Detection System...")
    print("="*60)
    
    system = EchoDetectionSystem()
    
    # 1. Generate LIGO data
    print("\n1. Generating LIGO data...")
    system.generate_ligo_data()
    
    # 2. Generate visualizations
    print("\n2. Generating visualizations...")
    system.generate_visuals()
    
    # 3. Generate nanograph for WebGL
    print("\n3. Generating nanograph data...")
    system.generate_nanograph_data()
    
    # 4. Print results
    system.print_results()
    
    print("\n✅ Pipeline complete!")
    print(f"View outputs in: /app/echo-detection-system/outputs/")

if __name__ == "__main__":
    main()
