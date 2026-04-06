"""
CJPT System - Full Implementation from mobius_jacobi.txt
"""
import numpy as np
from scipy.integrate import trapezoid
import sys
import os

# Import user's actual files
sys.path.insert(0, os.path.dirname(__file__))

class CJPTSystem:
    def __init__(self):
        self.M_Pl = 2.435e18
        self.xi_H = 5e8
        self.eta = 0.8
        self.kappa_mc = 50
        
    def compute_M2(self, f2):
        return self.M_Pl / np.sqrt(f2)
    
    def compute_J_bound(self, f2, H):
        return self.kappa_mc * np.sqrt(f2) * (H / self.M_Pl)
    
    def generate_ligo_data_full(self, f2=1e-8):
        H = 1e13
        M2 = self.compute_M2(f2)
        J_bound = self.compute_J_bound(f2, H)
        
        freq = np.logspace(1, 3, 2000)
        
        # LIGO noise
        f0, S0 = 215.0, 1e-49
        psd = S0 * ((freq/10)**(-4) + 1.0 + (freq/500)**2)
        
        # Waveform
        chirp_mass = 25.0
        distance = 400 * 3.086e24
        h_char = (chirp_mass**(5/6) / distance) / (np.pi**(2/3) * freq**(7/6))
        
        # Echo with Schwarzschild reflection
        M_total = 60
        r_s = 2 * M_total * 1477
        omega = 2 * np.pi * freq
        k_r = omega * r_s / 3e8
        R_infinity = np.exp(-k_r) * np.sin(k_r) / (1 + k_r**2)
        h_echo = h_char * R_infinity * 0.1
        
        h_total = h_char + h_echo
        
        # SNR
        snr = np.sqrt(4 * trapezoid(np.abs(h_total)**2 / psd, freq))
        
        # Causal deviation
        phase_measured = np.angle(h_total * (1 + 0.01j * np.random.randn(len(freq))))
        log_mag = np.log(np.abs(h_total) + 1e-12)
        
        # Hilbert transform for KK phase
        from scipy.fft import fft, ifft
        H_sign = np.zeros(len(freq))
        H_sign[1:len(freq)//2] = 1
        H_sign[len(freq)//2+1:] = -1
        phase_kk = np.real(ifft(1j * H_sign * fft(log_mag)))
        
        delta_kk = np.linalg.norm(phase_measured - phase_kk) / np.linalg.norm(h_total)
        
        # Phase determination
        g_trap = H / M2 / 0.01
        
        if g_trap < 1.0 and delta_kk < 0.8 * J_bound:
            phase = "MINIMAL_PHASE"
        elif 1.0 <= g_trap <= 1.5 and 0.8*J_bound <= delta_kk <= 1.2*J_bound:
            phase = "BOUND_RECONSTRUCTION"
        elif g_trap > 1.5 and delta_kk > 1.2 * J_bound:
            phase = "DUAL_EMERGENCE"
        else:
            phase = "TRANSITION"
        
        return {
            'frequency': freq[::20].tolist(),
            'strain': np.abs(h_total)[::20].tolist(),
            'echo': np.abs(h_echo)[::20].tolist(),
            'psd': psd[::20].tolist(),
            'snr': float(snr),
            'detection': bool(snr > 8),
            'f2': float(f2),
            'M2': float(M2),
            'J_bound': float(J_bound),
            'delta_kk': float(delta_kk),
            'g_trap': float(g_trap),
            'phase': phase
        }
    
    def f2_scan(self, n_points=10):
        f2_values = np.logspace(-8.3, -7.7, n_points)
        results = []
        
        for f2 in f2_values:
            data = self.generate_ligo_data_full(f2)
            results.append({
                'f2': data['f2'],
                'snr': data['snr'],
                'delta_kk': data['delta_kk'],
                'phase': data['phase']
            })
        
        return results
    
    def generate_nanograph_physics(self):
        data = self.generate_ligo_data_full()
        
        nodes = []
        freq = data['frequency']
        strain = data['strain']
        
        for i, (f, s) in enumerate(zip(freq, strain)):
            nodes.append({
                'id': i,
                'freq': f,
                'amp': s,
                'x': float(np.log10(f)),
                'y': float(np.log10(s + 1e-12)),
                'z': 0.0
            })
        
        edges = [{'source': i, 'target': i+1} for i in range(len(nodes)-1)]
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'snr': data['snr'],
                'phase': data['phase'],
                'f2': data['f2']
            }
        }

cjpt = CJPTSystem()
