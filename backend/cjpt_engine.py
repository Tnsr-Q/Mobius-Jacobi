import numpy as np
from scipy.integrate import trapezoid
import sys
sys.path.insert(0, '/app/echo-detection-system/src')

class CJPTEngine:
    def __init__(self):
        self.M_Pl = 2.435e18
        self.xi_H = 5e8
        
    def generate_ligo_data(self):
        freq = np.logspace(1, 3, 1000)
        f0, S0 = 215.0, 1e-49
        psd = S0 * ((freq/10)**(-4) + 1.0 + (freq/500)**2)
        
        chirp_mass = 25.0
        distance = 400e6 * 3.086e22
        h_char = (chirp_mass**(5/6) / distance) / (np.pi**(2/3) * freq**(7/6))
        
        M_total = 60
        r_s = 2 * M_total * 1.477
        omega = 2 * np.pi * freq
        k_r = omega * r_s / 3e5
        R_inf = np.exp(-k_r) * np.sin(k_r) / (1 + k_r**2)
        h_echo = h_char * R_inf * 0.1
        h_total = h_char + h_echo
        
        snr = np.sqrt(4 * trapezoid(np.abs(h_total)**2 / psd, freq))
        
        return {
            'frequency': freq.tolist(),
            'strain': np.abs(h_total).tolist(),
            'echo': np.abs(h_echo).tolist(),
            'snr': float(snr),
            'detection': bool(snr > 8)
        }
    
    def generate_nanograph(self):
        nodes = []
        for i in range(100):
            angle = 2 * np.pi * i / 100
            r = 1 + 0.2 * np.sin(3 * angle)
            nodes.append({
                'id': i,
                'x': float(r * np.cos(angle)),
                'y': float(r * np.sin(angle)),
                'z': float(0.3 * np.sin(5 * angle))
            })
        edges = [{'source': i, 'target': (i+1)%100, 'weight': 1.0} for i in range(100)]
        return {'nodes': nodes, 'edges': edges}

engine = CJPTEngine()
