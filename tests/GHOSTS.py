import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqgv.modules.a_rg_sign_audit import b_total, SM

def running_f2(mu: np.ndarray, mu0: float, f20: float, b_tot: float) -> np.ndarray:
    return 1.0 / (1.0/f20 + b_tot*np.log(mu/mu0))

def create_merlin_viz():
    mu = np.logspace(0, 19, 300)  # GeV from 1 to Planck scale
    mu0 = 1.0
    b = b_total(SM, xi=1/6)

    # RG Flow for f2
    f2_tanner = running_f2(mu, mu0, f20=0.8, b_tot=b)
    
    # Merlin Mode Dynamics
    # Stability: 1 at UV (Planck scale), decays to 0 at IR
    # Backward propagation: High at UV, zero at IR
    merlin_stability = np.exp((np.log10(mu) - 19) / 2.0) 
    back_propagation = np.exp((np.log10(mu) - 19) / 1.0)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot RG Flow
    ax.plot(np.log10(mu), f2_tanner, merlin_stability, label='RG Flow (f2)', color='blue', linewidth=2)
    
    # Highlight UV and Merlin
    # UV is at mu = 10^19
    ax.scatter([19], [f2_tanner[-1]], [1], color='red', s=100, label='UV (Planck Scale)')
    ax.text(19, f2_tanner[-1], 1.1, "Merlin Mode Active", color='red', fontweight='bold')

    # Highlight IR
    ax.scatter([0], [f2_tanner[0]], [0], color='green', s=100, label='IR (Causality Emergent)')
    ax.text(0, f2_tanner[0], -0.1, "Merlin Decayed", color='green', fontweight='bold')

    # Visualize Backward Propagation as a "shadow" or secondary line
    ax.plot(np.log10(mu), f2_tanner, back_propagation, '--', color='purple', alpha=0.5, label='Backward Propagation')

    ax.set_xlabel('Energy Scale log10(mu/GeV)')
    ax.set_ylabel('Coupling f2')
    ax.set_zlabel('Merlin Stability / Acausality')
    ax.set_title('RG Flow with Merlin Mode as UV Regulator', fontsize=15, fontweight='bold')
    ax.legend()

    # Add text box for physical interpretation
    textstr = '\n'.join((
        r'UV: Merlin Active (Acausal)',
        r'IR: Merlin Decayed (Causal)',
        r'Causality is Emergent',
        r'UV Regulator = Unstable Ghost'
    ))
    ax.text2D(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('/home/ubuntu/merlin_rg_flow_3d.png', dpi=300)
    print("Visualization saved to /home/ubuntu/merlin_rg_flow_3d.png")

if __name__ == "__main__":
    create_merlin_viz()
