"""
Quick test of CJPT system - reduced complexity for fast verification.
"""

import sys
sys.path.append('/app/utils')
sys.path.append('/app/src')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from causal_enforcer import CausalEnforcer
from tensor_cell import TensorCell
from cjpt_system import CJPTSystem
from visualizer import CJPTVisualizer

print("="*60)
print("CJPT Quick Test")
print("="*60)

# Initialize components
print("\n1. Initializing components...")
cjpt = CJPTSystem()
enforcer = CausalEnforcer(kk_tolerance=0.8, enforce_kk=True)

genome = {
    'apply_rotation': True,
    'use_geometric_projections': True,
    'num_projections': 3,
    'projection_rank': 2,
}
tensor_cell = TensorCell(genome, causal_enforcer=enforcer)

print(f"   M_Pl = {cjpt.M_Pl:.2e} GeV")
print(f"   xi_H = {cjpt.xi_H:.2e}")
print(f"   TensorCell UUID: {tensor_cell.uuid}")

# Test single step
print("\n2. Testing single processing step...")
f2 = 1e-8
H = 1e13
M2 = cjpt.compute_M2(f2)
J_bound = cjpt.compute_J_bound(f2, H)

print(f"   f2 = {f2:.2e}")
print(f"   M2 = {M2:.2e} GeV")
print(f"   J_bound = {J_bound:.2e}")

# Generate test data
strain = np.random.randn(2, 500)
comb = np.ones_like(strain)

result = tensor_cell.solve_physics({'strain': strain, 'comb_mask': comb})
print(f"   Processing result: score={result['score']:.3e}, time={result['elapsed_ms']:.2f}ms")

# Test causal deviation
print("\n3. Testing causal deviation computation...")
omega = np.linspace(1e-2, 1e2, 200)
R_s = np.exp(-(omega - 50)**2 / 100) * np.exp(1j * 0.1 * omega)
delta_kk = enforcer.compute_causal_deviation(omega, R_s)
print(f"   Δ_KK = {delta_kk:.4e}")

# Test phase detection
print("\n4. Testing phase detection...")
M_matrix = np.array([[10*H**2, 2*H**2], [2*H**2, 8*H**2]])
Omega = np.eye(4)
sigma_env = 1e-8

g_trap = cjpt.geometric_trap_score(H, M2, M_matrix, Omega)
phase = cjpt.cjpt_phase_check(g_trap, delta_kk, J_bound, sigma_env, cjpt.xi_H, H)

print(f"   G_trap = {g_trap:.3f}")
print(f"   Phase: {phase}")

# Run mini rollout
print("\n5. Running mini rollout (10 steps)...")
for step in range(10):
    # Generate data
    strain = np.random.randn(2, 500) * (1 + step*0.1)
    result = tensor_cell.solve_physics({'strain': strain, 'comb_mask': np.ones_like(strain)})
    
    # Compute deviation
    R_s = np.exp(-(omega - 50)**2 / (100 + step*10)) * np.exp(1j * 0.1 * omega * (1 + step*0.05))
    delta_kk = enforcer.compute_causal_deviation(omega, R_s)
    
    # Phase check
    sigma_env = 1e-8 * (1 + step*0.1)
    g_trap = cjpt.geometric_trap_score(H, M2, M_matrix, Omega)
    phase = cjpt.cjpt_phase_check(g_trap, delta_kk, J_bound, sigma_env, cjpt.xi_H, H)
    
    if step % 3 == 0:
        print(f"   Step {step}: phase={phase:25s} Δ_KK={delta_kk:.3e}")

# Generate visualization
print("\n6. Generating phase diagram...")
visualizer = CJPTVisualizer(output_dir="/app/outputs")
visualizer.plot_phase_diagram(cjpt)
print("   Saved to /app/outputs/phase_diagram.png")

# Summary
print("\n" + "="*60)
print("QUICK TEST COMPLETE")
print("="*60)
print("\nGenerated outputs:")
output_dir = Path("/app/outputs")
for f in sorted(output_dir.glob("*.png")):
    print(f"  ✓ {f.name}")

print("\nAll core components functional!")
print("Run full simulation: python /app/src/cjpt_simulation.py")
print("="*60)
