import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import interp1d

# --- 1. Metric & Potential Setup (Same as before) ---
def calculate_metric_and_potential(M, a, b_val, r_grid):
    # m(r) for Lee-Wick 
    C = M / (2.0 * a)
    term1 = 2*a + (a**2 + b_val**2) * r_grid
    term2 = a**2 - b_val**2 + a * (a**2 + b_val**2) * r_grid
    bracket = term1 * np.cos(b_val * r_grid) + term2 * np.sin(b_val * r_grid)
    exp_part = np.exp(-a * r_grid)
    m_r = M - C * exp_part * bracket
    
    # Regularization: Enforce 2-2 Hole Structure (Deep Cavity, No Horizon)
    # We cap A(r) at epsilon to prevent division by zero, creating a deep potential well.
    A_raw = 1.0 - 2.0 * m_r / r_grid
    epsilon = 1e-5 
    A_r = np.maximum(A_raw, epsilon)
    
    V_eff = A_r * (6.0 / r_grid**2 - 6.0 * m_r / r_grid**3)
    return m_r, A_r, V_eff

# Parameters
M = 1.0
a = 20.0
ratio = 0.1
b_val = a / ratio # b = 200 (The "Photonic Crystal" Frequency)

# Grid Generation
r_fine = np.linspace(1e-4, 2.5, 5000) # Dense core
r_coarse = np.linspace(2.5, 50.0, 2000)
r_grid = np.concatenate([r_fine, r_coarse])
r_grid = np.sort(np.unique(r_grid))

m_r, A_r, V_eff = calculate_metric_and_potential(M, a, b_val, r_grid)

# --- 2. Tortoise Coordinate & Interpolation ---
integrand = 1.0 / A_r
r_star = cumulative_trapezoid(integrand, r_grid, initial=0)
# Shift to match Schwarzschild at large r
r_ref = r_grid[-1]
r_star_sch_ref = r_ref + 2*M * np.log(r_ref/(2*M) - 1)
shift = r_star_sch_ref - r_star[-1]
r_star += shift

# Interpolate V(r*) for the Solver
V_interp = interp1d(r_star, V_eff, kind='cubic', fill_value='extrapolate')

# --- 3. Wave Equation Solver ---
def wave_equation(t, y, omega):
    # t is r*, y is [psi, psi']
    psi, psip = y
    V = V_interp(t)
    psipp = -(omega**2 - V) * psi
    return [psip, psipp]

# Frequency Scan
omegas = np.linspace(0.1, 15.0, 100) # Scan up to high UV
reflectivities = []
phases = []

# Boundary Condition: Regular Core (psi ~ r^3 -> small value, finite derivative)
# Start integration deep in the well (r* is large negative)
r_start = r_star[50] # Just off the singularity
y0 = [1e-10, 1e-5] 

print(f"Starting Scattering Simulation (a={a}, b={b_val})...")

for omega in omegas:
    # Integrate Outward
    sol = solve_ivp(wave_equation, [r_start, r_star[-1]], y0, args=(omega,), 
                    max_step=0.05, method='RK45')
    
    # Extract Asymptotic Coefficients
    # psi ~ I * e^{-iwt} + O * e^{+iwt}  (where t is r*)
    # At large r*: psi(r*) = outgoing + ingoing
    
    psi_L = sol.y[0][-1]
    psip_L = sol.y[1][-1]
    r_L = sol.t[-1]
    
    # Solve for Outgoing (O) and Ingoing (I) amplitudes
    # Based on: psi = I e^{-i w r} + O e^{+i w r}
    term_O = (1j * omega * psi_L + psip_L) / (2j * omega) * np.exp(-1j * omega * r_L)
    term_I = (1j * omega * psi_L - psip_L) / (2j * omega) * np.exp(1j * omega * r_L)
    
    # Reflectivity R = |O/I|^2
    R = np.abs(term_O / term_I)**2
    reflectivities.append(R)
    
    # Phase Shift (delta)
    # The delay is determined by the phase of the reflection coefficient
    phase = np.angle(term_O / term_I)
    phases.append(phase)

# Unwrap phase to calculate time delay
phases = np.unwrap(phases)
# Time Delay = d(Phase)/d(omega)
time_delay = np.gradient(phases, omegas)

# --- 4. Plotting ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Plot Reflectivity
ax1.plot(omegas, reflectivities, 'b-', linewidth=2)
ax1.set_ylabel('Reflectivity $|R|^2$')
ax1.set_title(f'Scattering off the Time Parachute (a={a}, b={b_val})')
ax1.grid(True)
ax1.set_ylim(0.9, 1.1) # Expect unitary reflection (approx 1.0)

# Plot Time Delay (The Drag)
ax2.plot(omegas, time_delay, 'r-', linewidth=2)
ax2.set_ylabel('Time Delay $d\phi/d\omega$')
ax2.set_xlabel('Frequency $\omega M$')
ax2.set_title('The Drag: Resonant Time Delays in the UV')
ax2.grid(True)

plt.tight_layout()
plt.savefig('scattering_delay.png')
plt.show()
