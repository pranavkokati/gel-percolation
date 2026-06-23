# %% [markdown]
# # Notebook 1: Percolation Threshold and Inverse Percolation Transition
#
# This notebook demonstrates:
# 1. Building a 3D random geometric graph representing an alginate hydrogel
# 2. Computing P∞(ρ_x) — the percolation transition vs crosslink density
# 3. Simulating enzymatic (MMP) degradation — tracking P∞(t) and G'(t)
# 4. Fitting the critical scaling G' ~ |p − p_c|^f·ω^Δ

# %% Setup
import sys
sys.path.insert(0, "..")

import numpy as np
import matplotlib.pyplot as plt
from src.network_model import HydrogelNetwork, HydrogelParams
from src.mechanical_properties import (
    PercolationMechanics, MechanicsParams,
    PercolationCriticalExponents, FrequencySweepAnalyzer,
)

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
})
print("Setup complete.")

# %% [markdown]
# ## 1. Percolation Transition: P∞ vs Crosslink Density
#
# We build networks at varying ρ_x (crosslink density) and measure P∞.
# The sharp rise at p_c is the hallmark of a continuous phase transition.

# %% Build networks at varying rho_x
rho_x_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0])
p_inf_values = []
n_repeats = 3  # average over multiple seeds

for rho_x in rho_x_values:
    p_inf_reps = []
    for seed in range(n_repeats):
        params = HydrogelParams(box_size=30.0, rho_x=rho_x, r_c=5.0, covalent_fraction=1.0)
        net = HydrogelNetwork(params, seed=seed)
        p_inf_reps.append(net.get_percolation_order_parameter())
    p_inf_values.append(np.mean(p_inf_reps))

p_inf_values = np.array(p_inf_values)
print("P_inf values computed.")

# %% Plot P∞ vs ρ_x
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(rho_x_values, p_inf_values, "bo-", ms=8, lw=2, label="P∞ (simulation)")
ax.axvline(0.3, color="red", ls="--", alpha=0.7, label="Approx. p_c threshold")
ax.set_xlabel("Crosslink density ρ_x [µm⁻³]")
ax.set_ylabel("Percolation order parameter P∞")
ax.set_title("3D Percolation Transition in Hydrogel Network")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../results/01_percolation_transition.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figure saved.")

# %% [markdown]
# ## 2. Enzymatic Degradation: P∞(t) and G'(t)
#
# Starting from a fully intact network, we apply MMP and track how
# P∞ and G' evolve. Near p_c, the network mechanically collapses.

# %% Simulate degradation
params = HydrogelParams(box_size=40.0, rho_x=1.5, r_c=5.0, k_base=0.02, covalent_fraction=1.0)
net = HydrogelNetwork(params, seed=42)
mech = PercolationMechanics()
analyzer = FrequencySweepAnalyzer(mech)

mmp_concentrations = [0.0, 0.5, 1.0, 2.0]  # nM
omega = np.logspace(-1, 2, 30)

print(f"Initial network: {net}")

# Run degradation steps
n_deg_steps = 200
dt = 5.0
times = []
p_inf_time = []
G_prime_time = []
bond_fraction_time = []

mmp_field = np.ones((10, 10, 10)) * 1.0  # 1 nM uniform MMP

for step in range(n_deg_steps):
    p = net.get_bond_fraction()
    p_inf = net.get_percolation_order_parameter()
    G_prime = mech.compute_shear_modulus(p, omega=1.0)

    times.append(step * dt)
    p_inf_time.append(p_inf)
    G_prime_time.append(G_prime)
    bond_fraction_time.append(p)

    if G_prime == 0.0 and step > 10:  # network has dissolved
        print(f"  Gel-sol transition at step {step} (t={step*dt:.0f}s, p={p:.3f})")
        # Pad remaining with zeros
        remaining = n_deg_steps - step - 1
        times.extend([(step + i + 1) * dt for i in range(remaining)])
        p_inf_time.extend([0.0] * remaining)
        G_prime_time.extend([0.0] * remaining)
        bond_fraction_time.extend([p] * remaining)
        break

    net.degrade_step(mmp_field, dt=dt)

times = np.array(times)
p_inf_time = np.array(p_inf_time)
G_prime_time = np.array(G_prime_time)
bond_fraction_time = np.array(bond_fraction_time)

# %% Plot P∞(t) and G'(t)
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axes[0].plot(times / 60, p_inf_time, "b-", lw=2, label="P∞ (order parameter)")
axes[0].axhline(0.2593, color="red", ls="--", alpha=0.7, label="p_c = 0.259")
axes[0].set_ylabel("P∞")
axes[0].set_title("Inverse Percolation: Enzymatic Degradation Dynamics")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-0.05, 1.05)

axes[1].semilogy(times / 60, G_prime_time + 1e-3, "g-", lw=2, label="G' [Pa]")
axes[1].axhline(1.0, color="gray", ls=":", alpha=0.5)
axes[1].set_xlabel("Time [min]")
axes[1].set_ylabel("G' [Pa] (log scale)")
axes[1].set_title("Mechanical Collapse: G'(t)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../results/01_degradation_dynamics.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Frequency Sweep at Different Degradation Stages
#
# The Winter–Chambon criterion: at the gel point, tan δ = G''/G' is
# frequency-independent. This is the key experimental signature.

# %% Frequency sweeps at different p values
p_stages = [0.8, 0.5, 0.3, 0.27, 0.26]
omega_range = np.logspace(-2, 3, 50)
colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(p_stages)))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for p, color in zip(p_stages, colors):
    G_prime, G_dbl = analyzer.compute_frequency_sweep(p, omega_range)
    label = f"p = {p:.2f}"
    axes[0].loglog(omega_range, G_prime + 1e-3, "-", color=color, label=label, lw=2)
    axes[1].loglog(omega_range, G_dbl + 1e-3,  "--", color=color, label=label, lw=2)

axes[0].set_xlabel("ω [rad/s]")
axes[0].set_ylabel("G' [Pa]")
axes[0].set_title("G'(ω) at Different Degradation Stages")
axes[0].legend(fontsize=9)
axes[0].grid(True, which="both", alpha=0.3)

axes[1].set_xlabel("ω [rad/s]")
axes[1].set_ylabel("G'' [Pa]")
axes[1].set_title("G''(ω) at Different Degradation Stages")
axes[1].legend(fontsize=9)
axes[1].grid(True, which="both", alpha=0.3)

fig.suptitle("Frequency Sweeps: Approaching the Gel Point", fontsize=14)
plt.tight_layout()
plt.savefig("../results/01_frequency_sweeps.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Critical Scaling: G' ~ |p − p_c|^f
#
# Near p_c, the modulus follows a power law with exponent f.
# For alginate (bond-bending networks): f ≈ 2.1
# For PEG (central-force networks): f ≈ 3.75

# %% Fit critical exponent
from src.percolation_analysis import CriticalExponentFitter

p_vals = np.linspace(0.261, 0.8, 60)
G_vals = np.array([mech.compute_shear_modulus(p, omega=1.0) for p in p_vals])
G_vals_noisy = G_vals + 0.05 * np.max(G_vals) * np.random.default_rng(0).standard_normal(len(G_vals))
G_vals_noisy = np.maximum(G_vals_noisy, 0.0)

result = CriticalExponentFitter.fit_modulus_scaling(p_vals, G_vals, p_c=0.2593)
print(f"Fitted elastic exponent f = {result['f_fit']:.3f} (expected 2.1)")
print(f"Fit quality R² = {result['fit_quality']:.3f}")

fig = CriticalExponentFitter.plot_scaling_collapse(
    p_vals, G_vals_noisy, result["f_fit"], p_c=0.2593,
    xlabel="|p − p_c|", ylabel="G' [Pa]"
)
if fig:
    fig.suptitle(f"Critical Scaling: G' ~ |p−p_c|^f,  f_fit={result['f_fit']:.2f}")
    plt.tight_layout()
    plt.savefig("../results/01_critical_scaling.png", dpi=150, bbox_inches="tight")
    plt.show()

print("\nNotebook 1 complete.")
