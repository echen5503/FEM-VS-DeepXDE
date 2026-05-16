"""
Generate all figures for the FEM beamer presentation.
Outputs go to figures/ as both PDF (for LaTeX) and PNG (for preview).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs('figures', exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'lines.linewidth': 2.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
})

BLUE   = '#1565C0'
ORANGE = '#E65100'
GREEN  = '#2E7D32'
RED    = '#C62828'

def save(name):
    plt.savefig(f'figures/{name}.pdf', bbox_inches='tight')
    plt.savefig(f'figures/{name}.png', bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  saved {name}')


# ============================================================
# Figure 1: Bar Mesh / Discretization  (Slide 4)
# ============================================================
fig, ax = plt.subplots(figsize=(7.5, 2.2))
ax.set_xlim(-0.08, 1.08)
ax.set_ylim(-0.6, 0.85)
ax.axis('off')

bar_y, bar_h = 0.2, 0.28
nodes_x = [0, 1/3, 2/3, 1.0]
x_labels = ['$x_1=0$', '$x_2=h$', '$x_3=2h$', '$x_4=L$']
T_labels  = ['$T_1$', '$T_2$', '$T_3$', '$T_4$']
elem_labels = ['Element 1', 'Element 2', 'Element 3']
elem_colors = ['#BBDEFB', '#90CAF9', '#64B5F6']

for i in range(3):
    rect = mpatches.FancyBboxPatch(
        (nodes_x[i], bar_y - bar_h / 2), nodes_x[i+1] - nodes_x[i], bar_h,
        boxstyle='square,pad=0', facecolor=elem_colors[i], edgecolor=BLUE, linewidth=1.5)
    ax.add_patch(rect)
    ax.text((nodes_x[i] + nodes_x[i+1]) / 2, bar_y,
            elem_labels[i], ha='center', va='center', fontsize=9.5, fontweight='bold', color=BLUE)

for xi, xl, tl in zip(nodes_x, x_labels, T_labels):
    ax.plot(xi, bar_y, 'o', color=RED, markersize=11, zorder=5)
    ax.text(xi, bar_y - bar_h / 2 - 0.13, xl, ha='center', va='top', fontsize=8.5)
    ax.text(xi, bar_y + bar_h / 2 + 0.08, tl, ha='center', va='bottom',
            fontsize=10, color=RED, fontweight='bold')

ax.set_title('Discretization of a 1D Bar into Linear Elements', pad=6)
save('fig_bar_mesh')


# ============================================================
# Figure 2: Shape Functions  (Slide 4a)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

h = 1.0
xv = np.linspace(0, h, 200)
funcs  = [1 - xv / h, xv / h]
titles = ['Left node: $N_1(x) = 1 - x/h$', 'Right node: $N_2(x) = x/h$']
colors = [BLUE, ORANGE]
dots   = [(1, 0), (0, 1)]

for ax, N, ttl, col, (d0, d1) in zip(axes, funcs, titles, colors, dots):
    ax.plot(xv, N, color=col, linewidth=2.5)
    ax.fill_between(xv, N, alpha=0.12, color=col)
    ax.plot([0, h], [d0, d1], 'o', color=col, markersize=8, zorder=5)
    ax.set_xlim(-0.05, h + 0.05)
    ax.set_ylim(-0.15, 1.25)
    ax.set_xticks([0, h])
    ax.set_xticklabels(['$0$', '$h$'])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel('$x$ (local)')
    ax.set_title(ttl, fontsize=10)
    ax.axhline(0, color='black', linewidth=0.8)

fig.suptitle('Linear "Lego" Shape Functions for One Element', y=1.02, fontsize=12)
plt.tight_layout()
save('fig_shape_functions')


# ============================================================
# Figure 3: Example 1 — 3-node bar, Neumann BC at right
# ============================================================
x_nodes = np.array([0.0, 1.0, 2.0])
T_nodes = np.array([100.0, 130.0, 140.0])

x_exact = np.linspace(0, 2, 300)
T_exact = -10 * x_exact**2 + 40 * x_exact + 100

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.plot(x_exact, T_exact, '--', color=ORANGE, lw=2, label='Analytical (exact)')
ax.plot(x_nodes, T_nodes, 'o-', color=BLUE, lw=2, markersize=9, label='FEM nodes')
for xi, Ti in zip(x_nodes, T_nodes):
    ax.annotate(f'$T={Ti:.0f}°$C', (xi, Ti),
                textcoords='offset points', xytext=(4, 7), fontsize=9, color=BLUE)
ax.set_xlabel('Position $x$ (m)')
ax.set_ylabel('Temperature $T$ (°C)')
ax.set_title('Example 1: Heat Source + Insulated Right End')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Node 1\n($x=0$)', 'Node 2\n($x=h$)', 'Node 3\n($x=2h$)'])
ax.legend()
plt.tight_layout()
save('fig_example1')


# ============================================================
# Figure 4: Example 2 — Asymmetric Radiator
# ============================================================
x_nodes = np.array([0.0, 2.0, 3.0])
T_nodes = np.array([10.0, 38.0, 50.0])

x1 = np.linspace(0, 2, 200)
x2 = np.linspace(2, 3, 100)
T1 = -x1**2 + 16 * x1 + 10
T2 = 12 * x2 + 14

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.plot(np.concatenate([x1, x2]), np.concatenate([T1, T2]),
        '--', color=ORANGE, lw=2, label='Analytical (exact)')
ax.plot(x_nodes, T_nodes, 'o-', color=BLUE, lw=2, markersize=9, label='FEM nodes')
ax.axvline(2, color='grey', lw=1, linestyle=':', alpha=0.7)
ax.text(2.05, 12, 'Element boundary', fontsize=8, color='grey', rotation=90, va='bottom')
for xi, Ti in zip(x_nodes, T_nodes):
    ax.annotate(f'$T={Ti:.0f}°$C', (xi, Ti),
                textcoords='offset points', xytext=(4, 7), fontsize=9, color=BLUE)
ax.set_xlabel('Position $x$ (m)')
ax.set_ylabel('Temperature $T$ (°C)')
ax.set_title('Example 2: Asymmetric Radiator (Different Element Lengths)')
ax.set_xticks([0, 1, 2, 3])
ax.legend()
plt.tight_layout()
save('fig_example2')


# ============================================================
# Figure 5: Example 3 — Endothermic Fridge
# ============================================================
x_nodes = np.array([0.0, 1.0, 2.0])
T_nodes = np.array([100.0, 95.0, 100.0])

x_exact = np.linspace(0, 2, 300)
T_exact = 5 * x_exact**2 - 10 * x_exact + 100

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.plot(x_exact, T_exact, '--', color=ORANGE, lw=2, label='Analytical (exact)')
ax.plot(x_nodes, T_nodes, 'o-', color=BLUE, lw=2, markersize=9, label='FEM nodes')
ax.axhline(100, color=GREEN, lw=1, linestyle=':', alpha=0.7, label='Boundary temp (100°C)')
for xi, Ti in zip(x_nodes, T_nodes):
    ax.annotate(f'$T={Ti:.0f}°$C', (xi, Ti),
                textcoords='offset points', xytext=(4, -15), fontsize=9, color=BLUE)
ax.set_xlabel('Position $x$ (m)')
ax.set_ylabel('Temperature $T$ (°C)')
ax.set_title('Example 3: Endothermic Fridge ($Q < 0$, Both Ends at 100°C)')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Node 1\n($x=0$)', 'Node 2\n($x=h$)', 'Node 3\n($x=2h$)'])
ax.set_ylim(88, 108)
ax.legend()
plt.tight_layout()
save('fig_example3')


# ============================================================
# Figure 6: FEM Convergence — config problem  (Appendix)
# ============================================================
from config import L, k, Q, T0, TL

C2 = T0
C1 = (TL - T0 + Q * L**2 / (2 * k)) / L
x_fine = np.linspace(0, L, 500)
T_exact_cfg = (-Q / (2 * k)) * x_fine**2 + C1 * x_fine + C2

def fem_solve(n_elem):
    n_n = n_elem + 1
    xv = np.linspace(0, L, n_n)
    hv = L / n_elem
    K = np.zeros((n_n, n_n))
    F = np.zeros(n_n)
    Ke = (k / hv) * np.array([[1., -1.], [-1., 1.]])
    Fe = (Q * hv / 2.) * np.array([1., 1.])
    for i in range(n_elem):
        K[i:i+2, i:i+2] += Ke
        F[i:i+2] += Fe
    K[0, :] = 0.; K[0, 0] = 1.; F[0] = T0
    K[-1, :] = 0.; K[-1, -1] = 1.; F[-1] = TL
    return xv, np.linalg.solve(K, F)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))

ax1.plot(x_fine, T_exact_cfg, 'k--', lw=2, label='Analytical', zorder=5)
for n, col in zip([2, 4, 10], [RED, ORANGE, BLUE]):
    xv, Tv = fem_solve(n)
    ax1.plot(xv, Tv, 'o-', color=col, lw=1.5, markersize=5, label=f'FEM  n={n}')
ax1.set_xlabel('Position $x$ (m)')
ax1.set_ylabel('Temperature $T$ (°C)')
ax1.set_title('FEM Solutions vs Analytical')
ax1.legend(fontsize=8)

n_list = [1, 2, 3, 4, 6, 8, 10, 15, 20, 30, 50]
errors = []
for n in n_list:
    xv, Tv = fem_solve(n)
    T_ref = (-Q / (2 * k)) * xv**2 + C1 * xv + C2
    errors.append(np.sqrt(np.mean((Tv - T_ref)**2)))

ax2.loglog(n_list, errors, 'o-', color=BLUE, lw=2, markersize=6, label='L2 error')
ref_x = np.array([1, 50], dtype=float)
ref_y = errors[0] * (ref_x / n_list[0])**(-2)
ax2.loglog(ref_x, ref_y, '--', color='grey', lw=1.5, label='$O(h^2)$ slope')
ax2.set_xlabel('Number of Elements')
ax2.set_ylabel('RMS Error (°C)')
ax2.set_title('Convergence Rate')
ax2.legend(fontsize=8)

fig.suptitle('FEM Accuracy — Steady-State 1D Heat Conduction', fontsize=12, y=1.01)
plt.tight_layout()
save('fig_fem_convergence')


# ============================================================
# Figure 7: FEM vs Analytical, config problem, n=10
# ============================================================
xv, Tv = fem_solve(10)

fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.plot(x_fine, T_exact_cfg, 'k--', lw=2, label='Analytical')
ax.plot(xv, Tv, 'o-', color=BLUE, lw=2, markersize=7, label='FEM (10 elements)')
ax.set_xlabel('Position $x$ (m)')
ax.set_ylabel('Temperature $T$ (°C)')
ax.set_title('Steady-State 1D Heat Conduction: FEM vs Analytical')
ax.legend()
plt.tight_layout()
save('fig_fem_vs_analytical')


# ============================================================
# Figure 8: Reaction term breaks nodal exactness
#   PDE: -k T'' + β T = Q,  T(0)=T(L)=0
#   Exact: T = A cosh(μx) + B sinh(μx) + Q/β,  μ=√(β/k)
# ============================================================
k_r, beta, Q_r, L_r = 1.0, 20.0, 10.0, 1.0
mu = np.sqrt(beta / k_r)

# Exact solution coefficients (T(0)=T(L)=0)
# Derivation: T = A cosh(μx) + B sinh(μx) + Q/β
#   T(0)=0 → A = -Q/β
#   T(L)=0 → B = (Q/β)(cosh(μL)−1)/sinh(μL)
A = -Q_r / beta
B = (Q_r / beta) * (np.cosh(mu * L_r) - 1) / np.sinh(mu * L_r)

def T_exact_reaction(x):
    return A * np.cosh(mu * x) + B * np.sinh(mu * x) + Q_r / beta

def fem_reaction(n_elem):
    n_n = n_elem + 1
    hv = L_r / n_elem
    xv = np.linspace(0, L_r, n_n)
    Ke = (k_r / hv) * np.array([[1., -1.], [-1., 1.]])
    Me = (beta * hv / 6.) * np.array([[2., 1.], [1., 2.]])
    Fe = (Q_r * hv / 2.) * np.array([1., 1.])
    K = np.zeros((n_n, n_n))
    F = np.zeros(n_n)
    for i in range(n_elem):
        K[i:i+2, i:i+2] += Ke + Me
        F[i:i+2] += Fe
    K[0, :] = 0.; K[0, 0] = 1.; F[0] = 0.
    K[-1, :] = 0.; K[-1, -1] = 1.; F[-1] = 0.
    return xv, np.linalg.solve(K, F)

x_fine_r = np.linspace(0, L_r, 400)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

# Panel A: solution with visible nodal error for coarse mesh
n_coarse = 4
xv4, Tv4 = fem_reaction(n_coarse)
ax1.plot(x_fine_r, T_exact_reaction(x_fine_r), 'k-', lw=2, label='Exact')
ax1.plot(xv4, Tv4, 'o-', color=BLUE, lw=2, markersize=8, label=f'FEM ($n={n_coarse}$)')
# mark nodal errors with vertical lines
for xi, Ti_fem in zip(xv4[1:-1], Tv4[1:-1]):
    Ti_ex = T_exact_reaction(xi)
    ax1.plot([xi, xi], [Ti_fem, Ti_ex], color=RED, lw=1.5, zorder=5)
    ax1.plot(xi, Ti_ex, 'x', color=RED, markersize=8, markeredgewidth=2, zorder=6)
ax1.plot([], [], color=RED, lw=1.5, label='Nodal error')
ax1.set_xlabel('Position $x$ (m)')
ax1.set_ylabel('Temperature $T$')
ax1.set_title(r'Coarse mesh ($n=4$): nodes miss exact solution')
ax1.legend(fontsize=8.5)

# Panel B: convergence comparison β=0 vs β>0
n_list = [2, 4, 6, 8, 12, 16, 24, 32]

# Pure diffusion (β=0): nodal error should be machine eps
err_pure = []
for n in n_list:
    nn = n + 1
    hv = L_r / n
    xv = np.linspace(0, L_r, nn)
    Ke = (k_r / hv) * np.array([[1., -1.], [-1., 1.]])
    Fe = (Q_r * hv / 2.) * np.array([1., 1.])
    K = np.zeros((nn, nn)); F = np.zeros(nn)
    for i in range(n):
        K[i:i+2, i:i+2] += Ke; F[i:i+2] += Fe
    K[0,:]=0.; K[0,0]=1.; F[0]=0.
    K[-1,:]=0.; K[-1,-1]=1.; F[-1]=0.
    Tv = np.linalg.solve(K, F)
    # Exact for pure diffusion: T = Q/(2k) * x*(L-x)
    T_ex = (Q_r / (2 * k_r)) * xv * (L_r - xv)
    err_pure.append(np.max(np.abs(Tv - T_ex)))

# Reaction (β>0)
err_reaction = []
for n in n_list:
    xv, Tv = fem_reaction(n)
    err_reaction.append(np.max(np.abs(Tv - T_exact_reaction(xv))))

h_list = L_r / np.array(n_list)

ax2.loglog(h_list, err_pure,     'o--', color=GREEN,  lw=2, markersize=6,
           label=r'Pure diffusion ($\beta=0$): exact at nodes')
ax2.loglog(h_list, err_reaction, 's-',  color=RED,    lw=2, markersize=6,
           label=r'Reaction ($\beta=20$): $O(h^2)$ error')

# O(h^2) reference line anchored to first reaction error
ref_h = np.array([h_list[0], h_list[-1]])
ref_e = err_reaction[0] * (ref_h / h_list[0])**2
ax2.loglog(ref_h, ref_e, ':', color='grey', lw=1.5, label='$O(h^2)$ slope')

ax2.invert_xaxis()
ax2.set_xlabel('Element size $h$')
ax2.set_ylabel('Max nodal error')
ax2.set_title('Nodal exactness is lost with reaction term')
ax2.legend(fontsize=8.5)

fig.suptitle(
    r'Adding $\beta T$ term: $-kT^{\prime\prime} + \beta T = Q$ breaks the exactness proof  ($\beta=20$)',
    fontsize=11.5, y=1.02)
plt.tight_layout()
save('fig_reaction_term')

print('\nAll figures saved to figures/')
