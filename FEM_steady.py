import numpy as np
import matplotlib.pyplot as plt
from config import L, k, Q, T0, TL

# ==========================================
# 1. Mesh Generation
# ==========================================
n_elements = 10
n_nodes = n_elements + 1
x = np.linspace(0, L, n_nodes)
h = L / n_elements

# ==========================================
# 2. Element Matrices
# ==========================================
K_e = (k / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])
F_e = (Q * h / 2.0) * np.array([1.0, 1.0])

# ==========================================
# 3. Global Assembly
# ==========================================
K_global = np.zeros((n_nodes, n_nodes))
F_global = np.zeros(n_nodes)

for i in range(n_elements):
    n1, n2 = i, i + 1
    K_global[n1:n2+1, n1:n2+1] += K_e
    F_global[n1:n2+1] += F_e

# ==========================================
# 4. Apply Dirichlet Boundary Conditions
# ==========================================
K_global[0, :] = 0.0;  K_global[0, 0] = 1.0;  F_global[0] = T0
K_global[-1, :] = 0.0; K_global[-1, -1] = 1.0; F_global[-1] = TL

# ==========================================
# 5. Solve
# ==========================================
T_fem = np.linalg.solve(K_global, F_global)

# ==========================================
# 6. Analytical Solution
# ==========================================
# k*T'' + Q = 0  =>  T(x) = -Q/(2k) x^2 + C1*x + C2
x_fine = np.linspace(0, L, 300)
C2 = T0
C1 = (TL - T0 + (Q * L**2) / (2 * k)) / L
T_exact = (-Q / (2 * k)) * x_fine**2 + C1 * x_fine + C2

# ==========================================
# 7. Plot
# ==========================================
plt.figure(figsize=(9, 5))
plt.plot(x_fine, T_exact, 'k--', lw=2, label='Analytical')
plt.plot(x, T_fem, 'o-', lw=2, color='tab:blue', label=f'FEM ({n_elements} elements)')
plt.xlabel('Distance x (m)')
plt.ylabel('Temperature T (°C)')
plt.title('1D Steady-State Heat Conduction — FEM vs Analytical')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('FEM_1D_steady.png', dpi=150)
plt.show()
