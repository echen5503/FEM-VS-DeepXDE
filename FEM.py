import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Define Parameters
# ==========================================
L = 1.0           # Length of the domain (m)
k = 50.0          # Thermal conductivity (W/m.K)
Q = -100000.0        # Uniform heat source (W/m^3)
T0 = 100.0        # Boundary condition: Temperature at x = 0 (C)
TL = 50.0         # Boundary condition: Temperature at x = L (C)
n_elements = 3   # Number of finite elements
n_nodes_true = 100

# ==========================================
# 2. Mesh Generation
# ==========================================
n_nodes = n_elements + 1
x = np.linspace(0, L, n_nodes)
h = L / n_elements  # Uniform element length

# ==========================================
# 3. Global Matrix Initialization
# ==========================================
K_global = np.zeros((n_nodes, n_nodes))
F_global = np.zeros(n_nodes)

# Element stiffness matrix and force vector (linear elements)
# K_e = (k/h) * [ 1 -1]
#               [-1  1]
K_e = (k / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])

# F_e = (Q*h/2) * [1]
#                 [1]
F_e = (Q * h / 2.0) * np.array([1.0, 1.0])

# ==========================================
# 4. Assembly Process
# ==========================================
for i in range(n_elements):
    # Global node indices for the current element
    n1 = i
    n2 = i + 1
    
    # Add element contributions to the global matrices
    K_global[n1:n2+1, n1:n2+1] += K_e
    F_global[n1:n2+1] += F_e

# ==========================================
# 5. Apply Boundary Conditions (Dirichlet)
# ==========================================
# Modify the first row for T(0) = T0
K_global[0, :] = 0.0
K_global[0, 0] = 1.0
F_global[0] = T0

# Modify the last row for T(L) = TL
K_global[-1, :] = 0.0
K_global[-1, -1] = 1.0
F_global[-1] = TL

# ==========================================
# 6. Solve the System
# ==========================================
# Solve the linear system K * T = F
T_fem = np.linalg.solve(K_global, F_global)

# ==========================================
# 7. Analytical Solution (for validation)
# ==========================================
# T(x) = (-Q / (2k)) * x^2 + C1 * x + C2
C2 = T0
C1 = (TL - T0 + (Q * L**2) / (2 * k)) / L
x_true = np.linspace(0, L, n_nodes_true)

T_exact = (-Q / (2 * k)) * x_true**2 + C1 * x_true + C2

# ==========================================
# 8. Plot Results
# ==========================================
plt.figure(figsize=(8, 5))
plt.plot(x, T_fem, 'o-', linewidth=2, label='FEM Solution (2 elements)')
plt.plot(x_true, T_exact, '--', linewidth=2, label='Analytical Solution')
plt.xlabel('Distance $x$ (m)')
plt.ylabel('Temperature $T$ (°C)')
plt.title('1D Steady-State Heat Conduction')
plt.legend()
plt.grid(True)
plt.savefig('FEM_1D_heat_conduction.png')