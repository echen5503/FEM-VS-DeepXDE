import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from config import L, k, Q, T0, TL, rho, c_p, T_init, t_end, dt

# ==========================================
# 1. Mesh Generation
# ==========================================
n_elements = 2
n_nodes = n_elements + 1
x = np.linspace(0, L, n_nodes)
h = L / n_elements

# ==========================================
# 2. Element Matrices
# ==========================================
K_e = (k / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])
M_e = (rho * c_p * h / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
F_e = (Q * h / 2.0) * np.array([1.0, 1.0])

# ==========================================
# 3. Global Assembly
# ==========================================
K_global = np.zeros((n_nodes, n_nodes))
M_global = np.zeros((n_nodes, n_nodes))
F_global = np.zeros(n_nodes)

for i in range(n_elements):
    n1, n2 = i, i + 1
    K_global[n1:n2+1, n1:n2+1] += K_e
    M_global[n1:n2+1, n1:n2+1] += M_e
    F_global[n1:n2+1] += F_e

# ==========================================
# 4. Effective System Matrix (Backward Euler)
# ==========================================
A_eff = M_global + dt * K_global
A_eff[0, :] = 0.0;  A_eff[0, 0] = 1.0
A_eff[-1, :] = 0.0; A_eff[-1, -1] = 1.0

# ==========================================
# 5. Initial Condition
# ==========================================
T = np.full(n_nodes, T_init)
T[0] = T0
T[-1] = TL

# ==========================================
# 6. Steady-State Analytical Solution
# ==========================================
x_fine = np.linspace(0, L, 300)
C2 = T0
C1 = (TL - T0 + (Q * L**2) / (2 * k)) / L
T_steady = (-Q / (2 * k)) * x_fine**2 + C1 * x_fine + C2

# ==========================================
# 7. Fourier Coefficients for Transient Analytical Solution
# ==========================================
# T(x,t) = T_steady(x) + sum_n Bn * sin(n*pi*x/L) * exp(-alpha*(n*pi/L)^2*t)
# Bn = (2/L) * integral_0^L [T_init - T_steady(x)] * sin(n*pi*x/L) dx  (numerical)
alpha = k / (rho * c_p)
N_fourier = 50

x_int = np.linspace(0, L, 1000)
T_steady_int = (-Q / (2 * k)) * x_int**2 + C1 * x_int + C2
residual = T_init - T_steady_int  # T_init - T_steady(x)

Bn = np.zeros(N_fourier)
for n in range(1, N_fourier + 1):
    integrand = residual * np.sin(n * np.pi * x_int / L)
    Bn[n - 1] = (2.0 / L) * np.trapezoid(integrand, x_int)

def T_analytical(x_eval, t):
    T = (-Q / (2 * k)) * x_eval**2 + C1 * x_eval + C2  # steady state
    for n in range(1, N_fourier + 1):
        lam = alpha * (n * np.pi / L) ** 2
        T = T + Bn[n - 1] * np.sin(n * np.pi * x_eval / L) * np.exp(-lam * t)
    return T

# ==========================================
# 8. Pre-compute FEM Time Steps
# ==========================================
n_steps = int(t_end / dt)
frames_per_step = max(1, n_steps // 100)

times = [0.0]
T_history = [T.copy()]
T_cur = T.copy()
for step in range(1, n_steps + 1):
    rhs = M_global @ T_cur + dt * F_global
    rhs[0] = T0
    rhs[-1] = TL
    T_cur = np.linalg.solve(A_eff, rhs)
    if step % frames_per_step == 0:
        T_history.append(T_cur.copy())
        times.append(step * dt)

# ==========================================
# 9. Animation
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

line_fem,  = ax.plot(x,      T_history[0],              'o-',  lw=2,   color='tab:blue',   label='FEM')
line_exact, = ax.plot(x_fine, T_analytical(x_fine, 0.0), '-',   lw=2,   color='tab:orange', label='Analytical (Fourier)')
ax.plot(x_fine, T_steady, 'k--', lw=1.5, label='Steady-state')

title = ax.set_title('t = 0 s')
ax.set_xlabel('Distance x (m)')
ax.set_ylabel('Temperature T (°C)')
ax.set_ylim(min(T_init, TL) - 5, max(T0, max(T_steady)) + 5)
ax.legend()
ax.grid(True)
fig.tight_layout()

def update(frame):
    t = times[frame]
    line_fem.set_ydata(T_history[frame])
    line_exact.set_ydata(T_analytical(x_fine, t))
    title.set_text(f't = {t:.0f} s')
    return line_fem, line_exact, title

ani = FuncAnimation(fig, update, frames=len(T_history), interval=50, blit=True)
ani.save('FEM_1D_transient_heat.gif', writer='pillow', fps=20)
plt.show()
