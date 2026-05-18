"""
Animation: Adam gradient descent solving the inverse problem.
Recovers conductivity k from sparse noisy sensor measurements.
Saves to figures/fig_inverse_problem.gif
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from config import L, k as k_true, Q, T0, TL

BLUE   = '#1565C0'
ORANGE = '#E65100'
RED    = '#C62828'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# ── Analytical solution (acts as the PINN prediction) ────────────────────────
def T_pred(x, k):
    C1 = (TL - T0 + Q * L**2 / (2 * k)) / L
    return -(Q / (2 * k)) * x**2 + C1 * x + T0

def dT_pred_dk(x, k):
    return Q * x * (x - L) / (2 * k**2)

# ── Sensor data ───────────────────────────────────────────────────────────────
np.random.seed(7)
n_sensors = 8
x_sens = np.sort(np.random.uniform(0.05, 0.95, n_sensors))
T_sens = T_pred(x_sens, k_true) + np.random.randn(n_sensors) * 1.5

def loss(k):
    return float(np.mean((T_pred(x_sens, k) - T_sens) ** 2))

def grad(k):
    res = T_pred(x_sens, k) - T_sens
    return float(2 * np.mean(res * dT_pred_dk(x_sens, k)))

# ── Adam optimiser ────────────────────────────────────────────────────────────
k_init  = 15.0
lr_adam = 2.0
beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
n_iters = 1000

k_hist    = [k_init]
loss_hist = [loss(k_init)]
m_adam, v_adam, k_cur = 0.0, 0.0, k_init

for t in range(1, n_iters + 1):
    g = grad(k_cur)
    m_adam = beta1 * m_adam + (1 - beta1) * g
    v_adam = beta2 * v_adam + (1 - beta2) * g ** 2
    m_hat  = m_adam / (1 - beta1 ** t)
    v_hat  = v_adam / (1 - beta2 ** t)
    k_cur  = float(np.clip(k_cur - lr_adam * m_hat / (np.sqrt(v_hat) + eps_adam), 5.0, 300.0))
    k_hist.append(k_cur)
    loss_hist.append(loss(k_cur))

k_hist    = np.array(k_hist)
loss_hist = np.array(loss_hist)
print(f"Converged to k = {k_hist[-1]:.2f}  (true k = {k_true})")

# ── Frame selection: log-spaced so fast early motion is visible ───────────────
n_frames  = 120
raw_idx   = np.logspace(0, np.log10(n_iters), n_frames)
frame_idx = np.unique(np.clip(raw_idx.astype(int), 0, n_iters))

# ── Precompute y-limits ───────────────────────────────────────────────────────
x_fine = np.linspace(0, L, 400)
T_true = T_pred(x_fine, k_true)
T_init = T_pred(x_fine, k_init)
y_lo   = min(TL, T_sens.min()) - 15
y_hi   = T_init.max() + 20

# ── Build figure ──────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel A: data fit
ax1.plot(x_fine, T_true, 'k--', lw=2, label=f'True  ($k = {k_true:.0f}$)', zorder=4)
ax1.plot(x_sens, T_sens, 'o', color=ORANGE, markersize=9, zorder=5,
         label='Sensor data (noisy)')
pred_line, = ax1.plot(x_fine, T_init, '-', color=BLUE, lw=2.5,
                      label=f'Prediction  ($k = {k_init:.1f}$)')
ax1.set_xlim(0, L)
ax1.set_ylim(y_lo, y_hi)
ax1.set_xlabel('Position $x$ (m)')
ax1.set_ylabel('Temperature $T$ (°C)')
ax1.set_title('Current Prediction vs Sensor Data')
legend1 = ax1.legend(fontsize=9, loc='upper right')

info_box = ax1.text(
    0.03, 0.97,
    f'Iter 0  |  $k = {k_init:.1f}$  |  Loss = {loss_hist[0]:.0f} °C²',
    transform=ax1.transAxes, va='top', fontsize=9.5,
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='grey'))

# Panel B: k convergence + loss
ax2.axhline(k_true, color='black', ls='--', lw=1.8,
            label=f'True $k = {k_true:.0f}$ W/m·K', zorder=4)
k_line, = ax2.plot([], [], color=BLUE, lw=2.2, label='Estimated $k$')
ax2.set_xlim(0, n_iters)
ax2b.set_xlim(0, n_iters)
ax2.set_ylim(k_init * 0.8, k_true * 1.15)
ax2.set_xlabel('Adam iteration')
ax2.set_ylabel('Conductivity $k$ (W/m·K)')
ax2.set_title('Parameter Convergence')
ax2.legend(fontsize=9, loc='lower right')

ax2b = ax2.twinx()
ax2b.set_yscale('log')
loss_line, = ax2b.plot([], [], color=RED, lw=1.5, ls=':', alpha=0.8, label='Loss')
ax2b.set_ylim(max(loss_hist[-1] * 0.5, 0.1), loss_hist[0] * 3)
ax2b.set_ylabel('Data loss (log, °C²)', color=RED)
ax2b.tick_params(axis='y', colors=RED)
ax2b.legend(fontsize=9, loc='upper right')

fig.suptitle(
    r'Inverse Problem: recovering $k$ via $\nabla_k\,\mathcal{L}_{data}$ (Adam)',
    fontsize=12)
fig.tight_layout()

# ── Animation ─────────────────────────────────────────────────────────────────
def update(fi):
    i   = int(frame_idx[fi])
    k_i = k_hist[i]
    l_i = loss_hist[i]

    pred_line.set_ydata(T_pred(x_fine, k_i))
    legend1.get_texts()[2].set_text(f'Prediction  ($k = {k_i:.1f}$)')
    info_box.set_text(
        f'Iter {i}  |  $k = {k_i:.1f}$  |  Loss = {l_i:.1f} °C²')

    iters = np.arange(i + 1)
    k_line.set_data(iters, k_hist[:i + 1])
    loss_line.set_data(iters, loss_hist[:i + 1])

    return pred_line, info_box, k_line, loss_line

ani = FuncAnimation(fig, update, frames=len(frame_idx), interval=60, blit=True)
ani.save('figures/fig_inverse_problem.gif', writer='pillow', fps=15)
print('Saved figures/fig_inverse_problem.gif')
plt.show()
