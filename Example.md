---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# Stochastic Projected Gross-Pitaevskii (SPGP) simulation

This example uses the module sgp (stochastic Gross-Pitaevskii) to simulate the formation of a two-component BEC at finite temperature.
The package is based on the methodology described in the work from [M. Thudiyangal and A. del Campo](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.033152) which, in turn, utilized the $c$-field techniques developed by [A. S. Bradley, M. J. Davis, R. J. Ballagh and C. W. Gardiner](https://www.tandfonline.com/doi/abs/10.1080/00018730802564254).

To simulate condensation we artificially ramp the chemical potential from zero to a value $\mu_f$ above the critical point for condensation and let the system evolve until equilibrium. We employ the following SPGP:
$$
i\hbar\frac{\partial}{\partial t} \psi_j(\mathbf{r}, t) = \mathcal{P}\left\{ (1-i\gamma)\left[\left(-\frac{\hbar^2\nabla^2}{2m} + g|\psi_j(\mathbf{r}, t)|^2 + g_s|\psi_{3-j}|^2 - \mu(t)\right)\psi_j\right] + \eta_j(\mathbf{r}, t)\right\}
$$

where:
 - $\psi_j(\mathbf{r}, t)$ is the wave-function of the $j$-component.
 - $\mathcal{P}$ is a low energy projector that removes energies above a cutoff $\Lambda=k_B T \ln{2} + \mu_f$.
 - $T$ quantifies the thermal fluctuations.
 - $\eta_j(\mathbf{r}, t)$ is a normally distributed random variable with correlation:
 $$\langle \eta_i^*(\mathbf{r}, t)\eta_j(\mathbf{r}', t')\rangle = 2\hbar\gamma k_B T \delta(\mathbf{r}-\mathbf{r}')\delta(t -t')\delta_{ij}$$
 - $\gamma$ is the dissipation coefficient.
 - $g$/$g_s$ is the self/inter-species interaction strength.

The time-dependent chemical potential is set by the linear ramp
$$
\mu(t) = \mu_f \frac{(t - t_r)}{\tau}
$$
for $t \in [t_r, t_r + \tau]$ where $\tau$ is the ramp time and $t_r$ is when the ramp starts.
```{code-cell} python3
# Chemical potential schedule
def mu_fun(t, tau, mu_i=0.1, mu_f=20.0, t_r=10):
    return (mu_i * (t <= t_r)
            + (mu_i + (mu_f - mu_i) * (t - t_r) / tau) * (t > t_r) * (t <= (t_r + tau))
            + mu_f * (t > (t_r + tau)))
```
## 0. Packages and layout
```{code-cell} python3
import src.sgp.sgp as sgp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'axes.labelsize': 12,   # Font size for axis labels
    'font.size': 12,        # General font size
    'legend.fontsize': 12,  # Font size for legend
    'xtick.labelsize': 12,  # Font size for x-axis tick labels
    'ytick.labelsize': 12   # Font size for y-axis tick labels
})
plt.rcParams["figure.figsize"] = (4, 3)
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb, bm}'
```
## 1. System setup
```{code-cell} python3
nx = 2**7
ny = 2**7

lx = 30
ly = 30

dt = 2e-4

geo = sgp.Geometry(N=(nx, ny), L=(lx, ly))
dispersion = (geo.kx_**2 + geo.ky_**2) / 2

syst = sgp.System(g_2d=1.0,             # self-interaction strength
                  spin_g_factor=0.5,    # inter-species/self interaction ratio
                  gamma=0.03,           # dissipation coefficient
                  temp=0.2,             # temperature
                  )

syst.set_geometry(geo)
syst.set_dispersion(dispersion)
```
## 2. Evolution

```{code-cell} python3
try:
    psi1_t = np.load('psi1_miscible_sample_low_temp.npy')
    psi2_t = np.load('psi2_miscible_sample_low_temp.npy')

except FileNotFoundError:

    psi1_t, psi2_t = syst.evolve(psi1_0=np.zeros((nx, ny), dtype='complex'),
                                 psi2_0=np.zeros((nx, ny), dtype='complex'),
                                 n_steps=550000,
                                 dt=2e-4,
                                 e_cut=50,
                                 mu=mu_fun,
                                 snapshots=500,
                                 tau=100,
                                 mu_i=0.1,
                                 mu_f=20.0,
                                 t_r=10)
```
## 3. Display

```{code-cell} python3
# Parameters
t_axis = (np.arange(500)*(550000//499))*dt
n_frames = 500
X, Y = np.meshgrid(geo.x, geo.y)

# --- Set up the figure ---
fig, ax = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True, sharey=True)

im1 = ax[0].pcolormesh(X, Y, (np.abs(psi1_t[0])**2), cmap='Spectral_r', shading='gouraud', vmin=0, vmax=20)
im2 = ax[1].pcolormesh(X, Y, np.angle(psi1_t[0]), cmap='Spectral_r', shading='gouraud')

ax[0].set_xlabel(r"$x$")
ax[0].set_ylabel(r"$y$")
ax[1].set_xlabel(r"$x$")
# ax[1].set_ylabel(r"$y$")
ax[0].set_title(r"$|\psi_i|^2 (t/\tau=%.4f)$" % (t_axis[0]/100))
ax[1].set_title(r"$\arg{\psi_j}$")

# --- Update function ---
def update(frame):
    im1.set_array((np.abs(psi1_t[frame])**2).ravel())
    im2.set_array(np.angle(psi1_t[frame]).ravel())
    ax[0].set_title(r"$t/\tau_Q=%.4f$" % (t_axis[frame]/100))
    return im1, im2

# --- Animation ---
anim = FuncAnimation(fig, update, frames=n_frames, blit=True, interval=50)

# --- Display inline ---
HTML(anim.to_html5_video())

anim.save("output_animation.mp4", fps=20, dpi=300)


```
