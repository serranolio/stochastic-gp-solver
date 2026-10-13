# Stochastic Gross-Pitaevskii Solver

`sgp` is a small Python package for simulating the formation of a **two-component Bose-Einstein condensate (BEC) at finite temperature** in 2D, using the Stochastic Projected Gross-Pitaevskii (SPGP) equation.

The implementation follows the methodology of [M. Thudiyangal and A. del Campo, *Phys. Rev. Research* **6**, 033152 (2024)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.033152), which builds on the $c$-field techniques of [A. S. Bradley, M. J. Davis, R. J. Ballagh and C. W. Gardiner, *Adv. Phys.* **57**, 363 (2008)](https://www.tandfonline.com/doi/abs/10.1080/00018730802564254).

## Model

Condensation is driven by ramping the chemical potential from (near) zero to a final value $\mu_f$ above the critical point, then letting the system relax to equilibrium. Each component $j \in \{1, 2\}$ evolves according to

$$
i\hbar\frac{\partial \psi_j}{\partial t} = \mathcal{P}\left\{ (1-i\gamma)\left[-\frac{\hbar^2\nabla^2}{2m} + g|\psi_j|^2 + g_s|\psi_{3-j}|^2 - \mu(t)\right]\psi_j + \eta_j(\mathbf{r}, t)\right\}
$$

where

- $\psi_j(\mathbf{r}, t)$ is the classical field of component $j$;
- $\mathcal{P}$ projects onto the low-energy region below the cutoff $\Lambda = k_B T \ln 2 + \mu_f$;
- $\gamma$ is the dissipation coefficient and $T$ the temperature of the thermal reservoir;
- $g$ and $g_s$ are the intra- and inter-species interaction strengths;
- $\eta_j$ is Gaussian white noise with

$$
\langle \eta_i^*(\mathbf{r}, t)\,\eta_j(\mathbf{r}', t')\rangle = 2\hbar\gamma k_B T\,\delta(\mathbf{r}-\mathbf{r}')\,\delta(t-t')\,\delta_{ij}.
$$

The chemical potential follows a linear ramp of duration $\tau$ starting at $t_r$:

$$
\mu(t) = \mu_f\,\frac{t - t_r}{\tau}, \qquad t \in [t_r,\, t_r + \tau].
$$

By tuning the ratio $g_s/g$ the simulation can explore the **miscible**, **immiscible**, and **decoupled** regimes of the binary mixture.

## Installation

The project uses [pixi](https://pixi.sh) to manage its environment:

```bash
git clone git@github.com:serranolio/stochastic-gp-solver.git
cd stochastic-gp-solver
pixi install
```

## Usage

```python
import numpy as np
from sgp import sgp

# Chemical potential schedule: constant mu_i, linear ramp, then constant mu_f
def mu_fun(t, tau, mu_i=0.1, mu_f=20.0, t_r=10):
    return (mu_i * (t <= t_r)
            + (mu_i + (mu_f - mu_i) * (t - t_r) / tau) * (t > t_r) * (t <= t_r + tau)
            + mu_f * (t > t_r + tau))

nx, ny = 2**7, 2**7
geo = sgp.Geometry(N=(nx, ny), L=(30, 30))

syst = sgp.System(g_2d=1.0,            # self-interaction strength
                  spin_g_factor=0.5,   # inter-species / self interaction ratio
                  gamma=0.03,          # dissipation coefficient
                  temp=0.2)            # temperature
syst.set_geometry(geo)
syst.set_dispersion((geo.kx_**2 + geo.ky_**2) / 2)

psi1_t, psi2_t = syst.evolve(psi1_0=np.zeros((nx, ny), dtype=complex),
                             psi2_0=np.zeros((nx, ny), dtype=complex),
                             n_steps=550_000,
                             dt=2e-4,
                             e_cut=50,
                             mu=mu_fun,
                             snapshots=500,
                             tau=100, mu_i=0.1, mu_f=20.0, t_r=10)
```

`evolve` returns two lists of snapshots of $\psi_1$ and $\psi_2$, which can be used to plot or animate the density $|\psi_j|^2$ and phase $\arg\psi_j$. A full walkthrough, including the animation, is in [`Example.md`](Example.md).

### API overview

| Object | Description |
| --- | --- |
| `Geometry(N, L)` | Periodic 2D grid with `N=(nx, ny)` points and box size `L=(lx, ly)`; provides real-space (`x_`, `y_`) and momentum-space (`kx_`, `ky_`) meshes. |
| `System(g_2d, spin_g_factor, gamma, temp, seed)` | Physical parameters of the two-component system. |
| `System.set_geometry(geo)` / `System.set_dispersion(e_k)` | Attach the grid and the kinetic dispersion in momentum space. |
| `System.evolve(psi1_0, psi2_0, n_steps, dt, e_cut, mu, snapshots, **kwargs)` | Integrate the SPGP equations; extra `kwargs` are forwarded to `mu(t, ...)`. |

## Documentation

`Example.md` is a [MyST](https://mystmd.org) notebook. To build and browse it locally:

```bash
pixi run myst start
```

## Tests

```bash
pixi run pytest
```

## License

Released under the [MIT License](LICENSE).

## Author

Federico Serrano, Department of Physics & Astronomy, Washington State University. Developed as a project for PHYS 581 (2025).
