import sgp.sgp
import numpy as np
import pytest

def test_kinetic(N=(2**10, 2**10), L=(2, 2), kx=3*np.pi, ky=4*np.pi):
    geo = sgp.sgp.Geometry(N, L)
    system = sgp.sgp.System()
    system.set_geometry(geo)
    system.set_dispersion(geo.kx_**2 + geo.ky_**2)

    plane_wave = np.exp(1j*kx*geo.x_)*np.exp(1j*ky*geo.y_)
    norm = (np.abs(plane_wave)**2 * geo.dx * geo.dy).sum()
    plane_wave = plane_wave / np.sqrt(norm)

    assert np.allclose((np.abs(plane_wave)**2 * geo.dx * geo.dy).sum(), 1.0)
    exp_T = (plane_wave.conj() * 
             system.get_kinetic(plane_wave)).real.sum() * geo.dx * geo.dy

    assert np.allclose(exp_T, kx**2 + ky**2)


def test_project_low_energy(N=(2**10, 2**10), 
                            L=(2*np.pi, 2*np.pi),
                            kx=(5, 100),
                            ky=(4, 150),
                            e_cut=50):

    geo = sgp.sgp.Geometry(N, L)
    system = sgp.sgp.System()
    system.set_geometry(geo)
    system.set_dispersion(geo.kx_**2 + geo.ky_**2)

    plane_wave1 = (np.exp(1j*kx[0]*geo.x_)
                   * np.exp(1j*ky[0]*geo.y_)) / np.sqrt(geo.lx * geo.ly)
    plane_wave2 = (np.exp(1j*kx[1]*geo.x_)
                   * np.exp(1j*ky[1]*geo.y_)) / np.sqrt(geo.lx * geo.ly)
    norm = (np.abs(plane_wave1 + plane_wave2)**2 * geo.dx * geo.dy).sum()
    plane_wave = (plane_wave1 + plane_wave2) / np.sqrt(norm)
    assert np.allclose((np.abs(plane_wave)**2 * geo.dx * geo.dy).sum(), 1.0)

    low_energy_field = system.project_low_energy(plane_wave, e_cut)
    low_norm = ((np.abs(low_energy_field)**2 * geo.dx * geo.dy).sum())
    wave1_norm = ((np.abs(plane_wave1/np.sqrt(2))**2 * geo.dx * geo.dy).sum())
    assert np.allclose(low_norm, wave1_norm)

def test_non_discipative_evolution(N=(2**7, 2**7),
                                   L=(2*np.pi, 2*np.pi),
                                   dt=1e-4,
                                   n_steps=1000):

    geo = sgp.sgp.Geometry(N, L)
    system = sgp.sgp.System(g_2d=1.0, 
                            spin_g_factor=1.0, 
                            gamma=0.0, 
                            temp=0.0)
    system.set_geometry(geo)
    system.set_dispersion(geo.kx_**2 + geo.ky_**2)

    psi1_initial = (np.ones((geo.nx, geo.ny), dtype='complex')
                    / np.sqrt(2 * geo.lx * geo.ly))
    psi2_initial = (np.ones((geo.nx, geo.ny), dtype='complex')
                    / np.sqrt(2 * geo.lx * geo.ly))

    def mu_fun(t):
        return 0
    psi1_t, psi2_t = system.evolve(psi1_0=psi1_initial,
                                   psi2_0=psi2_initial,
                                   n_steps=n_steps,
                                   dt=dt,
                                   e_cut=np.inf,
                                   mu=mu_fun,
                                   snapshots=2)
    # check unitary evolution
    norm_i = (np.abs(psi1_t[0])**2
              + np.abs(psi2_t[0])**2).sum() * geo.dx * geo.dy 
    norm_f = (np.abs(psi1_t[-1])**2
              + np.abs(psi2_t[-1])**2).sum() * geo.dx * geo.dy

    assert np.allclose(norm_i, norm_f)

    # check eigen-state evolution
    assert np.allclose(psi1_t[-1], psi1_t[0])
    assert np.allclose(psi2_t[-1], psi2_t[0])

def test_discipative_evolution(N=(2**7, 2**7),
                               L=(2*np.pi, 2*np.pi),
                               dt=1e-4,
                               n_steps=1000):

    geo = sgp.sgp.Geometry(N, L)
    system = sgp.sgp.System(g_2d=1.0,
                            spin_g_factor=1.0,
                            gamma=1000,
                            temp=0)
    system.set_geometry(geo)
    system.set_dispersion(geo.kx_**2 + geo.ky_**2)

    psi1_initial = (np.ones((geo.nx, geo.ny), dtype='complex')                  
                    / np.sqrt(2 * geo.lx * geo.ly))
    psi2_initial = (np.ones((geo.nx, geo.ny), dtype='complex')                  
                    / np.sqrt(2 * geo.lx * geo.ly))

    def mu_fun(t):
        return 0
    psi1_t, psi2_t = system.evolve(psi1_0=psi1_initial,
                                   psi2_0=psi2_initial,
                                   n_steps=n_steps,
                                   dt=dt,
                                   e_cut=np.inf,
                                   mu=mu_fun,
                                   snapshots=2)

    norm_i = (np.abs(psi1_initial)**2                                              
              + np.abs(psi2_initial)**2).sum() * geo.dx * geo.dy                   
    norm_f = (np.abs(psi1_t[-1])**2                                             
              + np.abs(psi2_t[-1])**2).sum() * geo.dx * geo.dy

    decay_rate = 2 * system.gamma * system.g / (geo.lx * geo.ly)

    # check non-zero initial state
    print(np.exp(-decay_rate*dt))
    print(norm_f)
    assert np.allclose(norm_i, 1.0)
    # check zero-final state
    assert np.allclose(norm_f, 
                       np.exp(-decay_rate*dt), 
                       atol=1e-5, rtol=1e-5)





