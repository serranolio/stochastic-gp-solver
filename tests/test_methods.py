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



