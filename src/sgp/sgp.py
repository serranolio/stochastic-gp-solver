"""
Stochastic GP solver
Author: Federico Serrano
Physics & Astronomy
Washington State University
"""

import numpy as np



class Geometry:
    """Class to define the geometric basis to help solve the stochastic
    GP equation in 2D.

    """

    def __init__(self, N=(64, 64), L=(10, 10)):
        """initialize local variables"""
        self.nx = N[0]
        self.ny = N[1]
        self.lx = L[0]
        self.ly = N[1]
        self.init()

    def init(self):
        """initialize computed variables"""
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        self.x = np.arange(self.nx) * self.dx - (self.lx - self.dx) / 2.0
        self.y = np.arange(self.ny) * self.dy - (self.ly - self.dy) / 2.0
        self.kx = 2*np.pi * np.fft.fftfreq(self.nx, self.dx)
        self.ky = 2*np.pi * np.fft.fftfreq(self.ny, self.dy)
        self.x_, self.y_ = np.meshgrid(self.x, self.y, indexing='ij')
        self.kx_, self.ky_ = np.meshgrid(self.kx, self.ky, indexing='ij')

class System:
    """Class to define the system parameters:
    g_2d: effective GP interaction constant in dimensions of
    gamma: drift parameter
    temp: effective temperature

    """

    def __init__(self, g_2d=1.0, gamma=0.03, temp=1.0, seed=42):
        """initialize system parameters"""
        self.g_2d = g_2d
        self.gamma = gamma
        self.temp = temp
        self.noise_scale = np.sqrt(2.0 * temp * gamma)
        self.rng = np.random.default_rng(seed)
        self.geometry = None
        self.dispersion = None

    def set_geometry(self, geometry):
        """set the system's geometry"""
        self.geometry = geometry

    def set_dispersion(self, fourier_dispersion):
        """set the dispersion in momentum space"""
        self.dispersion = fourier_dispersion

    def update_rng(self, rng):
        """update the random number generator"""
        self.rng = rng

    def get_kinetic(self, psi):
        """returns the kinetic operator acting on psi"""
        f_k = self.dispersion
        return np.fft.ifftn(f_k * np.fft.fftn(psi))

    def project_low_energy(self, psi, e_cut):
        """projects the function psi to the lower energy region"""
        psi_k = np.fft.fftn(psi)
        psi_k[self.dispersion >= e_cut] = 0.0
        return np.fft.ifftn(psi_k)

    def sequence_step(self, psi, dt, e_cut, mu):
        
        # non-linear interaction
        interaction = (self.g_2d * np.abs(psi)**2 - mu) * psi
        
        # drift term
        drift = -(1j + self.gamma) * (self.get_kinetic(psi) + interaction)
        
        # complex Gaussian white noise
        dW = (self.noise_scale * np.sqrt(dt
              / (self.geometry.dx * self.geometry.dy))
              * (self.rng.normal(size=psi.shape) + 1j * self.rng.normal(size=psi.shape)))

        # field update
        dpsi = drift * dt + dW
        dpsi_projected = self.project_low_energy(dpsi, e_cut)
        psi_updated = psi + dpsi_projected

        return psi_updated

    def evolve(self, 
               psi_0, 
               n_steps, 
               dt, 
               e_cut, 
               mu, 
               snapshots, 
               **kwargs):
        snapshot_idx = np.arange(snapshots) * (n_steps//(snapshots-1))
        output = []
        psi_out = psi_0.copy()
        for n in range(n_steps):
            mu_ = mu(n*dt, **kwargs)
            psi_out = self.sequence_step(psi_out, dt, e_cut, mu_)

            if n in snapshot_idx:
                output.append(psi_out)
        return output




