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
        self.ly = L[1]
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
    spin_g_factor: scattering lengths ratio a_12 / a_11 (a_11=a_22)
    gamma: drift parameter
    temp: effective temperature

    """

    def __init__(self,
                 g_2d=1.0,
                 spin_g_factor=1.05,
                 gamma=0.03,
                 temp=1.0,
                 seed=42):
        """initialize system parameters"""
        self.g = g_2d
        self.g12 = spin_g_factor
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

    def sequence_step(self, psi1, psi2, dt, e_cut, mu):
        """evolution of the field after a lapse dt"""

        # non-linear interaction
        interaction1 = (self.g * (np.abs(psi1)**2
                                  + self.g12*np.abs(psi2)**2)
                        - mu) * psi1
        interaction2 = (self.g * (np.abs(psi2)**2
                                  + self.g12*np.abs(psi1)**2)
                        - mu) * psi2
        
        # drift term
        drift1 = -(1j + self.gamma) * (self.get_kinetic(psi1) +
                                       interaction1)
        drift2 = -(1j + self.gamma) * (self.get_kinetic(psi2) +
                                       interaction2)
        
        # complex Gaussian white noise
        dW1 = (self.noise_scale * np.sqrt(dt
               / (self.geometry.dx * self.geometry.dy))
               * (self.rng.normal(size=psi1.shape)
                  + 1j * self.rng.normal(size=psi1.shape)))
        dW2 = (self.noise_scale * np.sqrt(dt
               / (self.geometry.dx * self.geometry.dy))
               * (self.rng.normal(size=psi2.shape)
                  + 1j * self.rng.normal(size=psi2.shape)))

        # field update
        dpsi1 = drift1 * dt + dW1
        dpsi2 = drift2 * dt + dW2
        dpsi1_projected = self.project_low_energy(dpsi1, e_cut)
        dpsi2_projected = self.project_low_energy(dpsi2, e_cut)
        psi1_updated = psi1 + dpsi1_projected
        psi2_updated = psi2 + dpsi2_projected


        return psi1_updated, psi2_updated

    def evolve(self, 
               psi1_0,
               psi2_0,
               n_steps, 
               dt, 
               e_cut, 
               mu, 
               snapshots, 
               **kwargs):
        """evolution of the system with a linear ramp of the chemical pot."""
        snapshot_idx = np.arange(snapshots) * (n_steps//(snapshots-1))
        output1 = []
        output2 = []
        psi1_out = psi1_0.copy()
        psi2_out = psi2_0.copy()
        for n in range(n_steps):
            mu_ = mu(n*dt, **kwargs)
            psi1_out, psi2_out  = self.sequence_step(psi1_out,
                                                     psi2_out,
                                                     dt,
                                                     e_cut,
                                                     mu_)

            if n in snapshot_idx:
                output1.append(psi1_out)
                output2.append(psi2_out)
        return output1, output2

