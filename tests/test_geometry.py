import sgp.sgp
import numpy as np
import pytest

def test_geometry(N=(2**10, 2**10), L=(2, 2)):
    geo = sgp.sgp.Geometry(N, L)
    """compares the integral of x^2 + y^2 with analytic value"""
    f = (geo.x_**2 + geo.y_**2)
    vol = f.sum() * geo.dx * geo.dy
    print(vol)

    assert np.allclose(vol, (1/12.0 * (L[0]**3*L[1] + L[0]*L[1]**3)))

