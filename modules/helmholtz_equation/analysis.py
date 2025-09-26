import numpy as np
import scipy as sp
from scipy.special import hankel1

def calculate_source_term(domain, frequencies, velocity, u_arrays):
    """
    Calculate the source term from the computed wavefield using the Helmholtz Operator.
    
    Parameters
    ----------
        domain: Object
            with attributes:\n
                dx: float
                    grid spacing in x direction. 
                dy: float
                    grid spacing in y direction.
        frequencies: array
            A 1D numpy array of frequencies.
        velocity: array
            A 2D numpy array representing the velocity model.
        u_arrays: array
            A 3D numpy array of shape (nx, ny, frequencies) containing the computed wavefields for each frequency.
    
    Returns
    -------
        A 3D numpy array of shape (nx, ny, frequencies) representing the calculated source terms.
    """

    laplace = sp.ndimage.laplace(u_arrays, mode='mirror', axes=(0,1))/(domain.dx*domain.dy)
    second_term = np.einsum('k, ijk->ijk', (2*np.pi*frequencies)**2,
                            np.einsum('ij, ijk->ijk',(1/velocity)**2, u_arrays))
    return laplace + second_term

def greens_function(domain, frequency, velocity, eps=1e-10):
    """
    Compute the analytical Green's function for a 2D Helmholtz equation.
    
    Parameters
    ----------
        domain: Object
            with attributes:\n
                x_array: 1D array
                    x-coordinates of the grid points.
                y_array: 1D array
                    y-coordinates of the grid points.
        frequency: float
            The wave frequency.
        velocity: float
            The velocity of propagation.
        eps: float
            A small value to avoid singularity at the source location.

    Returns
    -------
        A 2D array representing the Green's function.
    """

    k = 2 * np.pi * frequency / velocity
    X, Y = np.meshgrid(domain.x_array, domain.y_array)
    r = np.sqrt(X**2 + Y**2)
    r[domain.nx//2, domain.ny//2] = eps
    G = 1j/4 * hankel1(0, k*r)
    return G