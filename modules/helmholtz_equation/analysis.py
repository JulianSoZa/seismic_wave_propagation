import numpy as np
import scipy as sp

def calculate_source_term(domain, frequencies, velocity, u_arrays):
    """
    Calculate the source term from the computed wavefield using the Helmholtz Operator.
    
    Parameters
    ----------
    domain: Object
        with attributes:
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
    
    Returns:
        A 3D numpy array of shape (nx, ny, frequencies) representing the calculated source terms.
    """

    laplace = sp.ndimage.laplace(u_arrays, mode='mirror', axes=(0,1))/(domain.dx*domain.dy)
    second_term = np.einsum('k, ijk->ijk', (2*np.pi*frequencies)**2,
                            np.einsum('ij, ijk->ijk',(1/velocity)**2, u_arrays))
    return laplace + second_term