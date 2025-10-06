import numpy as np
import scipy as sp

def calculate_source_term(model, freq_modes, freq_array):
    laplace = sp.ndimage.laplace(freq_modes, mode='mirror', axes=(1,2))/(model.spacing[0]*model.spacing[1])
    second_term = np.einsum('i,ijk->ijk', (2*np.pi*freq_array)**2,
                            np.einsum('ij,kij->kij', (1/model.vp.data)**2, freq_modes))
    return -laplace - second_term