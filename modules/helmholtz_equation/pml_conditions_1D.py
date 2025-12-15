#%%
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass

def helmholtz_1D_pml_solution(domain, frequency, velocity, source, alpha):
    """
    Solve the Helmholtz equation with PML boundary conditions.

    Parameters
    ----------
    domain: object
        with attributes:
            nx: int
                number of grid points in the x direction
            x_array: array
                array of x coordinates
            dx: float
                grid spacing in the x direction
            points: array
                global enumeration of the grid
            nk: int
                number of grid points
            nbl: int
                number of boundary layers
            lpml: float
                thickness of the PML layer
    frequency: float
        frequency of the wave
    velocity: array
        velocity model
    source: function
        source function
    alpha: float
        PML parameter

    Returns
    -------
    u: array
        solution array
    b: array
        right side array
    """
    nx = domain.nx
    x_array = domain.x_array
    dx = domain.dx
    points = domain.points
    nk = domain.nk
    nbl = domain.nbl
    lpml = domain.lpml

    data = []
    row = []
    col = []

    b = np.zeros(nk, dtype=complex)

    omega = 2*np.pi*frequency
    
    sigma_x = lambda x, ax: 2*np.pi*alpha*frequency*((abs(x_array[x])-abs(x_array[nbl]))*ax/lpml)**2

    tA = lambda x, ax: ((1)/(1-1j*sigma_x(x, ax)/omega))
    tC = lambda x, ax: ((1)*(1-1j*sigma_x(x, ax)/omega)*(omega/velocity[x])**2)

    for k in range(nk):
        i = k

        if k in points:
            if (i == 0 or i == nx-1):
                data.append(1)
                row.append(k)
                col.append(k)
                continue

            elif ((i<nbl) or (i>nx-nbl-1)):
                pml_x = 1
                
            else:
                pml_x = 0

            cA = - 2/(dx**2)*tA(i, pml_x) + tC(i, pml_x)
            cB = - 1/(4*dx**2)*tA(i-1, pml_x) + 1/(dx**2)*tA(i, pml_x) + 1/(4*dx**2)*tA(i+1, pml_x)
            cC =   1/(4*dx**2)*tA(i-1, pml_x) + 1/(dx**2)*tA(i, pml_x) - 1/(4*dx**2)*tA(i+1, pml_x)

            b[k] = source(x_array[i])

            data.append(cA)
            row.append(k)
            col.append(k)
            
            data.append(cB)
            row.append(k)
            col.append(i+1)
            
            data.append(cC)
            row.append(k)
            col.append(i-1)

        else:
            print(f'The {k} point is not in the domain points') # for debugging
            data.append(1)
            row.append(k)
            col.append(k)
        
    A = csr_matrix((data, (row, col)))
    
    U = spsolve(A,b)

    print('Solution computed')

    return U, b

#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main_domain_shape = 201
    main_domain_extension = (-0.5, 0.5)
    nbl = 100

    nx = main_domain_shape + nbl*2

    lpml = (main_domain_extension[1] - main_domain_extension[0])/(main_domain_shape-1) * nbl

    domain_shape = nx
    domain_extension = tuple(x - lpml if i % 2 == 0 else x + lpml for i, x in enumerate(main_domain_extension))

    x_array = np.linspace(domain_extension[0], domain_extension[1], nx)
    
    dx = x_array[1] - x_array[0]

    nk = nx
    points_number = np.arange(nk)
    
    frequency = 5
    velocity = 1.5

    velocity_array = velocity * np.ones(nx)
    velocity_array[ nx//2 : ] = 3

    alpha = 4

    source = lambda x: 1 * np.exp(-((x - (-0.2))**2) / (2 * 0.001**2))*np.exp(1j*np.pi/6)

    u_max = np.abs(np.trapezoid(source(x_array), x_array)*(1/(2*2*np.pi*frequency/velocity)))

    domain = dataclass(type('Domain', (), {
        'nx': nx,
        'x_array': x_array,
        'dx': dx,
        'points': points_number,
        'nk': nk,
        'nbl': nbl,
        'lpml': lpml
    }))

    u, b = helmholtz_1D_pml_solution(domain, frequency, velocity_array, source, alpha)

    plt.figure(figsize=(10, 6))
    plt.plot(x_array, np.real(u)/u_max, label='Real part of u', color='blue')
    plt.plot(x_array, np.imag(u)/u_max, label='Imaginary part of u', color='red')
    plt.legend()
    plt.title('Solution u')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.grid()

    plt.figure(figsize=(10, 6))
    plt.plot(x_array, np.real(b), label='Real part of the source', color='blue')
    plt.plot(x_array, np.imag(b), label='Imaginary part of the source', color='red')
    plt.legend()
    plt.title('Source')
    plt.xlabel('x')
    plt.ylabel('b')
    plt.grid()

    amplitude_u = np.abs(u)
    plt.figure(figsize=(10, 6))
    plt.plot(x_array, amplitude_u/u_max, label='Amplitude of u', color='green')
    plt.legend()
    plt.title('Amplitude of u')
    plt.xlabel('x')
    plt.ylabel('|u|')
    plt.grid()

    plt.show()
# %%
