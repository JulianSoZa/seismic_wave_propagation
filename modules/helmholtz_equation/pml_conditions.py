#%% Libraries
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

#%% Solver function
def helmholtz_pml_solution(domain, frequency, velocity, source, alpha):
    """
    Solve the Helmholtz equation with PML boundary conditions.

    Parameters
    ----------
    domain: object
        with attributes:
            nx: int
                number of grid points in the x direction
            ny: int
                number of grid points in the y direction
            x_array: array
                array of x coordinates
            y_array: array
                array of y coordinates
            dx: float
                grid spacing in the x direction
            dy: float
                grid spacing in the y direction
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
    ny = domain.ny
    x_array = domain.x_array
    y_array = domain.y_array
    dx = domain.dx
    dy = domain.dy
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
    sigma_y = lambda y, ay: 2*np.pi*alpha*frequency*((abs(y_array[y])-abs(y_array[nbl]))*ay/lpml)**2

    tA = lambda x, y, ax, ay: ((1-1j*sigma_y(y, ay)/omega)/(1-1j*sigma_x(x, ax)/omega))
    tB = lambda x, y, ax, ay: ((1-1j*sigma_x(x, ax)/omega)/(1-1j*sigma_y(y, ay)/omega))
    tC = lambda x, y, ax, ay: ((1-1j*sigma_y(y, ay)/omega)*(1-1j*sigma_x(x, ax)/omega)*(omega/velocity[x, y])**2)

    num = lambda x,y: nx*y+x

    for k in range(nk):
        i = k%(nx)
        j = int(k/(nx))

        if k in points:
            if (i == 0 or i == nx-1 or j == 0 or j == ny-1):
                data.append(1)
                row.append(k)
                col.append(k)
                continue

            elif ((i<nbl and j<nbl) or (i<nbl and j>ny-nbl-1) or (i>nx-nbl-1 and j<nbl) or (i>nx-nbl-1 and j>ny-nbl-1)):
                pml_x = 1
                pml_y = 1

            elif ((i<nbl) or (i>nx-nbl-1)):
                pml_x = 1
                pml_y = 0

            elif ((j<nbl) or (j>ny-nbl-1)):
                pml_x = 0
                pml_y = 1
                
            else:
                pml_x = 0
                pml_y = 0
                
            cA = - 2/(dx**2)*tA(i, j, pml_x, pml_y) - 2/(dy**2)*tB(i, j, pml_x, pml_y) + tC(i, j, pml_x, pml_y)
            cB = - 1/(4*dx**2)*tA(i-1, j, pml_x, pml_y) + 1/(dx**2)*tA(i, j, pml_x, pml_y) + 1/(4*dx**2)*tA(i+1, j, pml_x, pml_y)
            cC =   1/(4*dx**2)*tA(i-1, j, pml_x, pml_y) + 1/(dx**2)*tA(i, j, pml_x, pml_y) - 1/(4*dx**2)*tA(i+1, j, pml_x, pml_y)
            cD = - 1/(4*dy**2)*tB(i, j-1, pml_x, pml_y) + 1/(dy**2)*tB(i, j, pml_x, pml_y) + 1/(4*dy**2)*tB(i, j+1, pml_x, pml_y)
            cE =   1/(4*dy**2)*tB(i, j-1, pml_x, pml_y) + 1/(dy**2)*tB(i, j, pml_x, pml_y) - 1/(4*dy**2)*tB(i, j+1, pml_x, pml_y)

            b[k] = source(x_array[i], y_array[j])

            data.append(cA)
            row.append(k)
            col.append(k)
            
            data.append(cB)
            row.append(k)
            col.append(int(num(i+1,j)))
            
            data.append(cC)
            row.append(k)
            col.append(int(num(i-1,j)))
            
            data.append(cD)
            row.append(k)
            col.append(int(num(i,j+1)))
            
            data.append(cE)
            row.append(k)
            col.append(int(num(i,j-1)))

        else:
            # print(f'The {k} point is not part of the domain') # for debugging
            data.append(1)
            row.append(k)
            col.append(k)
        
    A = csr_matrix((data, (row, col)))
    
    U = spsolve(A,b)
    
    u_array_2D = np.zeros((nx, ny), dtype=complex)
    b_array_2D = np.zeros((nx, ny), dtype=complex)

    for k in range(nk):
        i = k%(nx)
        j = int(k/(nx))
        
        u_array_2D[i, j] = U[k]
        b_array_2D[i, j] = b[k]

    print('Solution computed')

    return u_array_2D, b_array_2D

#%% Example of use
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sources
    from domains import HelmholtzDomain
    import plotting

    # Define domain
    main_domain_shape = (201, 201)
    main_domain_extension = (-0.5, 0.5, -0.5, 0.5)
    domain = HelmholtzDomain(main_domain_shape, main_domain_extension)
    domain.pml_domain(100)
    
    # Define source
    source = sources.GaussianSource(amplitude=1, x_pos=0, y_pos=0.4, sigma=0.05, phase=0)

    # Define parameters
    frequency = 10
    velocity = 1.5

    velocity_array = np.ones((domain.nx, domain.ny))*velocity
    velocity_array[:, :(domain.ny-1)//2] = velocity*2

    alpha = 1.4

    # Solve
    u, b = helmholtz_pml_solution(domain, frequency, velocity_array, source, alpha)

#%% Plotting
    plotting.plot_solution(domain, np.array([frequency]), 0, u[..., np.newaxis], b[..., np.newaxis])
    plotting.plot_velocity(domain, velocity_array)
    plt.show()