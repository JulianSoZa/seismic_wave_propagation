#%% Libreries
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

#%% Solver function
def helmholtz_dirchlet_solution(nx, ny, x_array, y_array, dx, dy, points, nk, frequency, velocity, source):
    """Solve the 2D Helmholtz equation with Dirichlet boundary conditions using finite differences.

    Parameters
    ----------
    nx : int
        Number of grid points in the x direction.
    ny : int
        Number of grid points in the y direction.
    x_array : numpy.ndarray
        1D array of x coordinates.
    y_array : numpy.ndarray
        1D array of y coordinates.
    dx : float
        Grid spacing in the x direction.
    dy : float
        Grid spacing in the y direction.
    points : numpy.ndarray
        Array of grid point indices where the equation is solved.
    nk : int
        Total number of grid points (nx * ny).
    frequency : float
        Frequency of the wave.
    velocity : float
        Wave propagation velocity.
    source : callable
        Function defining the source term, should take x and y as inputs.
    
    Returns
    -------
    u_array_2D : numpy.ndarray
        2D array of the computed wavefield solution.
    b_array_2D : numpy.ndarray
        2D array of the source term.
    
    Examples
    --------
    >>> import numpy as np
    >>> nx, ny = 200, 200
    >>> x_array = np.linspace(-1, 1, nx)
    >>> y_array = np.linspace(-1, 1, ny)
    >>> dx = x_array[1] - x_array[0]
    >>> dy = y_array[1] - y_array[0]
    >>> nk = nx * ny
    >>> points = np.arange(nk)
    >>> frequency = 1
    >>> velocity = 2 * np.pi
    >>> source = lambda x, y: (-(2 * np.pi)**2 - (2 * np.pi)**2 + 1) * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
    >>> u, b = helmholtz_dirchlet_solution(nx, ny, x_array, y_array, dx, dy, points, nk, frequency, velocity, source)
    """
    data = []
    row = []
    col = []

    b = np.zeros(nk, dtype=complex)
    
    omega = 2*np.pi*frequency
    wavenumber = omega/velocity

    num = lambda x,y: nx*y+x
    
    cA = -2/dx**2 - 2/dy**2 + wavenumber**2
    cB = 1/dx**2
    cC = 1/dx**2
    cD = 1/dy**2
    cE = 1/dy**2

    for k in range(nk):
        i = k%(nx)
        j = int(k/(nx))
        
        if k in points:
            if (i == 0 or i == nx-1 or j == 0 or j == ny-1):
                data.append(1)
                row.append(k)
                col.append(k)

            else:
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
    # Parameters
    nx, ny = 500, 500
    domain_extension = (-1, 1, -1, 1)
    
    frequency = 1
    velocity = 2*np.pi
    wavenumber = 2*np.pi*frequency/velocity

    x_array = np.linspace(domain_extension[0], domain_extension[1], nx)
    y_array = np.linspace(domain_extension[2], domain_extension[3], ny)
    
    dx = x_array[1] - x_array[0]
    dy = y_array[1] - y_array[0]
    nk = nx * ny
    points = np.arange(nk)

    ax, ay = 2, 2
    source = lambda x, y: (-(ax * np.pi)**2 - (ay * np.pi)**2 + wavenumber**2) * np.sin(ax * np.pi * x) * np.sin(ay * np.pi * y)

    u, b = helmholtz_dirchlet_solution(nx, ny, x_array, y_array, dx, dy, points, nk, frequency, velocity, source)
    
    #%% Plotting
    from matplotlib.colors import LinearSegmentedColormap
    cmap_source = LinearSegmentedColormap.from_list('source', ['green', 'white', 'purple'])
    
    import os
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output'))

    fig, (ax0, ax1) = plt.subplots(1,2, figsize=(12, 5))
    vmax = np.max(np.abs(np.real(b)))
    im0 = ax0.imshow(np.real(b), extent=domain_extension, cmap=cmap_source, origin='lower', vmin=-vmax, vmax=vmax)
    fig.colorbar(im0, ax=ax0, shrink=0.7)
    ax0.set_title('Source (Real part)')
    ax0.set_xlabel('x')
    ax0.set_ylabel('z')

    vmax = np.max(np.abs(np.real(u)))
    im1 = ax1.imshow(np.real(u), extent=domain_extension, cmap='seismic', origin='lower', vmin=-vmax, vmax=vmax)
    fig.colorbar(im1, ax=ax1, shrink=0.7)
    ax1.set_title('Numerical Solution (Real part)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('z')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'helmholtz_dirichlet_numerical_solution.png'), dpi=300)

    #%% Analysis
    X, Y = np.meshgrid(x_array, y_array)
    u_analytical = np.sin(ax * np.pi * X) * np.sin(ay * np.pi * Y)

    fig2, (ax0, ax1) = plt.subplots(1,2, figsize=(12, 5))
    vmax = np.max(np.abs(np.real(u_analytical)))
    im0 = ax0.imshow(np.real(u_analytical), extent=domain_extension, cmap='seismic', origin='lower', vmin=-vmax, vmax=vmax)
    fig2.colorbar(im0, ax=ax0, shrink=0.7)
    ax0.set_title('Analytical Solution (Real part)')
    ax0.set_xlabel('x')
    ax0.set_ylabel('z')

    error_norm = np.linalg.norm(u - u_analytical)/np.linalg.norm(u_analytical)
    print(f'Relative error: {error_norm:.2e}')

    diff = np.abs(u - u_analytical)
    vmax = np.max(diff)
    im1 = ax1.imshow(diff, extent=domain_extension, cmap='gray', origin='lower')
    fig2.colorbar(im1, ax=ax1, shrink=0.7)
    ax1.set_title('Difference |Numerical - Analytical|')
    ax1.set_xlabel('x')
    ax1.set_ylabel('z')

    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'helmholtz_dirichlet_analytical_error.png'), dpi=300)

    plt.show()