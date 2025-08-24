import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def helmholtz_dirchlet_solution(nx, ny, x_array, y_array, dx, dy, points, nk, frequency, velocity, source):
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
            print('no')
            data.append(1)
            row.append(k)
            col.append(k)
        
    A = csr_matrix((data, (row, col)))
    
    U = spsolve(A,b)

    U_array_2D = np.zeros((nx, ny), dtype=complex)
    b_array_2D = np.zeros((nx, ny), dtype=complex)

    for k in range(nk):
        i = k%(nx)
        j = int(k/(nx))
        
        U_array_2D[i, j] = U[k]
        b_array_2D[i, j] = b[k]

    print('Se soluciona el campo en cartesianas')

    return U_array_2D, b_array_2D

if __name__ == "__main__":

    from sources import SinSinSource
    
    nx = 400
    ny = 400
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

    source = SinSinSource(ax=2, ay=2, wavenumber=wavenumber)

    u, b = helmholtz_dirchlet_solution(nx, ny, x_array, y_array, dx, dy, points, nk, frequency, velocity, source)

    fig, (ax0, ax1) = plt.subplots(1,2, figsize=(12, 5))
    im0 = ax0.imshow(np.real(b), extent=domain_extension, origin='lower')
    fig.colorbar(im0, ax=ax0, shrink=0.7)
    ax0.set_title('Source')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')

    im1 = ax1.imshow(np.real(u), extent=domain_extension, origin='lower')
    fig.colorbar(im1, ax=ax1, shrink=0.7)
    ax1.set_title('Field')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    fig.tight_layout()

    plt.show()