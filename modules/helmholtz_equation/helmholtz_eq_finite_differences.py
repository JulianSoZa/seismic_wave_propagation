import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def helmholtz_dirchlet_solution(nx, ny, x_dis, y_dis, delx, dely, puntos, nk, a1, a2, wavenumber):
    data = []
    row = []
    col = []

    b = np.zeros(nk)

    source = lambda x, y, ax, ay, wavenumber: (-(ax*np.pi)**2 -(ay*np.pi)**2 + wavenumber**2)*np.sin(ax*np.pi*x)*np.sin(ay*np.pi*y)
    #source = lambda x, y, sigma, a: a*np.exp(-((x)**2 + (y)**2) / (2 * sigma**2))

    num = lambda x,y: nx*y+x
    
    cA = -2/delx**2 - 2/dely**2 + wavenumber**2
    cB = 1/delx**2
    cC = 1/delx**2
    cD = 1/dely**2
    cE = 1/dely**2

    for k in range(nk):
        i = k%(nx)
        j = int(k/(nx))
        
        if k in puntos:
            if (i == 0 or i == nx-1 or j == 0 or j == ny-1):
                b[k] = 0

                data.append(1)
                row.append(k)
                col.append(k)

            else:
                b[k] = source(x_dis[i], y_dis[j], a1, a2, wavenumber)
                #b[k] = source(x_dis[i], y_dis[j], 0.01, 1)

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
    
    u = np.zeros((nk,3))
    U_array_2D = np.zeros((nx, ny))
    b_array_2D = np.zeros((nx, ny))

    for k in range(nk):
        i = k%(nx)
        j = int(k/(nx))
        u[k] = np.array([x_dis[i], y_dis[j], U[k]])
        U_array_2D[i, j] = U[k]
        b_array_2D[i, j] = b[k]

    print('Se soluciona el campo en cartesianas')

    return U, u, U_array_2D, b_array_2D

if __name__ == "__main__":

    nx = 400
    ny = 400
    a1 = 2
    a2 = 2
    wavenumber = 1

    x_dis = np.linspace(-1, 1, nx)
    y_dis = np.linspace(-1, 1, ny)
    delx = x_dis[1] - x_dis[0]
    dely = y_dis[1] - y_dis[0]
    nk = nx * ny
    puntosIndices = np.arange(nk)

    U_c, u_space_c, U_array_2D, b_array_2D = helmholtz_dirchlet_solution(nx, ny, x_dis, y_dis, delx, dely, puntosIndices, nk, a1, a2, wavenumber)

    fig, (ax0, ax1) = plt.subplots(1,2, figsize=(12, 5))
    im0 = ax0.imshow(b_array_2D, extent=(-1, 1, -1, 1), origin='lower')
    plt.colorbar(im0, ax=ax0, shrink=0.7)
    ax0.set_title('Source')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')

    im1 = ax1.imshow(U_array_2D, extent=(-1, 1, -1, 1), origin='lower')
    plt.colorbar(im1, ax=ax1, shrink=0.7)
    ax1.set_title('Field')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    plt.tight_layout()

    plt.show()