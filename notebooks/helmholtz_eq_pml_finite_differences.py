import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def helmholtz_dirchlet_solution(nx, ny, x_dis, y_dis, delx, dely, puntos, nk, freq, nbl, velocity, lnbl, source_amplitude, alpha):
    data = []
    row = []
    col = []

    b = np.zeros(nk, dtype=complex)

    omega = 2*np.pi*freq

    source = lambda x, y, sigma: source_amplitude*np.exp(-((x)**2 + (y)**2) / (2 * sigma**2))*np.exp(1j*np.pi/8)
    sigma_x = lambda x, ax: 2*np.pi*alpha*1.4*freq*((abs(x_dis[x])-abs(x_dis[nbl]))*ax/lnbl)**2
    sigma_y = lambda y, ay: 2*np.pi*alpha*1.4*freq*((abs(y_dis[y])-abs(y_dis[nbl]))*ay/lnbl)**2

    tA = lambda x, y, ax, ay: ((1-1j*sigma_y(y, ay)/omega)/(1-1j*sigma_x(x, ax)/omega))
    tB = lambda x, y, ax, ay: ((1-1j*sigma_x(x, ax)/omega)/(1-1j*sigma_y(y, ay)/omega))
    tC = lambda x, y, ax, ay: (((1-1j*sigma_y(y, ay)/omega))*((1-1j*sigma_x(x, ax)/omega))*(omega/velocity)**2)

    num = lambda x,y: nx*y+x

    for k in range(nk):
        i = k%(nx)
        j = int(k/(nx))

        if k in puntos:
            if (i == 0 or i == nx-1 or j == 0 or j == ny-1):
                b[k] = 0

                data.append(1)
                row.append(k)
                col.append(k)
                continue

            elif (i<nbl and j<nbl):
                cA = (-1/delx**2 * tA(i, j, 1, 1) - 1/delx**2 * tA(i-1, j, 1, 1)
                      - 1/dely**2 * tB(i, j, 1, 1) - 1/dely**2 * tB(i, j-1, 1, 1)
                      + tC(i, j, 1, 1))

                cB = 1/delx**2 * tA(i, j, 1, 1)
                cC = 1/delx**2 * tA(i-1, j, 1, 1)
                cD = 1/dely**2 * tB(i, j, 1, 1)
                cE = 1/dely**2 * tB(i, j-1, 1, 1)

            elif (i<nbl and j>ny-nbl-1):
                cA = (-1/delx**2 * tA(i, j, 1, 1) - 1/delx**2 * tA(i-1, j, 1, 1)
                      - 1/dely**2 * tB(i, j+1, 1, 1) - 1/dely**2 * tB(i, j, 1, 1)
                      + tC(i, j, 1, 1))

                cB = 1/delx**2 * tA(i, j, 1, 1)
                cC = 1/delx**2 * tA(i-1, j, 1, 1)
                cD = 1/dely**2 * tB(i, j+1, 1, 1)
                cE = 1/dely**2 * tB(i, j, 1, 1)

            elif (i>nx-nbl-1 and j<nbl):
                cA = (-1/delx**2 * tA(i+1, j, 1, 1) - 1/delx**2 * tA(i, j, 1, 1)
                      - 1/dely**2 * tB(i, j, 1, 1) - 1/dely**2 * tB(i, j-1, 1, 1)
                      + tC(i, j, 1, 1))

                cB = 1/delx**2 * tA(i+1, j, 1, 1)
                cC = 1/delx**2 * tA(i, j, 1, 1)
                cD = 1/dely**2 * tB(i, j, 1, 1)
                cE = 1/dely**2 * tB(i, j-1, 1, 1)

            elif (i>nx-nbl-1 and j>ny-nbl-1):
                cA = (-1/delx**2 * tA(i+1, j, 1, 1) - 1/delx**2 * tA(i, j, 1, 1)
                      - 1/dely**2 * tB(i, j+1, 1, 1) - 1/dely**2 * tB(i, j, 1, 1)
                      + tC(i, j, 1, 1))

                cB = 1/delx**2 * tA(i+1, j, 1, 1)
                cC = 1/delx**2 * tA(i, j, 1, 1)
                cD = 1/dely**2 * tB(i, j+1, 1, 1)
                cE = 1/dely**2 * tB(i, j, 1, 1)

            elif (i<nbl):

                cA = (-1/delx**2 * tA(i, 0, 1, 0) - 1/delx**2 * tA(i-1, 0, 1, 0)
                      - 1/dely**2 * tB(i, 0, 1, 0) - 1/dely**2 * tB(i, 0, 1, 0)
                      + tC(i, 0, 1, 0))

                cB = 1/delx**2 * tA(i, 0, 1, 0)
                cC = 1/delx**2 * tA(i-1, 0, 1, 0)
                cD = 1/dely**2 * tB(i, 0, 1, 0)
                cE = 1/dely**2 * tB(i, 0, 1, 0)

            elif (i>nx-nbl-1):
                cA = (-1/delx**2 * tA(i+1, 0, 1, 0) - 1/delx**2 * tA(i, 0, 1, 0)
                      - 1/dely**2 * tB(i, 0, 1, 0) - 1/dely**2 * tB(i, 0, 1, 0)
                      + tC(i, 0, 1, 0))

                cB = 1/delx**2 * tA(i+1, 0, 1, 0)
                cC = 1/delx**2 * tA(i, 0, 1, 0)
                cD = 1/dely**2 * tB(i, 0, 1, 0)
                cE = 1/dely**2 * tB(i, 0, 1, 0)

            elif (j<nbl):
                cA = (-1/delx**2 * tA(0, j, 0, 1) - 1/delx**2 * tA(0, j, 0, 1)
                      - 1/dely**2 * tB(0, j, 0, 1) - 1/dely**2 * tB(0, j-1, 0, 1)
                      + tC(0, j, 0, 1))

                cB = 1/delx**2 * tA(0, j, 0, 1)
                cC = 1/delx**2 * tA(0, j, 0, 1)
                cD = 1/dely**2 * tB(0, j, 0, 1)
                cE = 1/dely**2 * tB(0, j-1, 0, 1)

            elif (j>ny-nbl-1):
                cA = (-1/delx**2 * tA(0, j, 0, 1) - 1/delx**2 * tA(0, j, 0, 1)
                      - 1/dely**2 * tB(0, j+1, 0, 1) - 1/dely**2 * tB(0, j, 0, 1)
                      + tC(0, j, 0, 1))

                cB = 1/delx**2 * tA(0, j, 0, 1)
                cC = 1/delx**2 * tA(0, j, 0, 1)
                cD = 1/dely**2 * tB(0, j+1, 0, 1)
                cE = 1/dely**2 * tB(0, j, 0, 1)

            else:
                cA = (-2/delx**2 - 2/dely**2 + (omega/velocity)**2)
                cB = 1/delx**2
                cC = 1/delx**2
                cD = 1/dely**2
                cE = 1/dely**2

            b[k] = source(x_dis[i], y_dis[j], sigma=0.05)

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
    
    u = np.zeros((nk,3), dtype=complex)
    U_array_2D = np.zeros((nx, ny), dtype=complex)
    b_array_2D = np.zeros((nx, ny), dtype=complex)

    for k in range(nk):
        i = k%(nx)
        j = int(k/(nx))
        u[k] = np.array([x_dis[i], y_dis[j], U[k]])
        U_array_2D[i, j] = U[k]
        b_array_2D[i, j] = b[k]

    print('Se soluciona el campo en cartesianas')

    return U, u, U_array_2D, b_array_2D

if __name__ == "__main__":

    nx = 501
    ny = 501
    nbl = 150
    freq = 10
    velocity = 1.5

    x_dis = np.linspace(-1.25, 1.25, nx)
    y_dis = np.linspace(-1.25, 1.25, ny)
    delx = x_dis[1] - x_dis[0]
    dely = y_dis[1] - y_dis[0]
    nk = nx * ny
    puntosIndices = np.arange(nk)

    lnbl = 0.75
    source_amplitude = 1.0
    alpha = 1.0

    U_c, u_space_c, U_array_2D, b_array_2D = helmholtz_dirchlet_solution(nx, ny, x_dis, y_dis, delx, dely, puntosIndices, nk, freq, nbl, velocity, lnbl, source_amplitude, alpha)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 6))
    vmax = np.max(np.abs(b_array_2D))
    im0 = ax0.imshow(np.real(b_array_2D).T, extent=(-1.25, 1.25, -1.25, 1.25), origin='lower', cmap='PRGn', vmin=-vmax, vmax=vmax)
    plt.colorbar(im0, ax=ax0, shrink=0.5)
    ax0.set_title('Source')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')

    vmax = np.max(np.abs(np.real(U_array_2D)))
    im1 = ax1.imshow(np.real(U_array_2D).T, extent=(-1.25, 1.25, -1.25, 1.25), origin='lower', cmap='seismic', vmin=-vmax, vmax=vmax)
    plt.colorbar(im1, ax=ax1, shrink=0.5)
    ax1.set_title('Field')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    im2 = ax2.imshow(np.angle(U_array_2D).T, extent=(-1.25, 1.25, -1.25, 1.25), origin='lower', cmap='twilight', vmin=-np.pi, vmax=np.pi)
    plt.colorbar(im2, ax=ax2, shrink=0.5)
    ax2.set_title('Phase')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    fig.tight_layout()

    plt.show()