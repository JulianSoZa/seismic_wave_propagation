import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

class GaussianSource:
    def __init__(self, amplitude, x_pos, y_pos, sigma, phase):
        self.amplitude = amplitude
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.sigma = sigma
        self.phase = phase

    def __call__(self, x, y):
        return self.amplitude * np.exp(-((x - self.x_pos)**2 + (y - self.y_pos)**2) / (2 * self.sigma**2)) * np.exp(1j * self.phase)

def helmholtz_pml_solution(nx, ny, x_array, y_array, dx, dy, points, nk, frequency, nbl, velocity, lpml, source, alpha):
    data = []
    row = []
    col = []

    b = np.zeros(nk, dtype=complex)

    omega = 2*np.pi*frequency
    
    sigma_x = lambda x, ax: 2*np.pi*alpha*frequency*((abs(x_array[x])-abs(x_array[nbl]))*ax/lpml)**2
    sigma_y = lambda y, ay: 2*np.pi*alpha*frequency*((abs(y_array[y])-abs(y_array[nbl]))*ay/lpml)**2

    tA = lambda x, y, ax, ay: ((1-1j*sigma_y(y, ay)/omega)/(1-1j*sigma_x(x, ax)/omega))
    tB = lambda x, y, ax, ay: ((1-1j*sigma_x(x, ax)/omega)/(1-1j*sigma_y(y, ay)/omega))
    tC = lambda x, y, ax, ay: (((1-1j*sigma_y(y, ay)/omega))*((1-1j*sigma_x(x, ax)/omega))*(omega/velocity[x, y])**2)

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
            print('no')
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

    print('Se soluciona el campo en cartesianas')

    return u_array_2D, b_array_2D

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    main_domain_shape = (201, 201)
    main_domain_extension = (-0.5, 0.5, -0.5, 0.5)
    nbl = 100

    nx = main_domain_shape[0] + nbl*2
    ny = main_domain_shape[1] + nbl*2

    lpml = (main_domain_extension[1] - main_domain_extension[0])/(main_domain_shape[0]-1) * nbl

    domain_shape = (nx, ny)
    domain_extension = tuple(x - lpml if i % 2 == 0 else x + lpml for i, x in enumerate(main_domain_extension))

    x_array = np.linspace(domain_extension[0], domain_extension[1], nx)
    y_array = np.linspace(domain_extension[2], domain_extension[3], ny)
    
    dx = x_array[1] - x_array[0]
    dy = y_array[1] - y_array[0]
    
    nk = nx * ny
    points_number = np.arange(nk)
    
    frequency = 10
    velocity = 1.5
    
    velocity_array = np.ones((nx, ny))*velocity
    velocity_array[:, :(ny-1)//2] = velocity*2
    
    alpha = 1.4
    
    source = GaussianSource(amplitude=1, x_pos=0, y_pos=0.4, sigma=0.05, phase=np.pi/8)

    u, b = helmholtz_pml_solution(nx, ny, x_array, y_array, dx, dy, points_number, nk, frequency,
                                                                         nbl, velocity_array, lpml, source, alpha)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 6))
    rect_params = dict(
        xy=(main_domain_extension[0], main_domain_extension[2]), 
        width=main_domain_extension[1]-main_domain_extension[0], 
        height=main_domain_extension[3]-main_domain_extension[2], 
        linewidth=1, edgecolor='r', facecolor='none')
    
    vmax = np.max(np.abs(b))
    im0 = ax0.imshow(np.real(b).T, extent=domain_extension, origin='lower', cmap='PRGn', vmin=-vmax, vmax=vmax)
    plt.colorbar(im0, ax=ax0, shrink=0.5)
    ax0.add_patch(patches.Rectangle(**rect_params))
    ax0.set_title('Source')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')

    vmax = np.max(np.abs(np.real(u)))
    im1 = ax1.imshow(np.real(u).T, extent=domain_extension, origin='lower', cmap='seismic', vmin=-vmax, vmax=vmax)
    plt.colorbar(im1, ax=ax1, shrink=0.5)
    ax1.add_patch(patches.Rectangle(**rect_params))
    ax1.set_title('Field')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    im2 = ax2.imshow(np.angle(u).T, extent=domain_extension, origin='lower', cmap='twilight', vmin=-np.pi, vmax=np.pi)
    plt.colorbar(im2, ax=ax2, shrink=0.5)
    ax2.add_patch(patches.Rectangle(**rect_params))
    ax2.set_title('Phase')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    fig.tight_layout()
    
    fig2 = plt.figure()
    plt.imshow(velocity_array.T, extent=domain_extension, origin='lower', cmap='viridis')
    plt.colorbar(shrink=0.5)
    plt.title('Velocity')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()