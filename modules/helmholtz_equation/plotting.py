import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

from matplotlib.colors import LinearSegmentedColormap
cmap_source = LinearSegmentedColormap.from_list('source', ['green', 'white', 'purple'])

def plot_velocity(domain, velocity, figsize=(8, 6)):
    """
    Plots the velocity model

    Parameters
    ----------
        domain: object
            with attributes:\n
                extension: touple of 4 floats
                    (x_min, x_max, y_min, y_max) defining the plot extent
        velocity: array
            A 2D numpy array representing the velocity model.
    """
    plt.figure(figsize=figsize)
    plt.imshow(velocity.T, extent=domain.extension, origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title('Velocity Model')
    plt.xlabel('x')
    plt.ylabel('y')

def plot_solution(domain, frequency_array, this_nfrequency, u_arrays, b_arrays, u_vmax=None, figsize=(14, 5)):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f'Frequency: {frequency_array[this_nfrequency]:.1f} Hz', fontsize=16)
    rect_params = dict(
        xy=(domain.main_extension[0], domain.main_extension[2]), 
        width=domain.main_extension[1]-domain.main_extension[0], 
        height=domain.main_extension[3]-domain.main_extension[2], 
        linewidth=1, edgecolor='r', facecolor='none')

    vmax = np.max(np.abs(b_arrays[..., this_nfrequency]))
    im0 = ax0.imshow(np.real(b_arrays[..., this_nfrequency]).T, extent=domain.extension, origin='lower', cmap=cmap_source, vmin=-vmax, vmax=vmax)
    fig.colorbar(im0, ax=ax0, shrink=0.5)
    ax0.add_patch(patches.Rectangle(**rect_params))
    ax0.set_title('Source (Real part)')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')

    vmax = np.max(np.abs(np.real(u_arrays[..., this_nfrequency]))) if u_vmax is None else u_vmax
    im1 = ax1.imshow(np.real(u_arrays[..., this_nfrequency]).T, extent=domain.extension, origin='lower', cmap='seismic', vmin=-vmax, vmax=vmax)
    fig.colorbar(im1, ax=ax1, shrink=0.5)
    ax1.add_patch(patches.Rectangle(**rect_params))
    ax1.set_title('Field (Real part)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    im2 = ax2.imshow(np.angle(u_arrays[..., this_nfrequency]).T, extent=domain.extension, origin='lower', cmap='twilight', vmin=-np.pi, vmax=np.pi)
    fig.colorbar(im2, ax=ax2, shrink=0.5)
    ax2.add_patch(patches.Rectangle(**rect_params))
    ax2.set_title('Phase')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    fig.tight_layout()

def plot_main_solution(domain, frequency_array, this_nfrequency, u_arrays, b_arrays, u_vmax=None, figsize=(14, 5)):
    idx_main_domain = (slice(domain.nbl, -domain.nbl), slice(domain.nbl, -domain.nbl), slice(None))

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f'Frequency: {frequency_array[this_nfrequency]:.1f} Hz', fontsize=16)

    vmax = np.max(np.abs(b_arrays[..., this_nfrequency]))
    im0 = ax0.imshow(np.real(b_arrays[..., this_nfrequency])[idx_main_domain[:2]].T, extent=domain.main_extension, origin='lower', cmap=cmap_source, vmin=-vmax, vmax=vmax)
    fig.colorbar(im0, ax=ax0, shrink=0.5)
    ax0.set_title('Source (Real part)')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')

    vmax = np.max(np.abs(np.real(u_arrays[..., this_nfrequency]))) if u_vmax is None else u_vmax
    im1 = ax1.imshow(np.real(u_arrays[..., this_nfrequency])[idx_main_domain[:2]].T, extent=domain.main_extension, origin='lower', cmap='seismic', vmin=-vmax, vmax=vmax)
    fig.colorbar(im1, ax=ax1, shrink=0.5)
    ax1.set_title('Field (Real part)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    im2 = ax2.imshow(np.angle(u_arrays[..., this_nfrequency])[idx_main_domain[:2]].T, extent=domain.main_extension, origin='lower', cmap='twilight', vmin=-np.pi, vmax=np.pi)
    fig.colorbar(im2, ax=ax2, shrink=0.5)
    ax2.set_title('Phase')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    fig.tight_layout()

def plot_source_term(domain, frequencies, this_nfrequency, source_injected, source_term, figsize=(14, 5)):
    idx_main_domain = (slice(domain.nbl, -domain.nbl), slice(domain.nbl, -domain.nbl), slice(None))
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f'Frequency: {frequencies[this_nfrequency]:.1f} Hz', fontsize=16)

    vmax = np.max(np.abs(source_injected[..., this_nfrequency]))
    im0 = ax0.imshow(np.real(source_injected[..., this_nfrequency])[idx_main_domain[:2]].T, extent=domain.main_extension, origin='lower', cmap=cmap_source, vmin=-vmax, vmax=vmax)
    fig.colorbar(im0, ax=ax0, shrink=0.5)
    ax0.set_title('Source (Real part)')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')

    vmax = np.max(np.abs(source_term[..., this_nfrequency]))
    im1 = ax1.imshow(np.real(source_term[..., this_nfrequency])[idx_main_domain[:2]].T, extent=domain.main_extension, origin='lower', cmap=cmap_source, vmin=-vmax, vmax=vmax)
    fig.colorbar(im1, ax=ax1, shrink=0.5)
    ax1.set_title('Source Calculated (Real part)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    error = (source_injected - source_term)
    vmax = np.max(np.abs(error[..., this_nfrequency])[idx_main_domain[:2]])
    im2 = ax2.imshow(np.real(error[..., this_nfrequency])[idx_main_domain[:2]].T, extent=domain.main_extension, cmap=cmap_source, origin='lower', vmin=-vmax, vmax=vmax)
    fig.colorbar(im2, ax=ax2, shrink=0.5)
    ax2.set_title('Source Error')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    fig.tight_layout()

def plot_source_comparison(domain, this_nfrequency, frequency_array, source_injected, source_term, source_analytical, pos_injection, figsize=(14, 5)):
    idx_main_domain = (slice(domain.nbl, -domain.nbl), slice(domain.nbl, -domain.nbl), slice(None))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)
    ax0.imshow(np.real(source_injected)[..., this_nfrequency][idx_main_domain[:2]].T, origin='lower')
    ax0.set_title('Injected Source')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')

    ax1.plot(frequency_array, np.real(source_injected)[idx_main_domain[:2]].T[:, pos_injection[0], pos_injection[1]], label='Injected')
    ax1.plot(frequency_array, np.real(source_term)[idx_main_domain[:2]].T[:, pos_injection[0], pos_injection[1]], '--', label='Calculated')
    ax1.plot(frequency_array, source_analytical, '.', label='Analytical')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Source Comparison at {pos_injection}')
    ax1.grid()
    ax1.legend()
    fig.tight_layout()

def plot_field(domain, u, vmax=None, figsize=(8, 6), title='Field'):
    """
    Plots the field.

    Parameters
    ----------
        domain: object
            with attributes:\n
                extension: touple of 4 floats
                    (x_min, x_max, y_min, y_max) defining the plot extent
        u: array
            A 2D numpy array representing the field solution.
    """
    plt.figure(figsize=figsize)
    vmax = np.max(np.abs(u)) if vmax is None else vmax
    plt.imshow(u.T, extent=domain.extension, origin='lower', cmap='seismic', vmin=-vmax, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')

def plot_pml_coefficients(domain, frequency, alpha):
    """
    Visualize PML attenuation coefficients (sigma_x, sigma_z, and total).

    Parameters
    ----------
    domain : object
        HelmholtzDomain object with PML configuration
    frequency : float
        Frequency of the wave in Hz
    alpha : float
        PML parameter
    """
    import matplotlib.pyplot as plt
    
    nx = domain.nx
    ny = domain.ny
    nbl = domain.nbl
    lpml = domain.lpml
    x_array = domain.x_array
    y_array = domain.y_array

    # Sigma functions (same as in the solver)
    sigma_x = lambda x, ax: 2*np.pi*alpha*frequency*((abs(x_array[x])-abs(x_array[nbl]))*ax/lpml)**2
    sigma_y = lambda y, ay: 2*np.pi*alpha*frequency*((abs(y_array[y])-abs(y_array[nbl]))*ay/lpml)**2

    # Create 2D sigma matrices
    sigma_x_field = np.zeros((nx, ny))
    sigma_y_field = np.zeros((nx, ny))

    for i in range(nx):
        for j in range(ny):
            # Determine if we're in the PML zone (same logic as solver)
            if ((i<nbl and j<nbl) or (i<nbl and j>ny-nbl-1) or (i>nx-nbl-1 and j<nbl) or (i>nx-nbl-1 and j>ny-nbl-1)):
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
            
            sigma_x_field[i, j] = sigma_x(i, pml_x)
            sigma_y_field[i, j] = sigma_y(j, pml_y)

    # Main domain coordinates (without PML)
    x_main_min = x_array[nbl]
    x_main_max = x_array[nx-nbl-1]
    z_main_min = y_array[nbl]
    z_main_max = y_array[ny-nbl-1]

    # Midpoints for indicator lines
    x_mid = (x_array[0] + x_array[-1]) / 2
    z_mid = (y_array[0] + y_array[-1]) / 2

    # Midpoints between domain edge and box edge (middle of PML)
    x_pml_mid_left = (x_array[0] + x_main_min) / 2
    z_pml_mid_bottom = (y_array[0] + z_main_min) / 2

    # Offset to avoid overlap
    offset_z = 0.4
    offset_x = 0.4

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    extent = [x_array[0], x_array[-1], y_array[0], y_array[-1]]

    # Sigma_x
    im0 = axes[0].imshow(sigma_x_field.T, extent=extent, origin='lower', cmap='Blues', aspect='auto')
    axes[0].set_title(r'$\sigma_x$ (attenuation in x)', fontsize=14)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('z')
    plt.colorbar(im0, ax=axes[0], label=r'$\sigma_x$')
    # Draw main domain box
    axes[0].plot([x_main_min, x_main_max, x_main_max, x_main_min, x_main_min],
                [z_main_min, z_main_min, z_main_max, z_main_max, z_main_min],
                'r--', linewidth=2)
    # L_pml indicator line in x direction (left)
    axes[0].plot([x_array[0], x_main_min], [z_mid, z_mid], 'k-', linewidth=2)
    axes[0].plot([x_array[0], x_array[0]], [z_mid-0.02, z_mid+0.02], 'k-', linewidth=2)
    axes[0].plot([x_main_min, x_main_min], [z_mid-0.02, z_mid+0.02], 'k-', linewidth=2)
    axes[0].text((x_array[0] + x_main_min)/2, z_mid+0.05, r'$L_{PML}$', 
                fontsize=12, ha='center', va='bottom')
    # Indicate Lx (from box edge to PML midpoint, slightly higher)
    z_lx = z_mid + offset_z
    axes[0].plot([x_main_min, x_pml_mid_left], [z_lx, z_lx], 'g-', linewidth=2)
    axes[0].plot([x_main_min, x_main_min], [z_lx-0.02, z_lx+0.02], 'g-', linewidth=2)
    axes[0].plot([x_pml_mid_left, x_pml_mid_left], [z_lx-0.02, z_lx+0.02], 'g-', linewidth=2)
    axes[0].text((x_main_min + x_pml_mid_left)/2, z_lx+0.05, r'$L_x$', 
                fontsize=12, ha='center', va='bottom')

    # Sigma_z (formerly sigma_y)
    im1 = axes[1].imshow(sigma_y_field.T, extent=extent, origin='lower', cmap='Blues', aspect='auto')
    axes[1].set_title(r'$\sigma_z$ (attenuation in z)', fontsize=14)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('z')
    plt.colorbar(im1, ax=axes[1], label=r'$\sigma_z$')
    # Draw main domain box
    axes[1].plot([x_main_min, x_main_max, x_main_max, x_main_min, x_main_min],
                [z_main_min, z_main_min, z_main_max, z_main_max, z_main_min],
                'r--', linewidth=2)
    # L_pml indicator line in z direction (bottom)
    axes[1].plot([x_mid, x_mid], [y_array[0], z_main_min], 'k-', linewidth=2)
    axes[1].plot([x_mid-0.02, x_mid+0.02], [y_array[0], y_array[0]], 'k-', linewidth=2)
    axes[1].plot([x_mid-0.02, x_mid+0.02], [z_main_min, z_main_min], 'k-', linewidth=2)
    axes[1].text(x_mid+0.05, (y_array[0] + z_main_min)/2, r'$L_{PML}$', 
                fontsize=12, ha='left', va='center')
    # Indicate Lz (from box edge to PML midpoint, slightly to the right)
    x_lz = x_mid + offset_x
    axes[1].plot([x_lz, x_lz], [z_main_min, z_pml_mid_bottom], 'g-', linewidth=2)
    axes[1].plot([x_lz-0.02, x_lz+0.02], [z_main_min, z_main_min], 'g-', linewidth=2)
    axes[1].plot([x_lz-0.02, x_lz+0.02], [z_pml_mid_bottom, z_pml_mid_bottom], 'g-', linewidth=2)
    axes[1].text(x_lz+0.05, (z_main_min + z_pml_mid_bottom)/2, r'$L_z$', 
                fontsize=12, ha='left', va='center')

    # Total sigma (combined)
    sigma_total = np.sqrt(sigma_x_field**2 + sigma_y_field**2)
    im2 = axes[2].imshow(sigma_total.T, extent=extent, origin='lower', cmap='Blues', aspect='auto')
    axes[2].set_title(r'$\sqrt{\sigma_x^2 + \sigma_z^2}$ (total attenuation)', fontsize=14)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('z')
    plt.colorbar(im2, ax=axes[2], label=r'$|\sigma|$')
    # Draw main domain box
    axes[2].plot([x_main_min, x_main_max, x_main_max, x_main_min, x_main_min],
                [z_main_min, z_main_min, z_main_max, z_main_max, z_main_min],
                'r--', linewidth=2)
    # L_pml indicator lines
    # Horizontal
    axes[2].plot([x_array[0], x_main_min], [z_mid, z_mid], 'k-', linewidth=2)
    axes[2].plot([x_array[0], x_array[0]], [z_mid-0.02, z_mid+0.02], 'k-', linewidth=2)
    axes[2].plot([x_main_min, x_main_min], [z_mid-0.02, z_mid+0.02], 'k-', linewidth=2)
    axes[2].text((x_array[0] + x_main_min)/2, z_mid+0.05, r'$L_{PML}$', 
                fontsize=12, ha='center', va='bottom')
    # Vertical
    axes[2].plot([x_mid, x_mid], [y_array[0], z_main_min], 'k-', linewidth=2)
    axes[2].plot([x_mid-0.02, x_mid+0.02], [y_array[0], y_array[0]], 'k-', linewidth=2)
    axes[2].plot([x_mid-0.02, x_mid+0.02], [z_main_min, z_main_min], 'k-', linewidth=2)
    axes[2].text(x_mid+0.05, (y_array[0] + z_main_min)/2, r'$L_{PML}$', 
                fontsize=12, ha='left', va='center')
    # Indicate Lx (from box edge to PML midpoint, slightly higher)
    z_lx = z_mid + offset_z
    axes[2].plot([x_main_min, x_pml_mid_left], [z_lx, z_lx], 'g-', linewidth=2)
    axes[2].plot([x_main_min, x_main_min], [z_lx-0.02, z_lx+0.02], 'g-', linewidth=2)
    axes[2].plot([x_pml_mid_left, x_pml_mid_left], [z_lx-0.02, z_lx+0.02], 'g-', linewidth=2)
    axes[2].text((x_main_min + x_pml_mid_left)/2, z_lx+0.05, r'$L_x$', 
                fontsize=12, ha='center', va='bottom')
    # Indicate Lz (from box edge to PML midpoint, slightly to the right)
    x_lz = x_mid + offset_x
    axes[2].plot([x_lz, x_lz], [z_main_min, z_pml_mid_bottom], 'g-', linewidth=2)
    axes[2].plot([x_lz-0.02, x_lz+0.02], [z_main_min, z_main_min], 'g-', linewidth=2)
    axes[2].plot([x_lz-0.02, x_lz+0.02], [z_pml_mid_bottom, z_pml_mid_bottom], 'g-', linewidth=2)
    axes[2].text(x_lz+0.05, (z_main_min + z_pml_mid_bottom)/2, r'$L_z$', 
                fontsize=12, ha='left', va='center')

    plt.suptitle(f'PML Attenuation Coefficients (α={alpha}, f={frequency} Hz, nbl={nbl})', fontsize=14, y=1.00)
    plt.tight_layout()