import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

from matplotlib.colors import LinearSegmentedColormap
cmap_source = LinearSegmentedColormap.from_list('source', ['green', 'white', 'purple'])

def plot_velocity(domain, velocity, figsize=(8, 6)):
    """
    Plots the velocity model

    Parameters:
    domain: object
        with attributes:
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