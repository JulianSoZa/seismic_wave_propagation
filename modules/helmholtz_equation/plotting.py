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
    