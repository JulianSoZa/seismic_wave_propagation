import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_source_time_analysis(geometry, tlim=500, flim=50):
    amplitude = geometry.src.data
    time = geometry.src.time_values

    dt = np.diff(time)[0]
    frequency_spectrum = np.fft.rfft(amplitude, axis=0)*dt*1e-3
    frequency_array = np.fft.rfftfreq(time.shape[-1], d=dt)*1e3
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1])
    
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(time, amplitude)
    ax0.set_title('Ricker Wavelet Time Domain')
    ax0.set_xlim(0, tlim)
    ax0.set_xlabel('Time (s)')
    ax0.set_ylabel('Amplitude')
    ax0.grid()

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(frequency_array, np.abs(frequency_spectrum)**2)
    ax1.set_title('Ricker Wavelet Power Spectrum')
    ax1.set_xlim(0, flim)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power')
    ax1.grid()

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(frequency_array, np.real(frequency_spectrum))
    ax2.set_title('Ricker Wavelet Real Spectrum')
    ax2.set_xlim(0, flim)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Real Part')
    ax2.grid()

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(frequency_array, np.angle(frequency_spectrum))
    ax3.set_title('Ricker Wavelet Phase Spectrum')
    ax3.set_xlim(0, flim)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Phase (radians)')
    ax3.grid()

    fig.tight_layout()
    
def plot_source(geometry, model, src_space, tlim=500):
    idx_domain_ROI = (slice(model.nbl, -model.nbl), slice(model.nbl, -model.nbl))
    domain_size = 1.e-3 * np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]

    amplitude = geometry.src.data
    time = geometry.src.time_values
    
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(src_space.data[idx_domain_ROI].T, extent=extent, cmap='gray')
    ax0.set_title('Source Spatial Function')
    ax0.set_xlabel('X position (km)')
    ax0.set_ylabel('Depth (km)')
    plt.colorbar(ax0.images[0], ax=ax0, shrink=1)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(time, amplitude)
    ax1.set_title('Ricker Wavelet Time Domain')
    ax1.set_xlim(0, tlim)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid()

    fig.tight_layout()
    
def plot_instantaneous_wavefield(u, geometry, model, time_instant, vmax=1, vmin=None):
    if vmin is None:
        vmin = -vmax
    
    idx_domain_ROI = (slice(None), slice(model.nbl, -model.nbl), slice(model.nbl, -model.nbl))
    domain_size = 1.e-3 * np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]

    plt.figure(figsize=(7, 5))
    im1 = plt.imshow(u.data[idx_domain_ROI][time_instant].T, extent=extent, cmap='seismic', vmin=vmin, vmax=vmax)
    plt.title(f'Wavefield at {geometry.time_axis.time_values[time_instant]:.1f} ms')
    plt.colorbar(im1, shrink=1)
    plt.xlabel('X position (km)')
    plt.ylabel('Depth (km)')
    plt.tight_layout()
    
def plot_wavefield_animation(u, geometry, model, vmax=None, vmin=None):
    idx_domain_ROI = (slice(None), slice(model.nbl, -model.nbl), slice(model.nbl, -model.nbl))
    data = u.data[idx_domain_ROI]
    nt = data.shape[0]

    domain_size = 1.e-3 * np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0],
                model.origin[1] + domain_size[1], model.origin[1]]

    fig, ax = plt.subplots(figsize=(7, 5))

    if vmax is None:
        vmax = np.max(np.abs(data))
        
    if vmin is None:
        vmin = -vmax
        
    im = ax.imshow(data[0].T, extent=extent, cmap='seismic', vmin=vmin, vmax=vmax)

    cb = plt.colorbar(im, ax=ax, shrink=1)

    title = ax.set_title(f'Wavefield at {geometry.time_axis.time_values[0]:.1f} ms')

    plt.xlabel('x position (km)')
    plt.ylabel('Depth (km)')
    plt.tight_layout()

    def update(frame):
        im.set_data(data[frame].T)
        title.set_text(f'Wavefield at {geometry.time_axis.time_values[frame]:.1f} ms')
        return [im, title]

    ani = FuncAnimation(fig, update, frames=nt, interval=30, blit=True)
    return ani

def plot_frequency_modes(model, freq_modes, freq_array, freq_plot, vmaxInt=None, vminInt=None, vmaxReal=None, vminReal=None, figsize=(14, 20)):
    if vmaxInt is None:
        vmaxInt = np.max(np.abs(freq_modes.data))
    if vminInt is None:
        vminInt = 0
        
    if vmaxReal is None:
        vmaxReal = np.max(np.abs(np.real(freq_modes.data)))
    if vminReal is None:
        vminReal = -vmaxReal
    
    idx_domain_ROI = (slice(None), slice(model.nbl, -model.nbl), slice(model.nbl, -model.nbl))
    
    domain_size = 1.e-3 * np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0],
                model.origin[1] + domain_size[1], model.origin[1]]
    
    plt.figure(figsize=figsize)
    plt.suptitle('Frequency Modes (ROI)', fontsize=20)
    
    for i, freq in enumerate(freq_plot):
        plt.subplot(5, 3, 3*i+1)
        im1 = plt.imshow(np.abs(freq_modes.data[freq].T[idx_domain_ROI[1:]]), extent=extent, cmap='gray', vmin=vminInt, vmax=vmaxInt)
        plt.title(f'Magnitude {1e3*freq_array[freq]:.1f} Hz', fontsize=14)
        plt.colorbar(im1, shrink=0.8)
        plt.xlabel('x position (km)')
        plt.ylabel('Depth (km)')

        plt.subplot(5, 3, 3*i+2)
        im2 = plt.imshow(np.angle(freq_modes.data[freq].T[idx_domain_ROI[1:]]), extent=extent, cmap='twilight', vmin=-np.pi, vmax=np.pi)
        plt.title(f'Angle {1e3*freq_array[freq]:.1f} Hz', fontsize=14)
        plt.colorbar(im2, shrink=0.8)
        plt.xlabel('x position (km)')
        plt.ylabel('Depth (km)')

        plt.subplot(5, 3, 3*i+3)
        im3 = plt.imshow(np.real(freq_modes.data[freq].T[idx_domain_ROI[1:]]), extent=extent, cmap='seismic', vmin=vminReal, vmax=vmaxReal)
        plt.title(f'Real part {1e3*freq_array[freq]:.1f} Hz', fontsize=14)
        plt.colorbar(im3, shrink=0.8)
        plt.xlabel('x position (km)')
        plt.ylabel('Depth (km)')

    plt.tight_layout()