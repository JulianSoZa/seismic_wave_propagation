import numpy as np
import matplotlib.pyplot as plt

class GaussianSource:
    def __init__(self, amplitude, x_pos, y_pos, sigma, phase):
        self.amplitude = amplitude
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.sigma = sigma
        self.phase = phase

    def __call__(self, x, y):
        return self.amplitude * np.exp(-((x - self.x_pos)**2 + (y - self.y_pos)**2) / (2 * self.sigma**2)) * np.exp(1j * self.phase)

class SinSinSource:
    def __init__(self, ax, ay, wavenumber):
        self.ax = ax
        self.ay = ay
        self.wavenumber = wavenumber

    def __call__(self, x, y):
        return (-(self.ax * np.pi)**2 - (self.ay * np.pi)**2 + self.wavenumber**2) * np.sin(self.ax * np.pi * x) * np.sin(self.ay * np.pi * y)

class RickerWaveletFD(GaussianSource):
    def __init__(self, f, f0, t0, amplitude, x_pos, y_pos, sigma):
        super().__init__(amplitude, x_pos, y_pos, sigma, phase=0)
        self.f = f
        self.f0 = f0
        self.t0 = t0

    def __call__(self, x, y):
        spatial_distribution = super().__call__(x, y)
        frequency_distribution = self.frequency_spectrum(self.f)
        return spatial_distribution * frequency_distribution
    
    def frequency_spectrum(self, f):
        return (2/np.sqrt(np.pi) * f**2/self.f0**3) * np.exp(-(f/self.f0)**2) * np.exp(-1j*2*np.pi*f*self.t0)

    def plot(self, axs, frequency_array, nfreq=1000, dt=0.001):
        axs[0].plot(frequency_array, np.abs(self.frequency_spectrum(frequency_array))**2)
        axs[0].set_title('Ricker Wavelet Power Spectrum')
        axs[0].set_xlabel('Frequency (Hz)')
        axs[0].set_ylabel('Power')
        axs[0].grid()

        axs[1].plot(frequency_array, np.angle(self.frequency_spectrum(frequency_array)))
        axs[1].set_title('Ricker Wavelet Phase Spectrum')
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('Phase (radians)')
        axs[1].grid()

        axs[2].plot(frequency_array, np.real(self.frequency_spectrum(frequency_array)))
        axs[2].set_title('Ricker Wavelet Real Spectrum')
        axs[2].set_xlabel('Frequency (Hz)')
        axs[2].set_ylabel('Real Part')
        axs[2].grid()

        frequency_spectrum = self.frequency_spectrum(np.fft.rfftfreq(nfreq, d=dt))
        time_distribution = np.fft.irfft(frequency_spectrum)/dt
        
        axs[3].plot(np.arange(time_distribution.size)*dt, time_distribution)
        axs[3].set_title('Ricker Wavelet Time Domain')
        axs[3].set_xlabel('Time (s)')
        axs[3].set_ylabel('Amplitude')
        axs[3].grid()