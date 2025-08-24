import numpy as np

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