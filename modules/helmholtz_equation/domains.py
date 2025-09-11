import numpy as np

class HelmholtzDomain:
    def __init__(self, main_shape, main_extension):
        self.main_shape = main_shape
        self.main_extension = main_extension
        self.nx = self.main_shape[0]
        self.ny = self.main_shape[1]

    def pml_domain(self, nbl):
        self.nbl = nbl

        self.nx = self.main_shape[0] + self.nbl*2
        self.ny = self.main_shape[1] + self.nbl*2

        self.lpml = (self.main_extension[1] - self.main_extension[0])/(self.main_shape[0]-1) * self.nbl

        self.shape = (self.nx, self.ny)
        self.extension = tuple(x - self.lpml if i % 2 == 0
                               else x + self.lpml for i, x in enumerate(self.main_extension))

        self.x_array = np.linspace(self.extension[0], self.extension[1], self.nx)
        self.y_array = np.linspace(self.extension[2], self.extension[3], self.ny)

        self.dx = self.x_array[1] - self.x_array[0]
        self.dy = self.y_array[1] - self.y_array[0]

        self.nk = self.nx * self.ny
        self.points = np.arange(self.nk)