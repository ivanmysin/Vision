import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class HyperColomn:
    def __init__(self, centers, xx, yy, angles, sigmas):

        self.cent_x = centers[0]
        self.cent_y = centers[1]
        self.xx = xx - self.cent_x
        self.yy = yy - self.cent_y

        self.x = self.xx[0, :]
        self.y = self.yy[:, 0]

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        self.angles = angles
        self.sigmas = sigmas

        self.aproximate_hilbert_kernel()


    def aproximate_hilbert_kernel(self):
        w0 = np.ones(self.sigmas.size, dtype=np.float)
        opt_res = minimize(self._hilbert_dgs_diff, x0=w0, args=(self.xx, self.yy, self.sigmas))
        self.dg_weiths = opt_res.x
        # np.save(self.file_saving_ws, self.dg_weiths)
        self.H_aproxed = self.sum_gaussian_derivaries(self.dg_weiths, self.xx, self.yy, self.sigmas)  #
        self.H_aproxed_normed = self.H_aproxed / np.sqrt(np.sum(self.H_aproxed ** 2))

    def rotate_xy(self):
        pass

    def compute_kernels(self):
        pass

    def get_rickers(self, sigma, xx, yy):
        ricker = 0.005 * (4 - xx**2 / sigma**2) * np.exp(-(xx**2 + 4 * yy**2) / (8 * sigma**2)) / sigma**4
        return ricker

    def get_gaussian_derivative(self, sigma, xx, yy):
        gaussian = np.exp(-(xx**2 + 4 * yy**2) / (8 * sigma**2)) / (4 * np.pi * sigma**2)
        gaussian_dx = gaussian * -0.25 * xx / (sigma**2)
        return gaussian_dx



    def sum_gaussian_derivaries(self, w, xx, yy, sigmas):
        dgsum = np.zeros_like(xx)
        for idx, sigma in enumerate(sigmas):
            gaussian_dx = self.get_gaussian_derivative(sigma, xx, yy)
            dgsum += w[idx] * gaussian_dx
        return dgsum

    def _hilbert_dgs_diff(self, w, xx, yy, sigmas):
        H =  1 / (np.pi * self.x)
        sum_dg = self.sum_gaussian_derivaries(w, xx, yy, sigmas)
        E = np.sum( (H - sum_dg[self.x.size//2, :])**2 )

        return E



if __name__ == '__main__':
    centers = [0, 0]
    centers = [0, 0]
    angles = [0, 0.5*np.pi]
    Len_y = 100
    Len_x = 100

    nsigmas = 8
    sigminimum = 0.05
    sigmaximum = 0.005
    sigmas = np.linspace(sigminimum, sigmaximum, nsigmas)
    xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, Len_y), np.linspace(0.5, -0.5, Len_x))

    hc = HyperColomn(centers, xx, yy, angles,sigmas)

    print(hc.H_aproxed_normed.shape)
    print("Hello")