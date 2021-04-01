import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import convolve2d


class HilbertByGaussianDerivative2D:

    def __init__(self, Npoints, file_saving_ws, nsigmas=8, isplotarox=False):
        sigminimum = 0.05 # 0.53
        sigmaximum = 0.005

        Len_x = Npoints
        Len_y = Npoints
        self.xx, self.yy = np.meshgrid(np.linspace(-0.5, 0.5, Len_y), np.linspace(0.5, -0.5, Len_x))
        self.x = self.xx[0, :]
        self.y = self.yy[:, 0]

        self.file_saving_ws = file_saving_ws

        self.sigmas = np.linspace(sigminimum, sigmaximum, nsigmas)  # [0.05, ] #

        w0 = np.ones(self.sigmas.size, dtype=np.float)
        opt_res = minimize(self.hilbert_dgs_diff, x0=w0)
        self.dg_weiths = opt_res.x

        np.save(self.file_saving_ws, self.dg_weiths)

        self.H_aproxed = self.sum_gaussian_derivaries(self.dg_weiths)  #
        self.H_aproxed_normed = self.H_aproxed / np.sqrt(np.sum(self.H_aproxed ** 2))



        if isplotarox:
            x = self.x
            H = 1 / (np.pi * x)
            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(1, 1, 1)

            # Move left y-axis and bottim x-axis to centre, passing through (0,0)
            ax.spines['left'].set_position('center')
            ax.spines['bottom'].set_position('center')

            # Eliminate upper and right axes
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')

            # Show ticks in the left and lower axes only
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

            ax.plot(x[x<0], self.H_aproxed[x<0], color="blue", label="Aproximated")
            ax.plot(x[x>0], self.H_aproxed[x>0], color="blue") # , label="Aproximated"

            ax.plot(x[x<0], H[x<0], color="green", label=r"$\frac{1}{\pi \cdot x}$")
            ax.plot(x[x>0], H[x>0], color="green" ) # label=r"$\frac{1}{\pi \cdot x}$"
            ax.legend()

    def hilbert2DbyX(self, u):
         u_imag = convolve2d(u, self.H_aproxed_normed, mode="same")

         return u_imag

    def inverse_hilbert2DbyX(self, u_imag):
         u = -convolve2d(u_imag, self.H_aproxed_normed, mode="same")
         return u

    def sum_gaussian_derivaries(self, w):

        dgsum = np.zeros_like(self.xx)

        for idx, sigma_y in enumerate(self.sigmas):
            gaussian = np.exp(-(self.xx ** 2 + 4 * self.yy ** 2) / (8 * sigma_y ** 2)) / (4 * np.pi * sigma_y ** 2)
            gaussian_dx = gaussian * -0.25 * self.xx / (sigma_y ** 2)

            dgsum += w[idx] * gaussian_dx

        # dgsum = dgsum / len(sigmas)

        return dgsum

    def hilbert_dgs_diff(self, w):
        H =  1 / (np.pi * self.x)
        sum_dg = self.sum_gaussian_derivaries(w)
        E = np.sum( (H - sum_dg[self.x.size//2, :])**2 )

        return E

if __name__ == '__main__':
    
    Npoints = 30
    file_saving_ws = "./results/weights_for_gaussian_derivatives_sum.npy"
    hilb2D = HilbertByGaussianDerivative2D(Npoints, file_saving_ws, nsigmas=8, isplotarox=False)

    # plt.pcolormesh(hilb2D.x, hilb2D.y, hilb2D.H_aproxed, cmap="rainbow", shading='auto')
    #H = 1 / (hilb2D.x * np.pi)

    #plt.plot(hilb2D.x, hilb2D.H_aproxed[15, :])
    #plt.plot(hilb2D.x, H)

    sine2D = np.cos(2*np.pi*hilb2D.xx * 2)

    plt.figure()
    plt.pcolormesh(hilb2D.x, hilb2D.y, sine2D, cmap="rainbow", shading='auto')

    sine2D_imag = hilb2D.hilbert2DbyX(sine2D)
    plt.figure()
    plt.pcolormesh(hilb2D.x, hilb2D.y, sine2D_imag, cmap="rainbow", shading='auto')


    plt.show()