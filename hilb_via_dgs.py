import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import hilbert

def sum_gaussian_derivaries(w, x, sigmas):

    dgsum = np.zeros_like(x)

    for idx, sigma in enumerate(sigmas):
        dg = np.exp(-0.5 * (x/sigma)**2) * -x / sigma**2

        dgsum += w[idx] * dg

    dgsum = dgsum / len(sigmas)

    return dgsum

def hilbert_dgs_diff(w, x, sigmas):
    H =  1 / (np.pi * x)
    sum_dg = sum_gaussian_derivaries(w, x, sigmas)
    E = np.sum( (H - sum_dg)**2 )

    return E

class HilbertByGaussianDerivative:

    def __init__(self, Npoints, file_saving_ws, nsigmas=8, isplotarox=False):
        sigminimum = 0.05 # 0.53
        sigmaximum = 0.005
        self.x = np.linspace(-0.5, 0.5, Npoints)

        self.file_saving_ws = file_saving_ws

        self.sigmas = np.linspace(sigminimum, sigmaximum, nsigmas)  # [0.05, ] #

        w0 = np.ones(self.sigmas.size, dtype=np.float)
        opt_res = minimize(hilbert_dgs_diff, x0=w0, args=(self.x, self.sigmas))
        self.dg_weiths = opt_res.x

        np.save(self.file_saving_ws, self.dg_weiths)

        self.H_aproxed = sum_gaussian_derivaries(self.dg_weiths , self.x, self.sigmas)  #
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

    def hilbert(self, u):
        u_imag = np.convolve(u, self.H_aproxed_normed , mode="same")

        return u_imag

    def inverse_hilbert(self, u_imag):
        u = -np.convolve(u_imag, self.H_aproxed_normed, mode="same")
        return u

if __name__ == "__main__":
    file_saving_ws = "./results/weights_for_gaussian_derivatives_sum.npy"

    Npoints = 30


    x, dx = np.linspace(-0.5, 0.5, Npoints, retstep=True)

    myHilbert = HilbertByGaussianDerivative(Npoints, file_saving_ws, nsigmas=8, isplotarox=True)

    """
    possible_freqs = np.fft.rfftfreq(Npoints, d=dx)[1:]
    # print(possible_freqs)
    sigmaximum = 1 / (possible_freqs[-1]**2 )
    sigminimum = 0.5 / (possible_freqs[0]**2 )
    """


    fig2 = plt.figure(figsize=(15, 15))

    U = np.cos(2*np.pi*2.1*x)
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(x, U, label="Origin signal")

    """
    dg_convs = np.zeros_like(x)
    U_restored = np.zeros_like(x)
    for idx, sigma in enumerate(sigmas):
        dg = np.exp(-0.5 * (x/sigma)**2) * -x / sigma**2

        dg_con = -opt_res.x[idx] * np.convolve(dg, U, mode="same")
        dg_convs += dg_con

    # dg_convs = dg_convs / np.max(dg_convs) # !!!!!!!!


    for idx, sigma in enumerate(sigmas):
        dg = np.exp(-0.5 * (x / sigma) ** 2) * -x / sigma ** 2

        dg_con = opt_res.x[idx] * np.convolve(dg, dg_convs, mode="same")
        U_restored += dg_con
    
    # U_restored = U_restored / np.max(U_restored)
    """
    dg_convs = myHilbert.hilbert(U)
    U_restored = myHilbert.inverse_hilbert(dg_convs)

    ax2.plot(x, dg_convs, label="Image part with gaussian derivative ")
    Uhib = hilbert(U)

    ax2.plot(x, Uhib.imag, label="Image part with hilbert transform")
    ax2.plot(x, U_restored, label="Restored origin signal")

    ax2.legend()

    # fig.savefig("./results/hilbert/H_kernel_aproxed.png")
    fig2.savefig("./results/hilbert/U_coded_and_resrored.png")


    plt.show()

