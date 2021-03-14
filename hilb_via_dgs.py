import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def sum_gaussian_derivaries(w, x, sigmas):


    dgsum = np.zeros_like(x)

    for idx, sigma in enumerate(sigmas):
        dg = np.exp(-0.5 * (x/sigma)**2) * -x / sigma**2

        dgsum -= w[idx] * dg

    dgsum = dgsum / len(sigmas)

    return dgsum

def hilbert_dgs_diff(w, x, sigmas):
    H =  1 / (np.pi * x)
    sum_dg = sum_gaussian_derivaries(w, x, sigmas)
    E = np.sum( (H - sum_dg)**2 )

    return E



if __name__ == "__main__":
    file_saving_ws = "./results/weights_for_gaussian_derivatives_sum.npy"
    Npoints = 100
    x = np.linspace(-10, 10, Npoints)

    sigmas = np.linspace(0.1, 2, 8)  # [0.05, ] #
    w0 = np.ones(8, dtype=np.float)
    #H_aproxed = sum_gaussian_derivaries(w0, x) #
    H = 1 / (np.pi * x)

    opt_res = minimize(hilbert_dgs_diff, x0=w0, args=(x, sigmas))

    print(opt_res.x)

    np.save(file_saving_ws, opt_res.x)


    H_aproxed = sum_gaussian_derivaries(opt_res.x, x, sigmas) #


    fig = plt.figure()
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

    ax.plot(x[x<0], H_aproxed[x<0], color="blue", label="Aproximated")
    ax.plot(x[x>0], H_aproxed[x>0], color="blue") # , label="Aproximated"

    ax.plot(x[x<0], H[x<0], color="green", label=r"$\frac{1}{\pi \cdot x}$")
    ax.plot(x[x>0], H[x>0], color="green" ) # label=r"$\frac{1}{\pi \cdot x}$"
    ax.legend()

    plt.show()

