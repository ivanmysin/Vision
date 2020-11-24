import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.signal import hilbert

def get_gaussian_derivative(x, sigma, order=1):

    a = 1 / sigma**2
    if order == 1:
        kernel = np.exp(-a * x**2) * -2 * a * x
    elif order == 2:
        kernel = 2 * np.exp(-a * x**2) * ( 1 - 2 * a * x**2)

    # kernel = zscore(kernel)
    return kernel

def get_signal():
    t = np.linspace(0, 1, 900)
    sigma_ends = 0.01
    ends = np.ones_like(t)
    cuted = t - t[0] - 5*sigma_ends
    ends[cuted<0] *=  np.exp( -0.5*(cuted[cuted<0] /sigma_ends)**2 )
    cuted = t - t[-1] + 5*sigma_ends
    ends[cuted>0] *=  np.exp( -0.5*(cuted[cuted>0] /sigma_ends)**2 )

    w = 50 #  np.linspace(2, 25, t.size)
    u = np.cos(2*np.pi*w*t) # + 0.9*np.cos(2*np.pi*5*t)
    u *= ends
    
    return t, u

x = np.linspace(-10, 10, 200)
sigma = 0.2
dg = get_gaussian_derivative(x, sigma, order=1)
t, signal = get_signal()
sig_imag = np.convolve(dg, signal, mode="same")
sig_imag = sig_imag * np.mean(sig_imag**2)
res = np.sqrt(signal**2 + sig_imag**2)

theor_res = np.abs(hilbert(signal)  )

fig, ax = plt.subplots(ncols=1, nrows=2)
ax[0].plot(t, signal, color="green")
ax[0].plot(t, sig_imag, color="m")

ax[1].plot(t, res, color="blue")
ax[1].plot(t, theor_res, color="red")
plt.show()
