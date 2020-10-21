import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import deconvolve, ricker
from scipy.interpolate import interp1d
from numpy.fft import fft, ifft, ifftshift
def wiener_deconvolution(signal, kernel, lambd):
    "lambd is the SNR"
    kernel = np.hstack((kernel, np.zeros(len(signal) - len(kernel)))) # zero pad the kernel to same length
    H = fft(kernel)
    deconvolved = np.real(ifft(fft(signal)*np.conj(H)/(H*np.conj(H) + lambd**2)))
    return deconvolved

sigma_ends = 0.05

t = np.linspace(0, 1, 900)

ends = np.ones_like(t)
cuted = t - t[0] - 5*sigma_ends
ends[cuted<0] *= np.exp( -0.5*(cuted[cuted<0] /sigma_ends)**2 )
cuted = t - t[-1] + 5*sigma_ends
ends[cuted>0] *= np.exp( -0.5*(cuted[cuted>0] /sigma_ends)**2 )

u = np.cos(2*np.pi*50*t)
u *= ends




sigma1 = 0.02
sigma2 = 0.05

tker = np.linspace(-0.15, 0.15, 70)
kernel = np.exp(-0.5 * (tker/sigma1)**2)/sigma1 - np.exp(-0.5 * (tker/sigma2)**2)/sigma2
kernel = (kernel-np.mean(kernel)) / np.std(kernel)

coded = np.convolve(u, kernel, mode="full")
coded_same = coded[kernel.size//2:u.size+kernel.size//2]



coded_ds = coded_same[::30]
t_ds = t[::30]
t_ds[-1] = t[-1]
f = interp1d(t_ds, coded_ds, kind='cubic')

coded_us = f(t)

# coded_ = np.append(np.zeros(34), coded_)
# coded_ = np.append(coded_, np.zeros(35))
#
# deconv = wiener_deconvolution(coded_, kernel, 0)
# deconv = deconv[:1-kernel.size]

fig, axes = plt.subplots(nrows=3)
axes[0].plot(t, u, color="blue")

axes[1].plot(t, coded_same, color="green", linewidth=3)
axes[1].scatter(t_ds, coded_ds, color="red", s=50)
axes[1].plot(t_ds, coded_ds, color="red", linewidth=1)

axes[2].plot(t, coded_us)

# plt.plot(deconv, color="green", linewidth=2)
# plt.plot(u, color="red")
#
#
# plt.savefig("./results/conv_algorhythm.png")
plt.show()

