import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import deconvolve, ricker
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
ends[cuted<0] *=  np.exp( -0.5*(cuted[cuted<0] /sigma_ends)**2 )
cuted = t - t[-1] + 5*sigma_ends
ends[cuted>0] *=  np.exp( -0.5*(cuted[cuted>0] /sigma_ends)**2 )

u = np.cos(2*np.pi*200.5*t)
u *= ends




sigma1 = 0.07
sigma2 = 0.09

tker = np.linspace(-0.15, 0.15, 70)
kernel = np.exp(-0.5 * (tker/sigma1)**2)/sigma1 - np.exp(-0.5 * (tker/sigma2)**2)/sigma2
#kernel = (kernel-np.mean(kernel)) / np.std(kernel)

coded = np.convolve(u, kernel, mode="full")




deconv = wiener_deconvolution(coded, kernel, 0)
print(deconv.shape)

plt.plot(deconv)
#plt.ylim(-1, 1)
plt.show()

