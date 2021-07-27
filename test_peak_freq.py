import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def get_phi_0(slope, phi_train, x_train):
    s = np.sum( np.cos(phi_train - 2*np.pi*slope*x_train) ) + 1j * np.sum( np.sin(phi_train - 2*np.pi*slope*x_train) )
    phi_0 = np.angle(s)
    return phi_0

def get_Dist(slope, phi_train, x_train):
    phi_0 = get_phi_0(slope, phi_train, x_train)
    D = 2 * (1 - np.mean( np.cos(phi_train - 2*np.pi*slope*x_train - phi_0)  ) )

    return D



x, dx = np.linspace(-0.5, 0.5, 200, retstep=True)
freq = 10
sig = np.cos(2*np.pi * x * freq)
H = 1 / (np.pi * x)

H = H / np.sqrt( np.sum(H**2) )
sig = sig + 1j * np.convolve(sig, H, mode='same')
# sig = hilbert(sig)


phis = np.angle(sig)

res = minimize(get_Dist, x0=2, args=(phis, x), method='Powell' )

print (res.x)


"""
peak_freqs = np.convolve(phis, [1, -1], mode='valid')
peak_freqs[peak_freqs < 0] += 2 * np.pi

peak_freqs /= (2 * np.pi * dx)
peak_freqs = np.convolve(peak_freqs, [0.5, 0.5], mode='valid')

# plt.plot(x, sig.real)
# plt.plot(x, sig.imag)

#plt.plot(x[1:-1], peak_freqs)
plt.hist(peak_freqs, bins=30)
plt.show()
"""