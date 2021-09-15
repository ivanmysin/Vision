import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar

def get_phi_0(slope, phi_train, x_train):
    s = np.sum( np.cos(phi_train - 2*np.pi*slope*x_train) ) + 1j * np.sum( np.sin(phi_train - 2*np.pi*slope*x_train) )
    phi_0 = np.angle(s)
    return phi_0

def get_Dist(slope, phi_train, x_train):
    phi_0 = get_phi_0(slope, phi_train, x_train)
    D = 2 * (1 - np.mean( np.cos(phi_train - 2*np.pi*slope*x_train - phi_0)  ) )
    if slope < 0.5:
        D += 100

    return D



x, dx = np.linspace(-0.5, 0.5, 200, retstep=True)
freq = 5
sig = 0.5 * ( np.cos(2*np.pi * x * freq) + 1)
H = 1 / (np.pi * x)

H = H / np.sqrt( np.sum(H**2) )
analitic_sig = sig + 1j * np.convolve(sig, H, mode='same')
# sig = hilbert(sig)


phases_train = np.angle(analitic_sig)
x_train = x

freq = 1.5
npzfile = np.load("./results/saved.npz")
phases_train = npzfile["arr_0"]
x_train  = npzfile["arr_1"]




fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
axes[0].scatter(x_train, phases_train, s=1.5, label="Convolve with aproximated hilbert kernel")
#plt.plot(x_train, np.angle(hilbert(sig)), label="Scipy hilbert")

axes[0].set_xlabel("Значения Х, вдоль направления")
axes[0].set_ylabel("Фаза, вдоль направления, рад")
#res = minimize(get_Dist, x0=2, args=(phis, x), method='Powell' )

res = minimize_scalar(get_Dist, args=(phases_train, x_train), bounds=[0.5, 50], method='brent')

print ("Наклон после оптимизации ", res.x)

slopes = np.linspace(0.8, 50, 200)
D = np.zeros_like(slopes)
for idx, slope in enumerate(slopes):
    D[idx] = get_Dist(slope, phases_train, x_train)

slope_2 = slopes[ np.argmin(D) ]
print("Наклон после перебора ", slope_2)

trend_line = 2*np.pi*slope_2*x_train + get_phi_0(slope_2, phases_train, x_train)

axes[0].plot(x_train, trend_line, color="r", label="Линия тренда")

axes[0].legend()
axes[1].plot(slopes, D)
axes[1].set_xlabel("Наклон, Гц")
axes[1].set_ylabel("Оптимизируемая функция")
plt.show()

fig.savefig("./results/phases.png")



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