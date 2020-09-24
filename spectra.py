import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(-10, 10, 1000)
dt = t[1] - t[0]
a = 100.5
freqs = np.fft.fftshift( np.fft.fftfreq(t.size, dt) )


kernel = np.exp(-a * t**2) * -2 * a * t

"""
theor_spec = np.abs( freqs * 1j * np.exp(-(np.pi * freqs)**2 / a) * np.sqrt(np.pi / a) )
theor_spec = theor_spec / np.sum(theor_spec)


fft_spec = np.fft.fftshift( np.abs(np.fft.fft(kernel) ) ) # / kernel.size 3.2 
fft_spec = fft_spec / np.sum(fft_spec)


fig, ax = plt.subplots(ncols=2)

ax[0].plot(t, kernel)
ax[1].plot(freqs, theor_spec, color="blue", label="theor", linewidth=2)
ax[1].plot(freqs, fft_spec, color="green", label="fft")
ax[1].legend()

"""


"""
kernel = 2 * np.exp(-a * t**2) * ( 1 - 2 * a * t**2)
theor_spec = np.abs( -freqs**2 * np.exp(-(np.pi * freqs)**2 / a) * np.sqrt(np.pi / a) )
theor_spec = theor_spec / np.sum(theor_spec)
fft_spec = np.fft.fftshift( np.abs(np.fft.fft(kernel) ) )
fft_spec = fft_spec / np.sum(fft_spec)

fig, ax = plt.subplots(ncols=2)

ax[0].plot(t, kernel)
ax[1].plot(freqs, theor_spec, color="blue", label="theor", linewidth=2)
ax[1].plot(freqs, fft_spec, color="green", label="fft")
ax[1].legend()
"""
sigma1 = 0.01
sigma2 = 0.03
a1 = 0.5/sigma1**2
a2 = 0.5/sigma2**2

kernel = np.exp(-0.5 * (t/sigma1)**2)/sigma1 - np.exp(-0.5 * (t/sigma2)**2)/sigma2
theor_spec = np.abs( np.exp(-(np.pi * freqs)**2 / a1) * np.sqrt(np.pi / a1)/sigma1 -  np.exp(-(np.pi * freqs)**2 / a2) * np.sqrt(np.pi / a2)/sigma2 )
theor_spec = theor_spec / np.sum(theor_spec)
fft_spec = np.fft.fftshift( np.abs(np.fft.fft(kernel) ) )
fft_spec = fft_spec / np.sum(fft_spec)

fig, ax = plt.subplots(ncols=2)

ax[0].plot(t, kernel)
ax[1].plot(freqs, theor_spec, color="blue", label="theor", linewidth=2)
ax[1].plot(freqs, fft_spec, color="green", label="fft")
ax[1].legend()

plt.show()
