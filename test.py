import numpy as np
import matplotlib.pyplot as plt


points = 100

xvect = np.arange(0, points) - (points - 1.0) / 2
wp = 2 * np.pi * 0.03

dx = xvect[1] - xvect[0]
sigma = 1.0 / wp
a = 0.5 / sigma**2


kernel = np.sqrt(a / np.pi) * np.exp(-a * xvect**2)  * (-2.0 * a * xvect)
# kernel = kernel / np.sum(kernel**2)

# print(np.sum(kernel))
# print(np.sum(kernel**2))
kernel_fft = 2 * np.abs( np.fft.rfft(kernel) ) / kernel.size

freqs = np.fft.rfftfreq(kernel.size, xvect[1]-xvect[0])

plt.plot(freqs, kernel_fft)
plt.show()


"""
points = 100

xvect = np.arange(0, points) - (points - 1.0) / 2
wp = 2 * np.pi * 0.05

a = np.sqrt(2) / wp
A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
wsq = a**2
vec = np.arange(0, points) - (points - 1.0) / 2
xsq = vec**2
mod = (1 - xsq / wsq)
gauss = np.exp(-xsq / (2 * wsq))

ricker = A * mod * gauss

ricker_fft = 2 * np.abs( np.fft.rfft(ricker) ) / ricker.size

freqs = np.fft.rfftfreq(ricker.size, xvect[1]-xvect[0])
print( np.mean(np.abs(np.fft.fft(ricker)**2) ))
print( np.sum(ricker**2) )


wp = wp / 2 / np.pi
R = 2 * (freqs**2) / (np.sqrt(np.pi) * wp**3) * np.exp(-(freqs**2/wp**2) )
R = R / np.sum(R)
ricker_fft = ricker_fft / np.sum(ricker_fft)


plt.plot(freqs, ricker_fft)
plt.plot(freqs, R)
plt.show()
"""
