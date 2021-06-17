import numpy as np
import matplotlib.pyplot as plt
from sympy import diff, lambdify, exp, Symbol


Npoints = 1000
Len_x = Npoints
Len_y = Npoints
xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, Len_y), np.linspace(0.5, -0.5, Len_x))
x = Symbol("x") #  xx[0, :]
y = Symbol("y") # yy[:, 0]

sigma_x = Symbol("sigma_x") #
sigma_y = Symbol("sigma_y") #
gaussian = exp(- ( x**2/(2*sigma_x**2) + y**2/(2*sigma_y**2)) )  #/ (8 * sigma**2)) / (4 * np.pi * sigma**2)
    # exp(-(x**2 + 4 * y**2) / (8 * sigma**2)) / (4 * np.pi * sigma**2)

dgaussian = diff(gaussian, x, 1)
# np.exp(-(xx**2 + 4 * yy**2) / (8 * sigma**2)) / (4 * np.pi * sigma**2)
# ricker2D = xx / (64*np.pi*sigma**8) * np.exp( (-xx**2 - 4*yy**2) / (8 * sigma**2))

ricker2D_expr = diff(gaussian, x, 2)
ricker2D_func = lambdify( [x, y, sigma_x, sigma_y], ricker2D_expr)

#print(ricker2D_expr)
print(dgaussian)

sigma = 0.05
ricker2D = ricker2D_func(xx, yy, sigma, 0.5*sigma)

x = xx[0, :]
y = yy[:, 0]

dx = x[1] - x[0]
freqs = np.fft.rfftfreq(x.size, dx)
fft_spec = np.abs(np.fft.rfft(ricker2D[Npoints//2, :]) )  # / kernel.size 3.2
fft_spec = fft_spec / np.sum(fft_spec)

frq_m = 1 / (np.sqrt(2) * np.pi * sigma)

ricker = (1.0 - 2.0*(np.pi**2)*(frq_m**2)*(x**2)) * np.exp(-(np.pi**2)*(frq_m**2)*(x**2))
ricker = ricker / np.std(ricker)
rick2 = -ricker2D[Npoints//2, :] / np.std(ricker2D[Npoints//2, :])

fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].plot(x, rick2, linewidth=5)
axes[0].plot(x, ricker)
axes[1].plot(freqs, fft_spec)
axes[1].vlines(frq_m, 0, 0.4, color="red")


plt.figure()
plt.pcolormesh(x, y, ricker2D, cmap="rainbow", shading="auto")
plt.show()