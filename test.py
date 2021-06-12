import numpy as np
import matplotlib.pyplot as plt

Len_y = 200
Len_x = 200

xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, Len_y), np.linspace(0.5, -0.5, Len_x))
ws = 6.4
sigma = 1 / (np.sqrt(2) * np.pi * ws)

an = np.pi / 4
xx_ = xx * np.cos(an) - yy * np.sin(an)
yy_ = xx * np.sin(an) + yy * np.cos(an)
image = np.cos(2 * np.pi * xx * ws)

sigma_x = sigma
sigma_y = 0.5 * sigma_x
ricker1 = (-1 + xx_**2/sigma_x**2) * np.exp(-yy_**2/(2*sigma_y**2) - xx_**2/(2*sigma_x**2))/sigma_x**2
ricker1 = ricker1 / np.sqrt(np.sum(ricker1**2))
# 0.005 * (4 - xx_**2 / sigma**2) * np.exp(-(xx_**2 + 4 * yy_**2) / (8 * sigma**2)) / sigma**4
ricker2 = (-1 + xx**2/sigma_x**2) * np.exp(-yy**2/(2*sigma_y**2) - xx**2/(2*sigma_x**2))/sigma_x**2
# 0.005 * (4 - xx**2 / sigma**2) * np.exp(-(xx**2 + 4 * yy**2) / (8 * sigma**2)) / sigma**4
ricker2 = ricker2 / np.sqrt(np.sum(ricker2**2))

fig, axes = plt.subplots(ncols=3)
axes[0].pcolormesh(xx[0, :], yy[:, 0], image, cmap="rainbow", shading="auto")
axes[1].pcolormesh(xx[0, :], yy[:, 0], ricker1, cmap="rainbow", shading="auto")
axes[2].pcolormesh(xx[0, :], yy[:, 0], ricker2, cmap="rainbow", shading="auto")

fig, axes = plt.subplots(ncols=3)
axes[0].plot(xx[0, :], image[100, :] )
axes[0].plot(xx[0, :], -ricker2[100, :]/np.max(-ricker2[100, :]) )

axes[1].plot(xx[0, :], image[100, :] )
axes[1].plot(xx[0, :], -ricker1[100, :]/np.max(-ricker1[100, :]))

print( np.sum(ricker1*image), np.sum(ricker2*image) )
plt.show()