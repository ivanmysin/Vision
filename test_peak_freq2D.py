import numpy as np
import matplotlib.pyplot as plt
from HyperColomn2D import HyperColomn

Len_x = 500
Len_y = 500
xx, yy = np.meshgrid(np.linspace(-1.0, 1.0, Len_y), np.linspace(1.0, -1.0, Len_x))

freq = 5
U = np.cos(2*np.pi*yy*freq)


fig, axes = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)
axes[0].pcolor(xx[0, :], yy[:, 0], U, cmap='gray', shading='auto')

plt.show()