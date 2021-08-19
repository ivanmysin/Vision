import numpy as np
from skimage import data
from skimage import metrics
from skimage.color import rgb2gray
from HyperColomn2D import HyperColomn
import matplotlib.pyplot as plt

image = data.camera().astype(np.float)
Len_y, Len_x = image.shape

xx, yy = np.meshgrid(np.linspace(-1, 1, Len_x), np.linspace(1, -1, Len_y))

radiuses = np.geomspace(0.1, 0.8, 3)
angles = np.linspace(-np.pi, np.pi, 2, endpoint=True)

NGHs = int(radiuses.size*angles.size) # число гиперколонок

hc_centers_x = np.zeros(NGHs, dtype=np.float)
hc_centers_y = np.zeros(NGHs, dtype=np.float)



image_restored_by_HCs = np.zeros( (Len_y, Len_x, NGHs), dtype=np.float )
receptive_fields = np.zeros( (Len_y, Len_x, NGHs), dtype=np.float )

directions = np.linspace(-np.pi, np.pi, 16, endpoint=False)
idx = 0
for r in radiuses:
    for an in angles:
        hc_centers_x[idx] = r * np.cos(an)
        hc_centers_y[idx] = r * np.sin(an)

        nsigmas = 8
        sigminimum = 0.05 # функция от r
        sigmaximum = 0.1 * sigminimum  # 0.005
        frequencies = np.geomspace(0.1, 50, 5)
        sigmas = np.linspace(sigminimum, sigmaximum, nsigmas)

        hc = HyperColomn([hc_centers_x[idx], hc_centers_y[idx]], xx, yy, directions, sigmas, frequencies=frequencies)

        Encoded = hc.encode(image)
        image_restored_by_HCs[:, :, idx]  = hc.decode(Encoded)

        receptive_field = np.exp(-(yy - hc_centers_y[idx]) ** 2 / (2 * sigminimum**2) - (xx - hc_centers_x[idx])**2 / (2 * sigminimum**2))
        receptive_fields[:, :, idx] = receptive_field

        # Columns.append(hc)
        idx += 1

summ = np.sum(receptive_fields, axis=2)
for i in range(NGHs):
    receptive_fields[:, :, i] /= summ


image_restored = np.sum(image_restored_by_HCs * receptive_fields, axis=2)

fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
axes[0].imshow(image, cmap="gray")
axes[1].imshow(image_restored, cmap="gray")
axes[1].scatter(hc_centers_x, hc_centers_y, color="red")
fig.savefig("./results/hypercolumns2D.png")
plt.show()