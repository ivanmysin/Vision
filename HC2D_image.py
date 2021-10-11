import numpy as np
from skimage import data
from skimage import metrics
from skimage.color import rgb2gray
from HyperColomn2D import HyperColomn
import matplotlib.pyplot as plt


def get_gratings(freq, sigma=0.1, cent_x=0.2, cent_y=0.2, Len_x=500, Len_y=500, direction=2.3):
    xx, yy = np.meshgrid(np.linspace(-1, 1, Len_x), np.linspace(1, -1, Len_y))
    xx_rot = xx * np.cos(direction) - yy * np.sin(direction)
    image = 0.5*(np.cos(2*np.pi * xx_rot * freq)+1) * np.exp( -0.5*((xx - cent_x)/sigma)**2 - 0.5*((yy - cent_y)/sigma)**2 )
    #image = 1 - image

    # fig, axes = plt.subplots( figsize=(5, 5), sharex=True, sharey=True)
    # axes.pcolor(xx[0, :], yy[:, 0], image, cmap='gray', shading='auto')  # imshow(image, cmap="gray")
    #
    # plt.show(block=False)

    return image


# image = data.camera().astype(np.float)
# image = image / 255
image = get_gratings(15, sigma=0.1, cent_x=0.1, cent_y=0.1)

image += np.random.rand( *image.shape ) * 0.01

Len_y, Len_x = image.shape
# print(Len_y, Len_x)

xx, yy = np.meshgrid(np.linspace(-1, 1, Len_x), np.linspace(1, -1, Len_y))
delta_x = xx[0, 1] - xx[0, 0]
delta_y = yy[1, 0] - yy[0, 0]

params = {
    "use_circ_regression": True,
}

radiuses = np.asarray([0.05, 0.1, 0.2]) ## np.geomspace(0.1, 0.5, 10) # np.geomspace(0.1, 0.8, 30) #np.asarray([0.01, 0.05, 0.1]) #
angles =  np.linspace(-np.pi, np.pi, 10, endpoint=True) #np.asarray([0.25*np.pi, ]) # np.linspace(0, 0.5*np.pi, 10, endpoint=True) #

NGHs = int(radiuses.size*angles.size) # число гиперколонок

hc_centers_x = np.zeros(NGHs, dtype=np.float)
hc_centers_y = np.zeros(NGHs, dtype=np.float)



image_restored_by_HCs = np.zeros( (Len_y, Len_x, NGHs), dtype=np.float )
receptive_fields = np.zeros( (Len_y, Len_x, NGHs), dtype=np.float )

directions = np.linspace(-np.pi, np.pi, 32, endpoint=False)


freq_teor_max = 0.5 / (np.sqrt(delta_x**2 + delta_y**2))
sigma_teor_min = 1 / (2*np.pi*freq_teor_max)
print(freq_teor_max)
idx = 0
for r in radiuses:
    nsigmas = 8

    sigminimum = sigma_teor_min + 0.2*r # * (1 + r)
    sigmaximum = 10 * sigminimum # 0.005 # функция от r

    sigmas = np.linspace(sigminimum, sigmaximum, nsigmas)


    freq_max = freq_teor_max * 0.8*(1 - r)
    freq_min = 0.05*freq_max

    # print(freq_min, freq_max)
    frequencies = np.geomspace(freq_min, freq_max, 5) # np.asarray([12.0, ])  #

    print(r, frequencies)

    for an in angles:
        xc = r * np.cos(an)
        yc = r * np.sin(an)

        hc_centers_x[idx] = xc
        hc_centers_y[idx] = yc

        hc = HyperColomn([xc, yc], xx, yy, directions, sigmas, frequencies=frequencies, params=params)

        # !!!!
        # fig, axes = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)
        # axes[0].pcolor(xx[0, :], yy[:, 0], hc.mexican_hats[0][0], cmap='gray', shading='auto')


        Encoded = hc.encode(image)
        image_restored_by_HCs[:, :, idx] = hc.decode(Encoded)  # np.random.rand(xx.size).reshape(xx.shape) #

        sigma_rep_field = sigmaximum
        receptive_field = np.exp( -0.5*((yy - yc)/sigma_rep_field)**2 - 0.5*((xx - xc)/sigma_rep_field)**2 )
        receptive_fields[:, :, idx] = receptive_field

        # Columns.append(hc)
        idx += 1

summ = np.sum(receptive_fields, axis=2)
summ[summ == 0] = 0.001
for i in range(NGHs):
    receptive_fields[:, :, i] /= summ


image_restored = np.sum(image_restored_by_HCs*receptive_fields, axis=2)

fig, axes = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)
axes[0].pcolor(xx[0, :], yy[:, 0], image, cmap='gray', shading='auto')   # imshow(image, cmap="gray")
axes[1].pcolor(xx[0, :], yy[:, 0], image_restored, cmap='gray', shading='auto')
axes[1].scatter(hc_centers_x, hc_centers_y, s=2.5, color="red")
axes[1].hlines([0, ], xmin=-1, xmax=1, color="red")
axes[1].vlines([0, ], ymin=-1, ymax=1, color="red")

fig.savefig("./results/hypercolumns_image_restore2D.png")
plt.show()
