import numpy as np
from skimage import data
from skimage.transform import rescale, resize, downscale_local_mean
# from skimage import metrics
# from skimage.color import rgb2gray
from HyperColomn2D import HyperColomn
import matplotlib.pyplot as plt


def get_gratings(freq, sigma=0.1, cent_x=0.2, cent_y=0.2, Len_x=200, Len_y=200, direction=2.3):
    xx, yy = np.meshgrid(np.linspace(-1, 1, Len_x), np.linspace(1, -1, Len_y))
    xx_rot = xx * np.cos(direction) - yy * np.sin(direction)
    image = 0.5*(np.cos(2*np.pi * xx_rot * freq)+1) * np.exp( -0.5*((xx - cent_x)/sigma)**2 - 0.5*((yy - cent_y)/sigma)**2 )
    return image


image = data.camera().astype(np.float64) / 255
# print(image.shape)
# image = rescale(image, 0.4, anti_aliasing=False)
# print(image.shape)
# plt.imshow(image, cmap='gray')
# plt.show()


# image = get_gratings(15, sigma=0.2, cent_x=0.2, cent_y=0.2)
# image += np.random.rand( *image.shape ) * 0.01

Len_y, Len_x = image.shape
# print(Len_y, Len_x)

xx, yy = np.meshgrid(np.linspace(-1, 1, Len_x), np.linspace(1, -1, Len_y))
delta_x = xx[0, 1] - xx[0, 0]
delta_y = yy[1, 0] - yy[0, 0]

x = np.copy(xx[0, :])
y = np.copy(yy[:, 0])


params = {
    "use_circ_regression": False,
}

radiuses = np.geomspace(0.05, 0.95, 30) #np.asarray([0.1, 0.7, 0.9, ]) ##  np.geomspace(0.1, 0.8, 30) #np.asarray([0.01, 0.05, 0.1]) #
angles = np.linspace(-np.pi, np.pi, 30, endpoint=True) #np.asarray( [0.25*np.pi, 1.25*np.pi] ) #   # np.linspace(0, 0.5*np.pi, 10, endpoint=True) #

NGHs = int(radiuses.size*angles.size) # число гиперколонок

hc_centers_x = np.zeros(NGHs, dtype=np.float64)
hc_centers_y = np.zeros(NGHs, dtype=np.float64)



image_restored_by_HCs = np.zeros( (Len_y, Len_x, NGHs), dtype=np.float64 )
AA_by_HCs = np.zeros( (Len_y, Len_x, NGHs), dtype=np.float64 )
Phases_HCs  = np.zeros( (Len_y, Len_x, NGHs), dtype=np.float64 )
Mean_HCs  = np.zeros( (Len_y, Len_x, NGHs), dtype=np.float64 )


receptive_fields = np.zeros( (Len_y, Len_x, NGHs), dtype=np.float64 )
receptive_fields_A = np.zeros( (Len_y, Len_x, NGHs), dtype=np.float64 )

directions = np.linspace(-np.pi, np.pi, 32, endpoint=False)


freq_teor_max = 0.5 / (np.sqrt(delta_x**2 + delta_y**2))
sigma_teor_min = 1 / (2*np.pi*freq_teor_max)
# print(freq_teor_max)
idx = 0
for r in radiuses:
    nsigmas = 8

    sigminimum = sigma_teor_min + 0.01*r  # * (1 + r)
    sigmaximum = 10 * sigminimum # 0.005 # функция от r

    sigmas = np.linspace(sigminimum, sigmaximum, nsigmas)


    freq_max = freq_teor_max * 0.8*(1 - r)
    freq_min = 0.05*freq_max

    # print(freq_min, freq_max)
    frequencies = np.asarray([12.0, ])  #np.geomspace(freq_min, freq_max, 5) #5

    # print(r, frequencies)

    for an in angles:
        xc = r * np.cos(an)
        yc = r * np.sin(an)

        hc_centers_x[idx] = xc
        hc_centers_y[idx] = yc

        hc = HyperColomn([xc, yc], xx, yy, directions, sigmas, frequencies=frequencies, params=params)


        Encoded = hc.encode(image)
        # fig, axes = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)
        # axes[0].pcolor(x, y, image, cmap='gray', shading='auto')
        #
        # # axes[0].scatter(x[hc.cent_x_idx] + xc, y[hc.cent_y_idx] + yc, color='red')
        # # axes[1].pcolor(x, y, u_imag, cmap='gray', shading='auto')
        #
        # axes[1].scatter(xc, yc, color='red', s=10)
        #
        # # print(Encoded[0]['abs'])
        # plt.show()

        # abses[idx] = Encoded[0]['abs'] # np.sum( hc.gaussian * image) #



        #image_restored_by_HCs[:, :, idx] = hc.decode(Encoded)  # np.random.rand(xx.size).reshape(xx.shape) #
        image_restored_by_HCs[:, :, idx], AA_by_HCs[:, :, idx], Phases_HCs[:, :, idx],  Mean_HCs[:, :, idx] = hc.decode(Encoded)

        sigma_rep_field = 0.6*sigmaximum #  sigminimum
        receptive_fields[:, :, idx] = np.exp( -0.5*((yy - yc)/sigma_rep_field)**2 - 0.5*((xx - xc)/sigma_rep_field)**2 )

        receptive_fields_A[:, :, idx] = 1 / ((xx - xc) ** 2 + (yy - yc) ** 2)


        # fig, axes = plt.subplots(ncols=1, figsize=(10, 5), sharex=True, sharey=True)
        # axes.pcolor(x, y, receptive_field, cmap='gray', shading='auto')
        # axes.scatter(xc, yc, color='red')
        # plt.show()
        # Columns.append(hc)
        idx += 1

# fig, axes = plt.subplots(nrows=2, figsize=(10, 5))
# axes[0].scatter(hc_centers_x, abses)
# axes[1].scatter(hc_centers_y, abses)
# plt.show()

summ = np.sum(receptive_fields, axis=2)
summ[summ == 0] = 0.001
for i in range(NGHs):
    receptive_fields[:, :, i] /= summ

summ = np.sum(receptive_fields_A, axis=2)
summ[summ == 0] = 0.001
for i in range(NGHs):
    receptive_fields_A[:, :, i] /= summ

AA = np.sum(AA_by_HCs * receptive_fields_A, axis=2)
Phis = np.sum(Phases_HCs * receptive_fields_A, axis=2)
Mean = np.sum(Mean_HCs * receptive_fields_A, axis=2)

image_restored_A = AA * np.cos(Phis) + Mean
image_restored_I = np.sum(image_restored_by_HCs * receptive_fields, axis=2)


fig, axes = plt.subplots(ncols=3, figsize=(30, 10), sharex=True, sharey=True)
axes[0].pcolor(x, y, image, cmap='gray', shading='auto')   # imshow(image, cmap="gray")
axes[1].pcolor(x, y, image_restored_A, cmap='gray', shading='auto')
axes[2].pcolor(x, y, image_restored_I, cmap='gray', shading='auto')

for ax in axes:
    ax.scatter(hc_centers_x, hc_centers_y, s=2.5, color="red")
    ax.hlines([0, ], xmin=-1, xmax=1, color="green")
    ax.vlines([0, ], ymin=-1, ymax=1, color="green")

fig.savefig("./results/camera.png")
plt.show()
