import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from HyperColomn2D import HyperColomn


(X_train, y_train), (X_test, y_test) = mnist.load_data()


# print("X_train shape", X_train.shape)
# print("y_train shape", y_train.shape)
# print("X_test shape", X_test.shape)
# print("y_test shape", y_test.shape)

image = X_train[0, :, :] / 255

Len_y, Len_x = image.shape
# print(Len_y, Len_x)

xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, Len_x), np.linspace(0.5, -0.5, Len_y))
delta_x = xx[0, 1] - xx[0, 0]
delta_y = yy[1, 0] - yy[0, 0]

params = {
    "use_circ_regression": False,
}

radiuses = np.asarray([0.1, 0.2, 0.4]) ## np.geomspace(0.1, 0.5, 10) # np.geomspace(0.1, 0.8, 30) #np.asarray([0.01, 0.05, 0.1]) #
angles = np.linspace(-np.pi, np.pi, 10, endpoint=True) # np.linspace(0, 0.5*np.pi, 10, endpoint=True) #np.asarray([0.25*np.pi, ]) #

NGHs = int(radiuses.size*angles.size) # число гиперколонок

hc_centers_x = np.zeros(NGHs, dtype=np.float)
hc_centers_y = np.zeros(NGHs, dtype=np.float)



image_restored_by_HCs = np.zeros( (Len_y, Len_x, NGHs), dtype=np.float )
receptive_fields = np.zeros( (Len_y, Len_x, NGHs), dtype=np.float )

directions = np.linspace(-np.pi, np.pi, 16, endpoint=False)


freq_teor_max = 0.5 / (np.sqrt(delta_x**2 + delta_y**2))
sigma_teor_min = 1 / (2*np.pi*freq_teor_max)
# print(freq_teor_max)

print(sigma_teor_min)
idx = 0

Columns = []

for r in radiuses:
    nsigmas = 8

    sigminimum = sigma_teor_min + 0.2*r # * (1 + r)
    sigmaximum = 10 * sigminimum # 0.005 # функция от r

    print(r, sigminimum)


    sigmas = np.linspace(sigminimum, sigmaximum, nsigmas)

    frequencies = 1 / (2*np.pi*np.geomspace(sigminimum, sigmaximum, 5))  # np.asarray([0.5, 12.0, ])  #

#     for an in angles:
#         xc = r * np.cos(an)
#         yc = r * np.sin(an)
#
#         hc_centers_x[idx] = xc
#         hc_centers_y[idx] = yc
#
#         hc = HyperColomn([xc, yc], xx, yy, directions, sigmas, frequencies=frequencies, params=params)
#
#         # !!!!
#         # fig, axes = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)
#         # axes[0].pcolor(xx[0, :], yy[:, 0], hc.mexican_hats[0][0], cmap='gray', shading='auto')
#
#         Encoded = hc.encode(image)
#
#         Columns.append(hc)
#         idx += 1
#
#
#
# fig, axes = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)
# axes[0].pcolor(xx[0, :], yy[:, 0], image, cmap='gray', shading='auto')
# axes[1].pcolor(xx[0, :], yy[:, 0], image, cmap='gray', shading='auto')
# axes[1].scatter(hc_centers_x, hc_centers_y, s=2.5, color="red")
# axes[1].hlines([0, ], xmin=-0.5, xmax=0.5, color="blue")
# axes[1].vlines([0, ], ymin=-0.5, ymax=0.5, color="blue")
#
# plt.show()
