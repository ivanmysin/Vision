import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from SetHC import HyperColomns
import h5py


path = "./results/mnist/"
dsetfilename = "encoded.hdf5"
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# X_train = np.random.rand(1, 28, 28)

# print("X_train shape", X_train.shape)
# print("y_train shape", y_train.shape)
# print("X_test shape", X_test.shape)
# print("y_test shape", y_test.shape)

image = X_train[0, :, :] / 255

Len_y, Len_x = image.shape
# print(Len_y, Len_x)
xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, Len_x), np.linspace(0.5, -0.5, Len_y))

hcparams = {
    "use_circ_regression": False,
}

radiuses = np.asarray([0.1, 0.2, 0.4]) ## np.geomspace(0.1, 0.5, 10) # np.geomspace(0.1, 0.8, 30) #np.asarray([0.01, 0.05, 0.1]) #
angles = np.asarray([0.25*np.pi, ]) # np.linspace(-np.pi, np.pi, 30, endpoint=False) # np.linspace(0, 0.5*np.pi, 10, endpoint=True)  #
directions = np.linspace(-np.pi, np.pi, 16, endpoint=False)


V1 = HyperColomns(radiuses, angles, directions, xx, yy, hcparams)


keys = ["train", "test"]

with h5py.File(path+dsetfilename, 'w') as h5file:

    for key in keys:
        if key == "train":
            X = X_train
            Y = Y_train
        else:
            X = X_test
            Y = Y_test

        Nobj = X.shape[0]
        for idx in range(Nobj):
            image = X[idx, :, :] / 255
            image_class = np.zeros(10, dtype=np.int)
            image_class[ Y[idx] ] = 1
            encoded = V1.encode(image)
            # image_restored = V1.decode(encoded)
            image_feasures = V1.encoded2vector(encoded)

            if idx == 0:
                X_dset = h5file.create_dataset('X_'+key, (image_feasures.size, Nobj), dtype='f16' )
                Y_dset = h5file.create_dataset('Y_'+key, (10, Nobj), dtype='i8')

            X_dset[:, idx] = image_feasures
            Y_dset[:, idx] = image_class


# fig, axes = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)
# axes[0].pcolor(xx[0, :], yy[:, 0], image, cmap='gray', shading='auto')
# axes[1].pcolor(xx[0, :], yy[:, 0], image, cmap='gray', shading='auto')
# axes[1].scatter(V1.hc_centers_x, V1.hc_centers_y, s=2.5, color="red")
#
# # for circ in circles:
# #     axes[1].add_artist(circ)
#
# axes[1].hlines([0, ], xmin=-0.5, xmax=0.5, color="blue")
# axes[1].vlines([0, ], ymin=-0.5, ymax=0.5, color="blue")
#
# plt.show()
