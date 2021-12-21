import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_logarithmic_error

import h5py
import pandas as pd

dsetfilename =  "./results/mnist/encoded_usereg_true.hdf5" # "./results/mnist/encoded_full.hdf5"
feasures_desc_file = './results/mnist/feature_importances_use_reg.csv'
with h5py.File(dsetfilename, 'r') as h5file:
    X_train = h5file['X_train'][:]
    X_test = h5file['X_test'][:]

feasures_desc = pd.read_csv(feasures_desc_file, header=0, index_col=0)


abs_idxes = feasures_desc.index[ feasures_desc['Data type'] == ' abs' ]
abs_idxes = abs_idxes.to_numpy(dtype=np.int)
X_train[abs_idxes, :] = X_train[abs_idxes, :] / np.max( X_train[abs_idxes, :] )
X_test[abs_idxes, :] = X_test[abs_idxes, :] / np.max( X_train[abs_idxes, :] )

peak_freq_idxes = feasures_desc.index[ feasures_desc['Data type'] == ' peak_freq' ]
peak_freq_idxes = peak_freq_idxes.to_numpy(dtype=np.int)
X_train[peak_freq_idxes, :] = X_train[peak_freq_idxes, :] / np.max( X_train[peak_freq_idxes, :] )
X_test[peak_freq_idxes, :] = X_test[peak_freq_idxes, :] / np.max( X_train[peak_freq_idxes, :] )



(X_train_image, Y_train_label), (X_test_image, Y_test_label) = mnist.load_data()
X_train_image = X_train_image.reshape(X_train_image.shape[0], X_train_image.shape[1], X_train_image.shape[2], 1)
X_train_image = X_train_image / 255.0

X_test_image = X_test_image.reshape(X_test_image.shape[0], X_test_image.shape[1], X_test_image.shape[2], 1)
X_test_image = X_test_image / 255.0


# define model
model = Sequential()
model.add(Dense(units=1560, input_dim=1560, activation='relu'))
model.add(Dense(units=10000, activation='relu'))
model.add(Dense(units=22*22*12, activation='relu'))
# reshape vector of activations into 128 feature maps with 5x5
model.add(Reshape((22, 22, 12)))
# double input from 128 5x5 to 1 10x10 feature map
model.add(Conv2DTranspose(4, kernel_size=(4, 4), padding='valid', activation='relu'))
model.add(Conv2DTranspose(1, kernel_size=(4, 4), padding='valid'))
model.compile(optimizer=Adam(learning_rate=1e-3), loss=mean_squared_logarithmic_error)

try:
    model.load_weights('./results/mnist/dnn_weights.hdf5')
except:
    pass

model.fit(x=X_train, y=X_train_image, epochs=10)
model.save_weights('./results/mnist/dnn_weights.hdf5')

Y_pred_image = model.predict(X_test)

np.save('./results/mnist/dnn_decode', Y_pred_image)
