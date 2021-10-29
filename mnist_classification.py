import numpy as np
import h5py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

dsetfilename = "./results/mnist/encoded.hdf5"

with h5py.File(dsetfilename, 'r') as h5file:
    X_train = h5file["X_train"][:].T
    Y_train = h5file["Y_train"][:].T

    X_test = h5file["X_test"][:].T
    Y_test = h5file["Y_test"][:].T




clf = RandomForestClassifier(max_depth=50, n_jobs=-1)
clf.fit(X_train, Y_train)

Y_predicted = clf.predict(X_test)

print(classification_report(Y_test, Y_predicted))
print("##################################################")
# print( np.mean( Y_test = Y_predicted, axis=1 ) )



