import numpy as np
import h5py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pickle

dsetfilename =  "./results/mnist/encoded_usereg_true.hdf5" # "./results/mnist/encoded_full.hdf5"

with h5py.File(dsetfilename, 'r') as h5file:
    X_train = h5file["X_train"][:]
    Y_train = h5file["Y_train"][:]

    X_test = h5file["X_test"][:]
    Y_test = h5file["Y_test"][:]




clf = RandomForestClassifier(n_estimators=500, max_depth=250, n_jobs=-1, min_samples_leaf=2)

with open('./results/mnist/RandomForestClassifier_usereg_true', 'wb') as dump_file:
    clf.fit(X_train, Y_train)
    pickle.dump(clf, dump_file)


Y_predicted = clf.predict(X_test)
print(Y_predicted.shape)
print(Y_predicted[0, :])
print(classification_report(Y_test, Y_predicted))
print("##################################################")
print( np.mean( (np.argmax(Y_test, axis=1) == np.argmax(Y_predicted, axis=1)) ) )

# importance = clf.feature_importances_
# # feature importance
# for i,v in enumerate(importance):
# 	print('Feature: %0d, Score: %.5f' % (i,v))
# # plot feature importance
# plt.bar([x for x in range(len(importance))], importance)
# plt.show()


