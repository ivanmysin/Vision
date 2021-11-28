import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle


def get_dataset(X_image, Y_label):
    Nobj = X_image.shape[0]
    Nfeasures = X_image.shape[1] * X_image.shape[2]
    # print(Nobj, Nfeasures)
    X = np.zeros((Nobj, Nfeasures), dtype=np.float64)
    Y = np.zeros((Nobj, 10), dtype=np.int)

    for idx in range(Nobj):
        X[idx, :] = X_image[idx, :, :].ravel() / 255
        image_class = np.zeros(10, dtype=np.int)
        image_class[Y_label[idx]] = 1
        Y[idx, :] = image_class

    return X, Y
(X_train_image, Y_train_label), (X_test_image, Y_test_label) = mnist.load_data()

print("Преобразуем картинки в матрицу объекты-признаки")
X_train, Y_train = get_dataset(X_train_image, Y_train_label)
X_test, Y_test = get_dataset(X_test_image, Y_test_label)


print("Делаем классификацию")
clf = RandomForestClassifier(n_estimators=500, max_depth=250, n_jobs=-1, min_samples_leaf=2)

with open('./results/mnist/RandomForestClassifier_noncoded', 'wb') as dump_file:
    clf.fit(X_train, Y_train)
    pickle.dump(clf, dump_file)


Y_predicted = clf.predict(X_test)
print(Y_predicted.shape)
print(Y_predicted[0, :])
print(classification_report(Y_test, Y_predicted))
print("##################################################")
print( np.mean( (np.argmax(Y_test, axis=1) == np.argmax(Y_predicted, axis=1)) ) )
