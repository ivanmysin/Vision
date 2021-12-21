import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

Y_pred_image = np.load('./results/mnist/dnn_decode_!.npy')
Y_pred_image = Y_pred_image.reshape(10000, 28, 28)

(_, _), (X_test_image, Y_test_label) = mnist.load_data()

for idx in range(X_test_image.shape[0]):
    fig, axes = plt.subplots(ncols=2)
    fig.suptitle('Цифра ' + str(Y_test_label[idx]) )
    axes[0].imshow(X_test_image[idx, :, :], cmap="gray")
    axes[1].imshow(Y_pred_image[idx, :, :], cmap="gray")
    axes[0].set_title('Исходное изображение')
    axes[1].set_title('Восстановленное изображение')

    fig.savefig('./results/mnist/mnist_decoded_by_DNN/'+str(idx+1)+'.png')
    plt.close('all')


