import numpy as np
import matplotlib.pyplot as plt
from sympy import diff, lambdify, exp, Symbol

from scipy.ndimage import convolve


def Background(x_, y_):
    A = (x_ - 1.3) * 0.5 + (y_ - 0.7) * 0.8
    return A


def AmplitudesOfGratings(x_, y_, sigma=0.5, cent_x=0.2, cent_y=0.2):
    A = np.exp(-0.5 * ((x_ - cent_x) / sigma) ** 2 - 0.5 * ((y_ - cent_y) / sigma) ** 2)
    return A


def Gratings(freq, xx, yy, sigma=0.5, cent_x=0.2, cent_y=0.2, direction=2.3):
    xx_rot = xx * np.cos(direction * (1 + 2 * xx)) - yy * np.sin(direction * (1 + 2 * xx))
    # yy_rot = xx * np.sin(direction) + yy * np.cos(direction)
    image = np.cos(2 * np.pi * xx_rot * freq) * AmplitudesOfGratings(xx, yy, sigma, cent_x, cent_y) + Background(xx, yy)
    return image


def getHatfunc():
    x = Symbol("x")
    y = Symbol("y")
    sigma_x = Symbol("sigma_x")
    sigma_y = Symbol("sigma_y")
    gaussian = exp(- (x ** 2 / (2 * sigma_x ** 2) + y ** 2 / (2 * sigma_y ** 2)))  # гауссиана
    kernel_expr = diff(gaussian, y, 2)  # дифференцируем по х n раз
    # dGauss_x_expr = diff(gaussian, x, 1)  # дифференцируем по х n раз
    # компилируем выражение в выполняемую функцию
    kernel_func = lambdify([x, y, sigma_x, sigma_y], kernel_expr)
    return kernel_func

##### Stimulus ################################################################
hat_func = getHatfunc()
Freq = 8
Direction = np.pi / 5
ph_0 = 0
Sgm = 0.3

image_shift_x = 0.1
image_shift_y = 0.3
Nx = 200
Ny = 200
sgmGauss = 0.1 #  0.5 / (np.sqrt(2) * np.pi * Freq)
yy, xx = np.meshgrid(np.linspace(-1, 1, Ny), np.linspace(-1, 1, Nx))

image = Gratings(Freq, xx, yy, sigma=Sgm, cent_x=image_shift_x, cent_y=image_shift_y, direction=Direction)

fig, axes = plt.subplots(nrows=5, ncols=3)

for an_idx, angle in enumerate(np.linspace(0, np.pi, 5, endpoint=False)):
    xx_rot = xx * np.cos(angle) - yy * np.sin(angle)
    yy_rot = xx * np.sin(angle) + yy * np.cos(angle)
    hat = hat_func(xx_rot, yy_rot, 0.5*sgmGauss, sgmGauss)
    hat = hat - np.mean(hat)
    hat = hat / np.sqrt(np.sum(hat**2))

    image_filtered = convolve(image, hat)

    axes[an_idx, 0].imshow(image, cmap='gray')
    axes[an_idx, 1].imshow(hat, cmap='gray')
    axes[an_idx, 2].imshow(image_filtered, cmap='gray')

fig.savefig("../results/test_.png")
plt.show()
