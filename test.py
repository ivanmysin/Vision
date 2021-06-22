import numpy as np
import matplotlib.pyplot as plt
from sympy import diff, lambdify, exp, Symbol
from scipy.signal import convolve2d


## find receptive field form
x = Symbol("x")
y = Symbol("y")
sigma_x = Symbol("sigma_x")
sigma_y = Symbol("sigma_y")
gaussian = exp(- ( x**2/(2*sigma_x**2) + y**2/(2*sigma_y**2)) ) # гауссиана

kernel_expr = diff(gaussian, x, 0) # дифференцируем по х n раз

# компилируем выражение в выполняемую функцию
kernel_func = lambdify( [x, y, sigma_x, sigma_y], kernel_expr)

####################################################################
### Создаем коррдинатную сетку
Len_y = 200 # number points by Y axis in image
Len_x = 200 # number points by X axis in image

xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, Len_y), np.linspace(0.5, -0.5, Len_x))
# xx, yy - двумерные массивы задающие сетку от -0,5 до 0,5 по Len_x и Len_y точек соответсвенно
####################################################################
### Генерируем сигнал
signal_freq = 5.0 # частота синусойды в сигнале
image = np.cos(2 * np.pi * signal_freq * yy)

### рисуем сигнал
_, axes = plt.subplots()
axes.pcolormesh(xx[0, :], yy[:, 0], image, cmap="rainbow", shading="auto")
axes.set_title("Исходный сигнал")
####################################################################
### Вычисляем рецептивные поля под разными углами
angles = np.linspace(0, np.pi, 8, endpoint=False) # 8 равномерно распределенных напрвлений

# вычисляем сигмы по Х и по У
# В данном случае сигмы берутся так чтобы частота сигнала совпадала с максимальной частотой в спектре мексиканской шляпы
sigma_x = 1 / (np.sqrt(2) * np.pi * signal_freq)
sigma_y = 0.5 * sigma_x

fig, axes = plt.subplots(ncols=8, nrows=2, figsize=(24, 6))

# пробегаем по всем напрвлениям
for an_idx, an in enumerate(angles):
    # Поварачиваем координаты на угол
    xx_rot = xx * np.cos(an) - yy * np.sin(an)
    yy_rot = xx * np.sin(an) + yy * np.cos(an)

    kernel = kernel_func(xx_rot, yy_rot, sigma_x, sigma_y) # Вычисляем ядро
    kernel = kernel / np.sqrt(np.sum(kernel**2))           # Нормируем ядро

    response = image * kernel # Вычисляем ответ при совпадении центра рецептивного поля нейрона и гиперколонки
    # Или вычисляем всю свертку
    #response = convolve2d(image, kernel, mode='same')
    axes[0, an_idx].pcolormesh(xx[0, :], yy[:, 0], kernel, cmap="rainbow", shading="auto")
    axes[1, an_idx].pcolormesh(xx[0, :], yy[:, 0], response, cmap="rainbow", shading="auto")

    print( np.rad2deg(an), np.sum(response) )


axes[0, angles.size//2].set_title("Ядра")
axes[1, angles.size//2].set_title("Ответы")


plt.show()