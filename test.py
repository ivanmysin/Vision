import numpy as np
import matplotlib.pyplot as plt
from sympy import diff, lambdify, exp, Symbol
from scipy.signal import convolve2d

def get_direction(signal_freq=5, phi_0=0, show_figures=True):
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
    # signal_freq - частота синусойды в сигнале, получаем из аргументов функции
    # phi_0 - фаза синусойды в сигнале, получаем из аргументов функции
    image = np.cos(2 * np.pi * signal_freq * yy + phi_0)

    ### рисуем сигнал
    fig_sig, axes_sig = plt.subplots(ncols=2, figsize=(10, 5))
    axes_sig[0].pcolormesh(xx[0, :], yy[:, 0], image, cmap="rainbow", shading="auto")
    axes_sig[0].set_title("Исходный сигнал")
    ####################################################################
    ### Вычисляем рецептивные поля под разными углами
    angles = np.linspace(0, np.pi, 8, endpoint=False) # 8 равномерно распределенных направлений

    # вычисляем сигмы по Х и по У
    # В данном случае сигмы берутся так чтобы частота сигнала совпадала с максимальной частотой в спектре мексиканской шляпы
    sigma_x = 0.05 # 1 / (np.sqrt(2) * np.pi * signal_freq)
    sigma_y = 0.5 * sigma_x

    fig, axes = plt.subplots(ncols=8, nrows=2, figsize=(24, 6))

    responses = np.zeros_like(angles) # ответы по углам

    # пробегаем по всем направлениям
    for an_idx, an in enumerate(angles):
        # Поварачиваем координаты на угол
        xx_rot = xx * np.cos(an) - yy * np.sin(an)
        yy_rot = xx * np.sin(an) + yy * np.cos(an)

        kernel = kernel_func(xx_rot, yy_rot, sigma_x, sigma_y) # Вычисляем ядро
        kernel = kernel / np.sqrt(np.sum(kernel**2))           # Нормируем ядро

        response = image * kernel # Вычисляем ответ при совпадении центра рецептивного поля нейрона и гиперколонки
        responses[an_idx] = np.sum(response)
        # Или вычисляем всю свертку
        #response = convolve2d(image, kernel, mode='same')
        axes[0, an_idx].pcolormesh(xx[0, :], yy[:, 0], kernel, cmap="rainbow", shading="auto")
        axes[1, an_idx].pcolormesh(xx[0, :], yy[:, 0], response, cmap="rainbow", shading="auto")

        # print( np.rad2deg(an), np.sum(response) )


    axes[0, angles.size//2].set_title("Ядра")
    axes[1, angles.size//2].set_title("Ответы")

    # Рисуем график ответов в зависимости от угла
    axes_sig[1].plot(np.rad2deg(angles), responses)
    axes_sig[1].set_title("Распределение ответов по напрвлениям")
    axes_sig[1].set_ylabel("Ответ")
    axes_sig[1].set_xlabel("Направление, градусы")

    if not show_figures: # не показываем рисунки, если так задано в аргументах
        plt.close(fig=fig_sig)
        plt.close(fig=fig)


    return angles[np.argmax(responses)] # возвращаем направление с максимальным ответом

if __name__ == "__main__":
    signal_freqs = np.linspace(0.1, 80, 40) # вектор частот
    phi_0s = np.linspace(-np.pi, np.pi, 20, endpoint=False)  # вектор начальных фаз

    max_direction_response = np.zeros( [signal_freqs.size, phi_0s.size] , dtype=np.float )
    for freq_idx, signal_freq in enumerate(signal_freqs):
        for phi_idx, phi_0 in enumerate(phi_0s):
            max_direction_response[freq_idx, phi_idx] = get_direction(signal_freq=signal_freq, phi_0=phi_0, show_figures=False)

    max_direction_response = np.rad2deg(max_direction_response)
    fig, axes= plt.subplots(figsize=(10, 5))
    gr = axes.pcolormesh(phi_0s, signal_freqs, max_direction_response, cmap="rainbow", shading="auto")
    axes.set_title("Максимальное направление")
    axes.set_xlabel("Начальная фаза")
    axes.set_ylabel("Частота сигнала")
    plt.colorbar(gr)

    fig.savefig("./results/direnction_selectivity.png")
    plt.show()