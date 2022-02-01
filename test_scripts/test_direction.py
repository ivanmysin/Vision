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

    hat_expr = diff(gaussian, x, 2) # дифференцируем по х n раз
    #dg_expr = diff(gaussian, x, 1) # дифференцируем по х n раз

    # компилируем выражение в выполняемую функцию
    hat_func = lambdify( [x, y, sigma_x, sigma_y], hat_expr)
    #dg_func = lambdify( [x, y, sigma_x, sigma_y], dg_expr)

    #gaussian_func =  lambdify( [x, y, sigma_x, sigma_y], gaussian)

    ####################################################################
    ### Создаем коррдинатную сетку
    Len_y = 100 # number points by Y axis in image
    Len_x = 100 # number points by X axis in image

    xx, yy = np.meshgrid(np.linspace(-1.0, 1.0, Len_y), np.linspace(1.0, -1.0, Len_x))
    # xx, yy - двумерные массивы задающие сетку от -0,5 до 0,5 по Len_x и Len_y точек соответсвенно
    ####################################################################
    ### Генерируем сигнал
    # signal_freq - частота синусойды в сигнале, получаем из аргументов функции
    # phi_0 - фаза синусойды в сигнале, получаем из аргументов функции
    image = 0.5 * (np.cos(2 * np.pi * signal_freq * yy + phi_0) + 1)

    ### рисуем сигнал
    # fig_sig, axes_sig = plt.subplots(ncols=2, figsize=(10, 5))
    # axes_sig[0].pcolormesh(xx[0, :], yy[:, 0], image, cmap="rainbow", shading="auto")
    # axes_sig[0].set_title("Исходный сигнал")
    ####################################################################
    ### Вычисляем рецептивные поля под разными углами
    angles = np.linspace(-np.pi, np.pi, 32, endpoint=False) # 8 равномерно распределенных направлений

    # вычисляем сигмы по Х и по У
    # В данном случае сигмы берутся так чтобы частота сигнала совпадала с максимальной частотой в спектре мексиканской шляпы
    sigma_x = 1 / signal_freq #(np.sqrt(2) * np.pi * 0.05 #  0.005
    sigma_y = 0.5 * sigma_x

    #fig, axes = plt.subplots(ncols=angles.size, nrows=2, figsize=(24, 6))

    responses = np.zeros_like(angles) # ответы по углам

    # gauss = gaussian_func(xx, yy, 10*sigma_x, 10*sigma_x)
    # пробегаем по всем направлениям
    for an_idx, an in enumerate(angles):
        # Поварачиваем координаты на угол
        xx_rot = xx * np.cos(an) - yy * np.sin(an)
        yy_rot = xx * np.sin(an) + yy * np.cos(an)

        hat = hat_func(yy_rot,xx_rot, sigma_x, sigma_y) # Вычисляем ядро
        hat = hat / np.sqrt(np.sum(hat**2))           # Нормируем ядро

        # dgaus = dg_func(xx_rot, yy_rot, sigma_x, sigma_y) # Вычисляем ядро
        # dgaus = dgaus / np.sqrt(np.sum(dgaus ** 2))  # Нормируем ядро

        # Вычисляем ответ при совпадении центра рецептивного поля нейрона и гиперколонки
        # response = np.max( convolve2d(image, hat, mode='valid') )#np.sum(image * hat) #+ np.abs(np.sum(image * dgaus)) #  #
        responses[an_idx] = np.abs( np.sum(image * hat) )  # response
        # Или вычисляем всю свертку
        # response_full = convolve2d(image, hat, mode='same')
        # axes[0, an_idx].pcolormesh(xx[0, :], yy[:, 0], hat, cmap="rainbow", shading="auto")
        # axes[1, an_idx].pcolormesh(xx[0, :], yy[:, 0], response_full, cmap="rainbow", shading="auto")
        # axes[1, an_idx].set_title(str(np.max(np.round(response_full, 2))))
        # axes[0, an_idx].set_title(str(np.round(response, 2)))


        #print( np.rad2deg(an), response, np.max(response_full) )


    # axes[0, angles.size//2].set_title("Ядра")
    # axes[1, angles.size//2].set_title("Ответы")
    #
    # # Рисуем график ответов в зависимости от угла
    # axes_sig[1].plot(np.rad2deg(angles), responses)
    # axes_sig[1].set_title("Распределение ответов по напрвлениям")
    # axes_sig[1].set_ylabel("Ответ")
    # axes_sig[1].set_xlabel("Направление, градусы")

    # if not show_figures: # не показываем рисунки, если так задано в аргументах
    #     plt.close(fig=fig_sig)
    #     plt.close(fig=fig)

    direction_max_resp = angles[np.argmin(responses)]
    # direction_max_resp = angles[np.argmax(responses)]  # возвращаем направление с максимальным ответом
    # vectors_responses = responses * np.exp(1j * angles)
    # near_angles_idxs = np.argsort( np.cos(direction_max_resp - angles) )
    # summed_angles_idxs = near_angles_idxs[:angles.size//2]

    # direction_max_resp = np.angle( np.sum(vectors_responses[summed_angles_idxs]) ) # возвращаем направление после векторного усреднения ответов
    # if direction_max_resp < 0:
    #     direction_max_resp += np.pi
    # print(responses)
    #responses = np.abs(responses)
    # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # ax.scatter(angles, responses, color='blue', label='Модули ответов')
    # ax.plot([0.5 * np.pi, 0.5 * np.pi], [0, np.max(responses)], color='red', label='Истинное направление')
    # ax.plot([direction_max_resp, direction_max_resp], [0, np.max(responses)], color='green', label='Среднее направление')
    #
    # ax.scatter(angles[summed_angles_idxs], np.max(responses) * np.ones(summed_angles_idxs.size),
    #            color='orange', label='Усредняемые направления')
    # fig.legend() # bbox_to_anchor=(0.5, 0)
    # plt.show()



    return direction_max_resp

if __name__ == "__main__":
    signal_freqs =  np.linspace(1, 30, 25) # np.asarray([10.0, ]) # вектор частот
    phi_0s = np.linspace(-np.pi, np.pi, 10, endpoint=False)  #np.asarray([0, ])  #   #  вектор начальных фаз

    max_direction_response = np.zeros( [signal_freqs.size, phi_0s.size] , dtype=np.float64 )
    for freq_idx, signal_freq in enumerate(signal_freqs):
        for phi_idx, phi_0 in enumerate(phi_0s):
            max_direction_response[freq_idx, phi_idx] = get_direction(signal_freq=signal_freq, phi_0=phi_0, show_figures=False)
            # print("phi_0 = ", phi_0, "direction = ", np.rad2deg(max_direction_response[freq_idx, phi_idx]) )

    max_direction_response[max_direction_response < 0] += np.pi
    max_direction_response_deg = np.rad2deg(max_direction_response)
    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
    # axes[0].plot(phi_0s, max_direction_response_deg.ravel())
    gr = axes[0].pcolormesh(phi_0s, signal_freqs, max_direction_response_deg, cmap="rainbow", shading="auto")
    axes[0].set_title("Максимальное направление")
    axes[0].set_xlabel("Начальная фаза")
    axes[0].set_ylabel("Частота сигнала")
    plt.colorbar(gr, ax=axes[0])



    dev = np.abs(np.cos(max_direction_response - 0.5*np.pi)) #,  0.5*(np.cos( max_direction_response + 0.5*np.pi) + 1))
    # axes[1].plot(phi_0s, dev.ravel())
    gr2 = axes[1].pcolormesh(phi_0s, signal_freqs, dev, cmap="rainbow", shading="auto")
    axes[1].set_title("Точность определения направления")
    axes[1].set_xlabel("Начальная фаза")
    axes[1].set_ylabel("Частота сигнала")
    plt.colorbar(gr2, ax=axes[1])


    fig.savefig("../results/detect_direction.png")
    plt.show()
