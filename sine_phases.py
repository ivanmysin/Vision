import numpy as np
from skimage import metrics
import matplotlib.pyplot as plt
import matplotlib.patches as pchs

import progect_lib as lib
from copy import copy
from scipy import ndimage
from scipy import signal


default_params = {
    "use_average_grad" : False,
    "shift_xy_mean" : True,  # Сдвигать или нет центр гиперколонки
    "avg_grad_over" : True,  # Усреднять или нет градиент вдоль перпеникуляров к среднему направлению градиента
    "apply_AB_thresh" : False, # Применять или нет обрезание по средним значениях в граничных гиперколонках
    "apply_smooth_grad_dist" : True, # Уменьшать ли вес градиента при удалении от центра гиперколонки
    "apply_grad_in_extrems" : True, # Применять ли градиент, если среднее значение является меньше или больше обоих соседних

    "use_origin"  : False, # Использовать или нет исходное изображение для определения значений в точках А и B
    "n_iters" : 1,
}

path4figsaving = "/home/ivan/PycharmProjects/Vision/results/phases/"
Nganlcells = 500 * 500 # Число ганглиозных клеток
full_area_grad = np.asarray( [55, 60, 90, 60] ) #  градусы верх, низ, наружу, внутрь

Len_y = np.sqrt(Nganlcells * ( full_area_grad[0] + full_area_grad[1] ) / (full_area_grad[2] + full_area_grad[3]))
Len_y = int( np.round(Len_y) )
Len_x = int (np.round( Nganlcells / Len_y ))



x_gr = np.linspace(-full_area_grad[3], full_area_grad[2], Len_x)
y_gr = np.linspace(-full_area_grad[1], full_area_grad[0], Len_y)

x, y = np.meshgrid(np.linspace(-1, 1, Len_x), np.linspace(1, -1, Len_y))
x = x + 2 * (full_area_grad[2]  / (full_area_grad[2] + full_area_grad[3]) - 0.5)

phases = np.linspace(-np.pi, np.pi, 5)

spfr_coef = 150 * 0.5 # 150 размер поля , 0.5 - потому что x меняется от -1 до 1, т.е. два цикла
spacial_freqs = 0.6


res_image_all_phases = 0

for phase_idx, phase0 in enumerate(phases):
    params = copy(default_params)
    params["outfile"] = path4figsaving + "sine_" + str(phase0) + "_rad.png"

    image = 255 * 0.5 * (np.cos(2 * np.pi * spacial_freqs * spfr_coef * x + phase0) + 1)

    res_image, mean_intensity, mean_x, mean_y, abs_steps, angle_steps, xs_AB, ys_AB, cols_AB = lib.make_preobr(image, x, y, params)

    fig, ax = plt.subplots(ncols=1, nrows=2)
    ax[0].pcolor(x_gr, y_gr, image, cmap='gray', vmin=0, vmax=255)
    ax[1].pcolor(x_gr, y_gr, res_image, cmap='gray', vmin=0, vmax=255)

    fig.savefig(params["outfile"])
    plt.close("all")

    res_image_all_phases += res_image 


res_image_all_phases = res_image_all_phases / len(phases)



fig, ax = plt.subplots(ncols=1, nrows=1)
ax.pcolor(x_gr, y_gr, res_image_all_phases, cmap='gray', vmin=0, vmax=255)


fig.savefig(path4figsaving + "_avereged.png")

plt.close(fig)












