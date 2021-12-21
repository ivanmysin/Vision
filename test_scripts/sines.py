import numpy as np
from skimage import metrics, data
import matplotlib.pyplot as plt
import matplotlib.patches as pchs

import progect_lib as lib1
import progect_lib2 as lib2
from copy import copy
from scipy import ndimage
from scipy import signal


default_params = {
    "use_average_grad" : True,
    "shift_xy_mean" : True,  # Сдвигать или нет центр гиперколонки
    "avg_grad_over" : True,  # Усреднять или нет градиент вдоль перпеникуляров к среднему направлению градиента
    "apply_AB_thresh" : True, # Применять или нет обрезание по средним значениях в граничных гиперколонках
    "apply_smooth_grad_dist" : False, # Уменьшать ли вес градиента при удалении от центра гиперколонки
    "apply_grad_in_extrems" : False, # Применять ли градиент, если среднее значение является меньше или больше обоих соседних

    "use_origin"  : False, # Использовать или нет исходное изображение для определения значений в точках А и B
    "n_iters" : 1,
}


def get_csf(image, x, y, params):
    res_image, mean_intensity, mean_x, mean_y, abs_steps, angle_steps, xs_AB, ys_AB, cols_AB = lib.make_preobr(image, x, y, params)

    maxes = ndimage.maximum_filter(res_image, size=wind4csf, mode="wrap")
    mines = ndimage.minimum_filter(res_image, size=wind4csf, mode="wrap")

    michelson_contrast = (maxes - mines) / (maxes + mines)

    fig, ax = plt.subplots(nrows=1, ncols=4) #  , figsize=(15, 15)

    ax[0].imshow(image, cmap='gray', vmin=0, vmax=255)
    ax[1].imshow(res_image, cmap='gray', vmin=0, vmax=255)
    ax[2].imshow(michelson_contrast, cmap='gray', vmin=0, vmax=1)


    center_line = michelson_contrast[Len_x//2, Len_y//2:]
    # wind = signal.parzen(35)
    # center_line = np.convolve(center_line, wind, "same")

    ax[3].plot(center_line)

    fig.savefig(params["outfile"] + ".png", dpi=200)

    plt.close('all')

    return center_line

def get_contrast(image):
    im_max = np.max(image)
    im_min = np.min(image)

    contr = (im_max - im_min) / (im_max + im_min + 0.0000001)

    return contr



path4figsaving = "/home/ivan/PycharmProjects/Vision/results/large_wave/"

params = copy(default_params)
params["outfile"] = path4figsaving + "new_sine_camera.png"  # new_0.5_sigma_
params["sigma_multipl"] = 1.0

image = data.camera()

# wind4csf = 50 # Окно для расчета функции контраста в пикселях

# thresh4signif = 0.1

# Nganlcells = 200*200 #  500 * 500 # Число ганглиозных клеток
# full_area_grad = np.asarray( [55, 60, 90, 60] ) #  градусы верх, низ, наружу, внутрь

Len_y = image.shape[0] # np.sqrt(Nganlcells * ( full_area_grad[0] + full_area_grad[1] ) / (full_area_grad[2] + full_area_grad[3]))
# Len_y = int( np.round(Len_y) )
Len_x = image.shape[1] # int (np.round( Nganlcells / Len_y ))

# x_gr = np.linspace(-full_area_grad[3], full_area_grad[2], Len_x)
# y_gr = np.linspace(-full_area_grad[1], full_area_grad[0], Len_y)

"""
field_size = 10 # in grad
hfs = field_size * 0.5 # in grad
winds = np.array([10, 15, 20, 25])

wind_idx_start_x = Len_x * ( ( winds - hfs + full_area_grad[3] ) / (full_area_grad[2] + full_area_grad[3]) )
wind_idx_ends_x  = Len_x * ( ( winds + hfs + full_area_grad[3] ) / (full_area_grad[2] + full_area_grad[3]) )

wind_idx_start_x = wind_idx_start_x.astype(np.int)
wind_idx_ends_x = wind_idx_ends_x.astype(np.int)

wind_idx_start_y = Len_y * ( ( np.zeros_like(winds) - hfs + full_area_grad[1] ) / (full_area_grad[1] + full_area_grad[0]) )
wind_idx_ends_y  = Len_y * ( ( np.zeros_like(winds) + hfs + full_area_grad[1] ) / (full_area_grad[1] + full_area_grad[0]) )

wind_idx_start_y = wind_idx_start_y.astype(np.int)
wind_idx_ends_y = wind_idx_ends_y.astype(np.int)

# print(wind_idx_ends_x)
# print(Len_x)

"""



x, y = np.meshgrid(np.linspace(-1, 1, Len_x), np.linspace(1, -1, Len_y))
# x = x + 2 * (full_area_grad[2]  / (full_area_grad[2] + full_area_grad[3]) - 0.5)
"""
phase0 = 0 # np.linspace(-np.pi, np.pi, 5)

spfr_coef = 150 * 0.5 # 150 размер поля , 0.5 - потому что x меняется от -1 до 1, т.е. два цикла
spacial_frequensis = np.array([0.004, ])   # full_area_grad # 0.3, 0.6, 0.9, 1.2, 1.5


contast = []
win_contrats = np.zeros( [spacial_frequensis.size, winds.size] , dtype=np.float)
grad_selects = np.zeros( [spacial_frequensis.size, winds.size] , dtype=np.float)
"""







# image = 255 * 0.5 * (np.cos(2 * np.pi * spacial_freqs * spfr_coef * x + phase0) + 1) # 100 * (x + 1) #
    


# res_image, mean_intensity, mean_x, mean_y, abs_steps, angle_steps, xs_AB, ys_AB, cols_AB = lib1.make_preobr(image, x, y, params)
res_image = lib2.make_preobr(image, x, y, params)



# for win_idx in range(winds.size):
#     im_wind = res_image[ wind_idx_start_y[win_idx]:wind_idx_ends_y[win_idx], wind_idx_start_x[win_idx]:wind_idx_ends_x[win_idx] ]
#     win_contrats[freqs_idx, win_idx] = get_contrast(im_wind)
#     grad_selects[freqs_idx, win_idx] = get_gradient_selectivity(im_wind, angle=0)

fig, ax = plt.subplots(ncols=2, nrows=1)
ax[0].pcolor(x[0, :], y[:, 0], image, cmap='gray', vmin=0, vmax=255)
ax[1].pcolor(x[0, :], y[:, 0], res_image, cmap='gray', vmin=0, vmax=255)


"""
for wind in winds:
    rect =  pchs.Rectangle( [wind - hfs,  -hfs], field_size, field_size, linewidth=1, edgecolor='r', facecolor='none')
    ax[0].add_patch(rect)
"""

"""
abs_pix = np.sqrt((x**2 + y**2))
angle_pix = np.arctan2(y, x)

abs_steps = np.geomspace(0.01*(x.max() - x.min()), np.max(abs_pix)+0.0001, 30)  # np.linspace(np.min(abs_pix), np.max(abs_pix)+0.0001, 10) #
angle_steps = np.linspace(-np.pi, np.pi+0.0001, 30)
xxx, yyy = lib1.get_rete(abs_steps, angle_steps, Len_x, Len_y, onesformat=True)

for xx, yy in zip(xxx, yyy):
    # x, y = lib.pix2rel(x, y, Len_x, Len_y)
    # xx = (xx + 1) * 0.5 * (x_gr[-1] - x_gr[0]) + x_gr[0] - 15
    # yy = (yy + 1) * 0.5 * (y_gr[-1] - y_gr[0]) + y_gr[0]

    ax[1].plot(xx, yy, color="b", linewidth=0.1)
    ax[1].set_xlim(x_gr[0], x_gr[-1])
    ax[1].set_ylim(y_gr[0], y_gr[-1])
"""


fig.savefig(params["outfile"])

plt.show()
plt.close("all")



"""
# contast = np.asarray(contast)
# x = np.linspace(0, 1, center_line.size)



fig, ax = plt.subplots(nrows=1, ncols=2)

# sp_fr_contrast = []
# sp_fr_gradratio = []
for sp_fr_idx in range(spacial_frequensis.size):

    # cntr = win_contrats[:, win_idx]
    # cntr_idx = np.min( np.argwhere(cntr > thresh4signif) )
    # sp_fr_contrast.append(spacial_frequensis[cntr_idx])
    #
    # grsel = grad_selects[:, win_idx]
    # grsel_idx = np.min( np.argwhere(grsel > thresh4signif) )
    # sp_fr_gradratio.append(spacial_frequensis[grsel_idx])

    ax[0].plot(winds, win_contrats[sp_fr_idx, :], label=spacial_frequensis[sp_fr_idx] )  # , spacial_frequensis
    ax[1].plot(winds, grad_selects[sp_fr_idx, :], label=spacial_frequensis[sp_fr_idx] ) # , spacial_frequensis)

ax[0].set_title("Michelson contrasts")
ax[1].set_title("Gradient selectivity")


for axes in ax:
    axes.legend()
    axes.set_ylim(0, 1)
    axes.set_xlabel("Degrees")

# ax[0].bar(winds, sp_fr_contrast)
# ax[0].bar(winds, sp_fr_gradratio)

fig.savefig(path4figsaving + "contrast.png")
plt.show()

"""
