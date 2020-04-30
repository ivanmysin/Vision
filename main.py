import numpy as np
from skimage import data # , filters
from skimage import metrics
from skimage.filters import gaussian
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import progect_lib as lib
from copy import copy

# image = data.astronaut().astype(np.float)  # image = data.astronaut().astype(np.float)  #
# image = rgb2gray(image)
# image = image[:250, 100:350] # берем только лицо

default_params = {
    "shift_xy_mean" : True,  # Сдвигать или нет центр гиперколонки
    "avg_grad_over" : True,  # Усреднять или нет градиент вдоль перпеникуляров к среднему направлению градиента
    "apply_AB_thresh" : True, # Применять или нет обрезание по средним значениях в граничных гиперколонках
    "apply_smooth_grad_dist" : True, # Уменьшать ли вес градиента при удалении от центра гиперколонки
    "apply_grad_in_extrems" : False, # Применять ли градиент, если среднее значение является меньше или больше обоих соседних
    "smooth_bounds" : True, # не реализовано !!!   Применять или нет сглаживание на границах рецептивных полей

}

params = copy(default_params)
params["shift_xy_mean"] = True
params["apply_AB_thresh"] = True
params["apply_smooth_grad_dist"] = True
params["avg_grad_over"] = True
params["apply_grad_in_extrems"] = False

params["outfile"] = "experiment 1"


image = 255 * rgb2gray(data.astronaut()).astype(np.float) # data.camera().astype(np.float) #  / 255   # np.zeros( (201, 201), dtype=float ) #


Len_y, Len_x = image.shape
x, y = np.meshgrid(np.linspace(-1, 1, Len_x), np.linspace(1, -1, Len_y))
# y = np.flip(y, axis=0)



# image = -200 * x**2 + 100 + 10 * y**2 # + 100 #

centers_x_255 = np.array( [220, Len_x/2, 126, 103, 148, 102, 144] ) # + 100 # / Len_x * 2 - 1 # np.array( [-0.12, -0.2] ) # np.array( [-0.5, 0.5] ) #
centers_y_255 = np.array( [120, Len_y/2, 127, 100, 103, 140, 139] ) # / Len_y * 2 # np.array( [0.5, -0.5] )  # np.array( [0.5, 0.6] )   #



centers_x, centers_y = lib.pix2rel(centers_x_255, centers_y_255, Len_y, Len_x)

centers_x = centers_x[0] # !!!!!
centers_y = centers_y[0] # !!!!!

x = x - centers_x
y = y - centers_y


x_lim_min = np.min(x)
x_lim_max = np.max(x)

y_lim_min = np.min(y)
y_lim_max = np.max(y)

# image = np.zeros_like(image) + 100
# image[:, :Len_x//2] = 200


res_image, mean_intensity, mean_x, mean_y, abs_steps, angle_steps, xs_AB, ys_AB, cols_AB = lib.make_preobr(image, centers_x, centers_y, x, y, params)



# if use_not_only_grad:
#     res_image,  _, _, _, _, abs_steps, angle_steps, xs_AB, ys_AB, cols_AB = lib.make_preobr(res_image, centers_x, centers_y, x, y, use_not_only_grad, onlygrad)

# res_image, mean_intensity = lib.make_succades(image, centers_x, centers_y, x, y, weghts_sigm)


center_x_255, center_y_255 = lib.rel2pix(centers_x, centers_y, Len_x, Len_y)
mean_x_255, mean_y_255 = lib.rel2pix(mean_x, mean_y, Len_x, Len_y)

label = 'NRMSE = {:.4f}, PSNR = {:.4f}, SSIM = {:.4f}'

fig, ax = plt.subplots(nrows=2, ncols=2 , figsize=(15, 15) )

fig2, ax2 = plt.subplots(nrows=1, ncols=1 , figsize=(15, 15) )

ax[0, 0].pcolor(x[0, :], y[:, 0], image, cmap='gray', vmin=0, vmax=255) # imshow(image, cmap="gray", vmin=0, vmax=255)
ax[0, 0].set_title("Исходное изображение")
ax[0, 0].scatter(centers_x, centers_y, color="red", s=0.5) #scatter(center_x_255, center_y_255, color="red", s=0.5)

ax[0, 1].pcolor(x[0, :], y[:, 0], mean_intensity, cmap='gray', vmin=0, vmax=255) # imshow(mean_intensity, cmap="gray", vmin=0, vmax=255)
ax[0, 1].set_title("Средняя интенсивность")
ax[0, 1].scatter(centers_x, centers_y, color="red", s=0.5) # scatter(center_x_255, center_y_255, color="red", s=0.5)

nrmse = metrics.normalized_root_mse(image_true=image, image_test=mean_intensity)
psnr = metrics.peak_signal_noise_ratio(image_true=image, image_test=mean_intensity, data_range=255)
ssim = metrics.structural_similarity(image, mean_intensity, data_range=255)
ax[0, 1].set_xlabel(label.format(nrmse, psnr, ssim))

# ax[1, 0].imshow(onlygrad, cmap="gray", vmin=0, vmax=255)
ax[1, 0].set_title("Сетка")
# ax[1, 0].scatter(center_x_255, center_y_255, color="red", s=0.5)
# ax[1, 0].scatter(mean_x_255, mean_y_255, color="g", s=5.5)

ax[1, 1].pcolor(x[0, :], y[:, 0], res_image, cmap='gray', vmin=0, vmax=255) # imshow(res_image, cmap="gray", vmin=0, vmax=255)

ax2.pcolor(x[0, :], y[:, 0], res_image, cmap='gray', vmin=0, vmax=255)

ax[1, 1].set_title("Восстановленное изображение")
# ax[1, 1].scatter(center_x_255, center_y_255, color="red", s=0.5)
# ax2.scatter(center_x_255, center_y_255, color="r", s=0.5)
ax[1, 0].scatter(mean_x, mean_y, color="r", s=6.5) # scatter(mean_x_255, mean_y_255, color="r", s=6.5)
ax2.scatter(mean_x, mean_y, color="r", s=5.5)


# grad_x_acc = -0.02 * mean_x
# grad_y_acc = 0.02 * mean_y / 20
#
#
# for x_tmp, y_tmp, dx, dy in zip(mean_x, mean_y, grad_x_acc, grad_y_acc):
#     ax2.arrow(x_tmp, y_tmp, dx, dy, width = 0.01, head_length = 0.01, color="red")

# centers_x_1, centers_y_1 = lib.rel2pix( centers_x, centers_y, Len_x, Len_y )
xx, yy = lib.get_rete(abs_steps, angle_steps, Len_x, Len_y, onesformat=True)

for x, y in zip(xx, yy):
    #x, y = lib.pix2rel(x, y, Len_x, Len_y)

    ax[1, 0].plot(x, y, color="b")
    ax[0, 0].plot(x, y, color="b")
    ax2.plot(x, y, color="b")



for x_AB, y_AB in zip(xs_AB, ys_AB):
    ax2.plot(x_AB, y_AB, color="c")
    # x_AB, y_AB = lib.rel2pix(x_AB, y_AB, Len_x, Len_y)
    ax[1, 0].plot(x_AB, y_AB, color="c")


ax[1, 0].set_xlim(x_lim_min, x_lim_max)
ax[1, 0].set_ylim(y_lim_min, y_lim_max)

ax[0, 1].set_xlim(x_lim_min, x_lim_max)
ax[0, 1].set_ylim(y_lim_min, y_lim_max)

ax[0, 0].set_xlim(x_lim_min, x_lim_max)   # (0, Len_x)
ax2.set_xlim(x_lim_min, x_lim_max)
ax[0, 0].set_ylim(y_lim_min, y_lim_max)  # (Len_y, 0)
ax2.set_ylim(y_lim_min, y_lim_max)






nrmse = metrics.normalized_root_mse(image_true=image, image_test=res_image)
psnr = metrics.peak_signal_noise_ratio(image_true=image, image_test=res_image, data_range=255)
ssim = metrics.structural_similarity(image, res_image, data_range=255)
ax[1, 1].set_xlabel(label.format(nrmse, psnr, ssim))

resultline = params["outfile"] + " " + label.format(nrmse, psnr, ssim) + "\n"


print (resultline)

file = open("./results/metrics.txt", "a")
file.write(resultline)
file.close()

fig_saving_path = "./results/" # "./results/old_algorhythm/"
fig.savefig(fig_saving_path + params["outfile"] + ".png", dpi=200)
fig2.savefig(fig_saving_path + params["outfile"] + "_large.png", dpi=200)
# plt.show()




# for cols_x_A, cols_y_A, cols_x_B, cols_y_B, mean_xs, mean_ys in zip(cols_AB[0], cols_AB[1],cols_AB[2], cols_AB[3], cols_AB[4], cols_AB[5] ):
#         col_x_A = cols_x_A
#         col_y_A = cols_y_A
#         col_x_B = cols_x_B
#         col_y_B = cols_y_B
#
#         mean_x_ = mean_xs
#         mean_y_ = mean_ys
#
#
#         mean_x, mean_y = lib.rel2pix(mean_x_, mean_y_, Len_x, Len_y)
#
#         if col_x_A != None and col_y_A != None:
#             x_A, y_A = lib.rel2pix(col_x_A, col_y_A, Len_x, Len_y)
#             # ax[1, 1].arrow(x_A, y_A, mean_x - x_A, mean_y - y_A, width=0.1, color="r", head_width=1)
#             ax2.arrow(x_A, y_A, mean_x - x_A, mean_y - y_A, width=0.1, color="r", head_width=1)
#
#         if col_x_B != None and col_y_B != None:
#             x_B, y_B = lib.rel2pix(col_x_B, col_y_B, Len_x, Len_y)
#             # ax[1, 1].arrow(x_B, y_B, mean_x-x_B, mean_y-y_B, width=0.1, color="r", head_width=1)
#             ax2.arrow(x_B, y_B, mean_x-x_B, mean_y-y_B, width=0.1, color="r", head_width=1)

