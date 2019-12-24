import numpy as np
from skimage import data # , filters
from skimage import measure
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import progect_lib as lib

# image = data.astronaut().astype(np.float)  # image = data.astronaut().astype(np.float)  #
# image = rgb2gray(image)
# image = image[:250, 100:350] # берем только лицо

image = np.zeros( (201, 201), dtype=float ) # data.camera().astype(np.float)   #


Len_y, Len_x = image.shape
x, y = np.meshgrid(np.linspace(-1, 1, Len_x), np.linspace(1, -1, Len_y))
weghts_sigm = 0.5

image = -100 * x + 100 * y + 100

centers_x = np.array( [Len_x/2, 126, 103, 148, 102, 144] ) # + 100 # / Len_x * 2 - 1 # np.array( [-0.12, -0.2] ) # np.array( [-0.5, 0.5] ) #
centers_y = np.array( [Len_y/2, 127, 100, 103, 140, 139] ) # / Len_y * 2 # np.array( [0.5, -0.5] )  # np.array( [0.5, 0.6] )   #


centers_x, centers_y = lib.pix2rel(centers_x, centers_y, Len_y, Len_x)

centers_x = centers_x[0] # !!!!!
centers_y = centers_y[0] # !!!!!
# image = np.zeros_like(image) + 100
# image[:, :Len_x//2] = 200
res_image, mean_intensity, onlygrad, mean_x, mean_y, abs_steps, angle_steps, xs_AB, ys_AB, cols_AB, res_xys = lib.make_preobr(image, centers_x, centers_y, x, y)

# res_image, mean_intensity = lib.make_succades(image, centers_x, centers_y, x, y, weghts_sigm)


center_x_255, center_y_255 = lib.rel2pix(centers_x, centers_y, Len_x, Len_y)
mean_x_255, mean_y_255 = lib.rel2pix(mean_x, mean_y, Len_x, Len_y)

label = 'NRMSE = {:.4f}, PSNR = {:.4f}, SSIM = {:.4f}'

fig, ax = plt.subplots(nrows=2, ncols=2 , figsize=(15, 15) )

fig2, ax2 = plt.subplots(nrows=1, ncols=1 , figsize=(15, 15) )

ax[0, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
ax[0, 0].set_title("Исходное изображение")
ax[0, 0].scatter(center_x_255, center_y_255, color="red", s=0.5)

ax[0, 1].imshow(mean_intensity, cmap="gray", vmin=0, vmax=255)
ax[0, 1].set_title("Средняя интенсивность")
ax[0, 1].scatter(center_x_255, center_y_255, color="red", s=0.5)

nrmse = measure.compare_nrmse(im_true=image, im_test=mean_intensity, norm_type='Euclidean')
psnr = measure.compare_psnr(im_true=image, im_test=mean_intensity, data_range=255)
ssim = measure.compare_ssim(image, mean_intensity, data_range=255)
ax[0, 1].set_xlabel(label.format(nrmse, psnr, ssim))

ax[1, 0].imshow(onlygrad, cmap="gray", vmin=0, vmax=255)
ax[1, 0].set_title("Градиент")
ax[1, 0].scatter(center_x_255, center_y_255, color="red", s=0.5)
ax[1, 0].scatter(mean_x_255, mean_y_255, color="g", s=5.5)

ax[1, 1].imshow(res_image, cmap="gray", vmin=0, vmax=255)
ax2.imshow(res_image, cmap="gray", vmin=0, vmax=255)
ax[1, 1].set_title("Восстановленное изображение")
ax[1, 1].scatter(center_x_255, center_y_255, color="red", s=0.5)
# ax2.scatter(center_x_255, center_y_255, color="rs", s=0.5)
ax[1, 1].scatter(mean_x_255, mean_y_255, color="r", s=6.5)
ax2.scatter(mean_x_255, mean_y_255, color="r", s=6.5)

xx, yy = lib.get_rete(abs_steps, angle_steps, Len_x, Len_y)

for x, y in zip(xx, yy):
    ax[1, 1].plot(x, y, color="b")
    ax2.plot(x, y, color="b")

# for x, y in zip(mean_x_255, mean_y_255):
#     ax[1, 1].arrow(x, y, 10, 10, width=0.1, color="c", head_width=1)


for x_AB, y_AB in zip(xs_AB, ys_AB):
    x_AB, y_AB = lib.rel2pix(x_AB, y_AB, Len_x, Len_y)
    ax[1, 1].plot(x_AB, y_AB, color="c")
    ax2.plot(x_AB, y_AB, color="c")

ax[1, 1].set_xlim(0, Len_x)
ax2.set_xlim(0, Len_x)
ax[1, 1].set_ylim(Len_y, 0)
ax2.set_ylim(Len_y, 0)

for res_xy in res_xys:
    res_x, res_y = lib.rel2pix(res_xy[0], res_xy[1], Len_x, Len_y)
    ax[1, 1].scatter(res_x, res_y, color="red", s=50)
    ax2.scatter(res_x, res_y, color="c", s=50)



for cols_x_A, cols_y_A, cols_x_B, cols_y_B, mean_xs, mean_ys in zip(cols_AB[0], cols_AB[1],cols_AB[2], cols_AB[3], cols_AB[4], cols_AB[5] ):
        col_x_A = cols_x_A
        col_y_A = cols_y_A
        col_x_B = cols_x_B
        col_y_B = cols_y_B
        mean_x = mean_xs
        mean_y = mean_ys
        mean_x, mean_y = lib.rel2pix(mean_x, mean_y, Len_x, Len_y)

        if col_x_A != None and col_y_A != None:
            x_A, y_A = lib.rel2pix(col_x_A, col_y_A, Len_x, Len_y)
            # ax[1, 1].arrow(x_A, y_A, mean_x - x_A, mean_y - y_A, width=0.1, color="r", head_width=1)
            ax2.arrow(x_A, y_A, mean_x - x_A, mean_y - y_A, width=0.1, color="r", head_width=1)

        if col_x_B != None and col_y_B != None:
            x_B, y_B = lib.rel2pix(col_x_B, col_y_B, Len_x, Len_y)
            # ax[1, 1].arrow(x_B, y_B, mean_x-x_B, mean_y-y_B, width=0.1, color="r", head_width=1)
            ax2.arrow(x_B, y_B, mean_x-x_B, mean_y-y_B, width=0.1, color="r", head_width=1)


nrmse = measure.compare_nrmse(im_true=image, im_test=res_image, norm_type='Euclidean')
psnr = measure.compare_psnr(im_true=image, im_test=res_image, data_range=255)
ssim = measure.compare_ssim(image, res_image, data_range=255)
ax[1, 1].set_xlabel(label.format(nrmse, psnr, ssim))

# fig.savefig("./results/astronaut_only_face.png", dpi=500)
plt.show()








