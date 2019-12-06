import numpy as np
from skimage import data # , filters
from skimage import measure
import matplotlib.pyplot as plt
# from scipy.integrate import cumtrapz
# from scipy.interpolate import griddata
# from scipy.stats import circmean



image = data.camera().astype(np.float)

Len_y, Len_x = image.shape

image = np.zeros_like(image) + 100
image[:, :Len_x//2] = 200


center_x = 0 # -(Len_x - 240) * 2 / Len_x + 1
center_y = 0 # (Len_y - 125) * 2 / Len_y - 1
# Len_y, Len_x = (5, 5)

x, y = np.meshgrid(np.linspace(-1, 1, Len_x), np.linspace(1, -1, Len_y))
delta_x = x[0, 1] - x[0, 0]
delta_y = y[1, 0] - y[0, 0]

x = x - center_x
y = y - center_y



abs_pix = np.sqrt((x**2 + y**2))
angle_pix = np.arctan2(y, x)


# grad_x = filters.sobel_v(image) / delta_x
# grad_y = filters.sobel_h(image) / delta_y

grad_y, grad_x = np.gradient(image, delta_y, delta_x) #  !!!!!

# print(grad_y)
# print("###########################")
# print(grad_x)



abs_steps = np.geomspace(np.min(abs_pix), np.max(abs_pix)+0.0001, 30)  #  np.linspace(0, np.sqrt(2)+0.0001, 30)
angle_steps = np.linspace(-np.pi, np.pi+0.0001, 30)

mask = np.zeros_like(image, dtype=float)

res_image = np.zeros_like(image, dtype=float)
res_image2 = np.zeros_like(image, dtype=float)
res_image3 = np.zeros_like(image, dtype=float)
# avg_gradx = []
# avg_grady = []
# abs_grad = []
# ang_grad = []

for idx1, ab_step in enumerate(abs_steps[:-1]):
    for idx2, an_step in enumerate(angle_steps[:-1]):
        chosen_pix = (abs_pix >= ab_step) & (abs_pix < abs_steps[idx1+1]) & (angle_pix >= an_step) & (angle_pix < angle_steps[idx2+1])


        if np.sum(chosen_pix) == 0:
            continue
        elif np.sum(chosen_pix) < 2:
            mean_grad_x = 0
            mean_grad_y = 0
            mean_intens = np.mean(image[chosen_pix])
        else:
            mean_grad_x = np.mean(grad_x[chosen_pix])
            mean_grad_y = np.mean(grad_y[chosen_pix])
            mean_intens = np.mean(image[chosen_pix])

        mean_x = np.mean(x[chosen_pix])
        mean_y = np.mean(y[chosen_pix])

        # mask[chosen_pix] = np.sqrt(mean_grad_x**2 + mean_grad_y**2)      # np.random.randint(0, 255, 1)
        # print(mean_grad_x * np.mean(x[chosen_pix]), mean_grad_y * np.mean(y[chosen_pix]), mean_intens)
        # distances = np.sqrt((x[chosen_pix] - mean_x)**2 + (y[chosen_pix] - mean_y)**2)
        #
        # # print(distances)
        #
        # cos_alpha = (x[chosen_pix] - mean_x) / distances
        # cos_beta = (y[chosen_pix] - mean_y) / distances
        #
        # grad_l = mean_grad_x*cos_alpha + mean_grad_y*cos_beta    # np.sqrt( (mean_grad_x)**2 + (mean_grad_y)**2 ) # * (x[chosen_pix] / mean_x) #  * (y[chosen_pix] / mean_y)
        # delta_l = np.sqrt(1 - cos_alpha**2) / cos_alpha  * (x[chosen_pix] - mean_x)
        #
        #
        # res_image[chosen_pix] = mean_intens + grad_l*delta_l # +mean_grad_x * (x[chosen_pix] / mean_x) + mean_grad_y * (y[chosen_pix] / mean_y)

        res_image[chosen_pix] = mean_intens + mean_grad_x * (x[chosen_pix] - mean_x) + mean_grad_y * (y[chosen_pix] - mean_y)

        res_image2[chosen_pix] = mean_intens
        res_image3[chosen_pix] = mean_grad_x * (x[chosen_pix] - mean_x) + mean_grad_y * (y[chosen_pix] - mean_y)

        # print( grad_l*delta_l)
        # print("#########################")

        # avg_gradx.append(np.mean(grad_x[chosen_pix]))
        # avg_grady.append(np.mean(grad_y[chosen_pix]))
        # abs_grad.append( ab_step ) # (ab_step+abs_steps[idx1+1])/2
        # ang_grad.append( an_step )  # circmean(an_step+angle_steps[idx2+1])


"""
avg_gradx = np.asarray(avg_gradx)
avg_grady = np.asarray(avg_grady)

abs_grad = np.asarray(abs_grad)
ang_grad = np.asarray(ang_grad)

r_grad = np.sqrt(avg_gradx**2 + avg_grady**2)
fi_grad = np.arctan2(avg_grady, avg_gradx)


polar_image = np.empty(0, dtype=np.float)
polar_image_r = np.empty(0, dtype=np.float)
polar_image_fi = np.empty(0, dtype=np.float)


for an in np.unique(ang_grad):

    chosen_r = ang_grad == an

    gr = r_grad[chosen_r] #* np.cos(ang_grad[chosen_r] - an)

    int_r = cumtrapz(gr, initial=50)

    # print(abs_grad[chosen_r])

    polar_image = np.append(polar_image, int_r )
    polar_image_r = np.append(polar_image_r, abs_grad[chosen_r])
    polar_image_fi = np.append(polar_image_fi, np.zeros_like(int_r)+an  )

newx_pix = polar_image_r * np.cos(polar_image_fi)
newy_pix = polar_image_r * np.sin(polar_image_fi)

res_image = griddata( (newx_pix, newy_pix), polar_image, (x.ravel(), y.ravel()), method='nearest') #nearest cubic linear


res_image = res_image.reshape(Len_y, Len_x)
"""

# res_image = x * grad_x + y * grad_y + np.tile(image[:, Len_x//2],(Len_x,1)) + np.tile( image[Len_y//2, :].reshape(-1, 1), (1, Len_y) )

# res_image = (res_image - np.mean(res_image) ) / np.linalg.norm(res_image)


res_image[res_image < 0] = 0
res_image[res_image > 255] = 255





label = 'NRMSE = {:.4f}, PSNR = {:.4f}, SSIM = {:.4f}'

fig, ax = plt.subplots(nrows=2, ncols=2 , figsize=(15, 15) )
ax[0, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
ax[0, 0].set_title("Исходное изображение")
ax[0, 0].scatter(240, 125, color="red", s=0.5)

ax[0, 1].imshow(res_image2, cmap="gray", vmin=0, vmax=255)
ax[0, 1].set_title("Средняя интенсивность")
ax[0, 1].scatter(240, 125, color="red", s=0.5)

nrmse = measure.compare_nrmse(im_true=image, im_test=res_image2, norm_type='Euclidean')
psnr = measure.compare_psnr(im_true=image, im_test=res_image2, data_range=255)
ssim = measure.compare_ssim(image, res_image2, data_range=255)
ax[0, 1].set_xlabel(label.format(nrmse, psnr, ssim))

ax[1, 0].imshow(res_image3, cmap="gray", vmin=0, vmax=255)
ax[1, 0].set_title("Градиент")
ax[1, 0].scatter(240, 125, color="red", s=0.5)

ax[1, 1].imshow(res_image, cmap="gray", vmin=0, vmax=255)
ax[1, 1].set_title("Восстановленное изображение")
ax[1, 1].scatter(240, 125, color="red", s=0.5)

nrmse = measure.compare_nrmse(im_true=image, im_test=res_image, norm_type='Euclidean')
psnr = measure.compare_psnr(im_true=image, im_test=res_image, data_range=255)
ssim = measure.compare_ssim(image, res_image, data_range=255)
ax[1, 1].set_xlabel(label.format(nrmse, psnr, ssim))

fig.savefig("/home/ivan/res.png", dpi=500)
plt.show()



# fig = plt.figure()
# plt.imshow(mask, cmap="rainbow")
# plt.show()

