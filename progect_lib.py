import numpy as np
from skimage import data # , filters
from skimage import measure
import matplotlib.pyplot as plt




# image = data.camera().astype(np.float)
# center_x = 0 # -(Len_x - 240) * 2 / Len_x + 1
# center_y = 0 # (Len_y - 125) * 2 / Len_y - 1

def rel2pix(x, y, Len_x, Len_y):
    x_255 = (x + 1) * 0.5 * Len_x
    y_255 = (-y + 1) * 0.5 * Len_y
    return x_255, y_255


def pix2rel(x_255, y_255, Len_x, Len_y):
    x = x_255 / Len_x * 2 - 1
    y = 1 - y_255 / Len_y * 2
    return x, y

def make_preobr(image, center_x, center_y, x, y):

    delta_x = x[0, 1] - x[0, 0]
    delta_y = y[1, 0] - y[0, 0]

    x = x - center_x
    y = y - center_y



    abs_pix = np.sqrt((x**2 + y**2))
    angle_pix = np.arctan2(y, x)



    grad_y, grad_x = np.gradient(image, delta_y, delta_x)

    # print(grad_y)
    # print("###########################")
    # print(grad_x)



    abs_steps = np.geomspace(np.min(abs_pix), np.max(abs_pix), 5)  #  np.linspace(np.min(abs_pix), np.max(abs_pix)+0.0001, 30)
    angle_steps = np.linspace(-np.pi, np.pi, 6)
    # angle_steps[2] = -np.pi / 2
    # mask = np.zeros_like(image, dtype=float)

    res_image = np.zeros_like(image, dtype=float)
    mean_intensity = np.zeros_like(image, dtype=float)
    onlygrad = np.zeros_like(image, dtype=float)


    # fi_mins = []
    # fi_maxs = []
    # r_mins = []
    # r_maxs = []
    mean_intensity_list = []

    mean_xs = np.empty((0), dtype=np.float)
    mean_ys = np.empty_like(mean_xs)


    for idx1, ab_step in enumerate(abs_steps[:-1]):
        for idx2, an_step in enumerate(angle_steps[:-1]):
            chosen_pix = (abs_pix >= ab_step) & (abs_pix < abs_steps[idx1+1]) & (angle_pix >= an_step) & (angle_pix < angle_steps[idx2+1])
            if np.sum(chosen_pix) == 0:
                continue

            mean_xs = np.append(mean_xs, np.mean(x[chosen_pix]))
            mean_ys = np.append(mean_ys, np.mean(y[chosen_pix]))

            mean_intensity[chosen_pix] = np.mean(image[chosen_pix])
            # fi_mins.append(an_step)
            # fi_maxs.append(angle_steps[idx2+1])
            # r_mins.append(ab_step)
            # r_maxs.append(abs_steps[idx1+1])
            mean_intensity_list.append(np.mean(image[chosen_pix]))



    # fi_mins = np.asarray(fi_mins)
    # fi_maxs = np.asarray(fi_maxs)
    # r_mins = np.asarray(r_mins)
    # r_maxs = np.asarray(r_maxs)
    mean_intensity_list = np.asarray(mean_intensity_list)
    xs_AB, ys_AB = [], []


    for idx1, ab_step in enumerate(abs_steps[:-1]):
        for idx2, an_step in enumerate(angle_steps[:-1]):

            chosen_pix = (abs_pix >= ab_step) & (abs_pix < abs_steps[idx1+1]) & (angle_pix >= an_step) & (angle_pix < angle_steps[idx2+1])



            if np.sum(chosen_pix) == 0:
                continue
            elif np.sum(chosen_pix) < 2:
                # mean_grad_x = 0
                # mean_grad_y = 0
                # grad_angle = None
                mean_intens = np.mean(image[chosen_pix])
                res_image[chosen_pix] = mean_intens
                continue
            else:
                mean_grad_x = np.mean(grad_x[chosen_pix])
                mean_grad_y = np.mean(grad_y[chosen_pix])
                # grad_angle = np.arctan2( mean_grad_y, mean_grad_x ) # np.rad2deg( )
                mean_intens = np.mean(image[chosen_pix])



            mean_x = np.mean(x[chosen_pix])
            mean_y = np.mean(y[chosen_pix])

            #            get_distance2AB(fi_min, fi_max, abs_min, abs_max, grad_x, grad_y, x_mean, y_mean)
            x_AB, y_AB = get_distance2AB(an_step, angle_steps[idx2+1], ab_step, abs_steps[idx1+1], mean_grad_x, mean_grad_y, mean_x, mean_y)
            xs_AB.append(x_AB)
            ys_AB.append(y_AB)


            res_image[chosen_pix] = mean_intens + mean_grad_x * (x[chosen_pix] - mean_x) + mean_grad_y * (y[chosen_pix] - mean_y)
            onlygrad[chosen_pix] = mean_grad_x * (x[chosen_pix] - mean_x) + mean_grad_y * (y[chosen_pix] - mean_y)




    res_image[res_image < 0] = 0
    res_image[res_image > 255] = 255

    return res_image, mean_intensity, onlygrad, mean_xs, mean_ys, abs_steps, angle_steps, xs_AB, ys_AB



def make_succades(image, centers_x, centers_y, x, y, weghts_sigm):
    Len_y, Len_x = image.shape

    quality = {}
    quality["rmse"] = []
    quality["psnr"] = []
    quality["ssim"] = []

    res_images = np.empty((Len_y, Len_x, len(centers_x)), dtype=np.float)
    mean_intensitys = np.empty((Len_y, Len_x, len(centers_x)), dtype=np.float)

    weghts = np.empty((Len_y, Len_x, len(centers_x)), dtype=np.float)

    for idx, cent_x, cent_y in zip(range(len(centers_x)), centers_x, centers_y):
        res_image_tmp, mean_intensity_tmp, onlygrad = make_preobr(image, cent_x, cent_y, x, y)

        dists = np.sqrt((x - cent_x) ** 2 + (y - cent_y) ** 2)
        # dists = dists / np.sum(dists) # !!!!!!!

        weghts[:, :, idx] = np.exp(-dists / weghts_sigm)

        res_images[:, :, idx] = res_image_tmp
        mean_intensitys[:, :, idx] = mean_intensity_tmp

    norm_coeffs = np.sum(weghts, axis=2)
    for idx in range(len(centers_x)):
        weghts[:, :, idx] = weghts[:, :, idx] / norm_coeffs

    res_image = np.zeros((Len_y, Len_x), dtype=np.float)
    mean_intensity = np.zeros((Len_y, Len_x), dtype=np.float)

    for idx in range(len(centers_x)):
        res_image += weghts[:, :, idx] * res_images[:, :, idx]
        mean_intensity += weghts[:, :, idx] * mean_intensitys[:, :, idx]

    return res_image, mean_intensity

def get_rete(abs_steps, angle_steps, Len_x, Len_y):
    xx = []
    yy = []

    for ab in abs_steps:
        x = np.linspace(-ab, ab, 1000)
        y = np.sqrt(ab ** 2 - x ** 2)
        xx.append(x)
        xx.append(x)
        yy.append(y)
        yy.append(-y)

    for an in angle_steps:
        x = 0
        # print(an)
        if np.abs(an) == np.pi:
            x = np.array([-1, 0], dtype=float)
            y = np.zeros_like(x)
        elif an == 0:
            x = np.array([0, 1], dtype=float)
            y = np.zeros_like(x)
        elif an == np.pi / 2:
            x = np.array([0, 0], dtype=float)
            y = np.linspace(0, 1, 2)
        elif an == -np.pi / 2:
            x = np.array([0, 0], dtype=float)
            y = np.linspace(-1, 0, 2)
        elif np.abs(an) < np.pi/2:
            x = np.array([0, 1], dtype=float)
            k = np.tan(an)
            y = k * x
        elif np.abs(an) > np.pi/2:
            x = np.array([-1, 0], dtype=float)
            k = np.tan(an)
            y = k * x
        if not (type(x) is int):
            xx.append(x)
            yy.append(y)

    for idx, (x, y) in enumerate(zip(xx, yy)):
        x_255, y_255 = rel2pix(x, y, Len_x, Len_y)
        xx[idx] = x_255
        yy[idx] = y_255

    return xx, yy


def get_distance2AB(fi_min, fi_max, abs_min, abs_max, grad_x, grad_y, x_mean, y_mean):



    if grad_y == 0:
        x = np.array([x_mean, x_mean])
        y = np.sqrt( np.array([abs_min, abs_max])**2  - x_mean)

        x = np.append(x, x)
        y = np.append(y, -y)

        x = np.append(x, [x_mean, x_mean])
        y = np.append(y, np.tan([fi_min, fi_max]))

    else:
        tan_fi_grad = grad_x / grad_y
        alpha = -tan_fi_grad*x_mean + y_mean

        x = np.empty(0, dtype=np.float)

        # Пересечение с лучами
        x_tmp = alpha / (np.tan([fi_min, fi_max]) - tan_fi_grad)
        x = np.append(x, x_tmp)

        # пересечение с окружностями
        a = 1 + tan_fi_grad ** 2
        b = 2 * tan_fi_grad * alpha
        c = alpha**2 - np.array([abs_min, abs_max])**2
        D = b**2 - 4 * a * c

        non_negativeD = D >= 0
        D = D[non_negativeD]

        x_tmp = (-b + np.sqrt(D)) / (2 * a)
        x = np.append(x, x_tmp)

        x_tmp = (-b - np.sqrt(D)) / (2 * a)
        x = np.append(x, x_tmp)

        y = tan_fi_grad * x + alpha

    x = x.ravel()

    # angl = np.arctan(tan_fi_grad)
    # angl = 2*np.pi - angl
    # xn = x * np.cos(angl) + y * np.sin(angl)
    # yn = y * np.cos(angl) - x * np.sin(angl)
    # print(xn)
    # print(yn)

    dists_arg = np.argsort( (x - x_mean)**2 + (y - y_mean)**2 )

    x_res = x[dists_arg[0:2]]
    y_res = y[dists_arg[0:2]]

    AB_dist = np.sqrt( (x_res[0] - x_res[1])**2 + (y_res[0] - y_res[1])**2 )
    mean_A_dist = np.sqrt( (x_res[0] - x_mean)**2 + (y_res[0] - y_mean)**2 )
    mean_B_dist = np.sqrt( (x_res[1] - x_mean)**2 + (y_res[1] - y_mean)**2 )

    if np.abs( mean_A_dist + mean_B_dist - AB_dist) >= 0.001:
        x_res = x[dists_arg[ np.array([0, 2], dtype=int) ]]
        y_res = y[dists_arg[ np.array([0, 2], dtype=int) ]]
        print("Wrong distance!")





    return x_res, y_res

