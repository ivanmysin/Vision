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



    abs_steps = np.geomspace(np.min(abs_pix), np.max(abs_pix)+0.0001, 5)  #  np.linspace(np.min(abs_pix), np.max(abs_pix)+0.0001, 30)
    angle_steps = np.linspace(-np.pi, np.pi, 6)
    # angle_steps[2] = -np.pi / 2
    # mask = np.zeros_like(image, dtype=float)

    res_image = np.zeros_like(image, dtype=float)
    mean_intensity = np.zeros_like(image, dtype=float)
    onlygrad = np.zeros_like(image, dtype=float)

    mean_xs, mean_ys = [], []

    fi_mins = []
    fi_maxs = []
    r_mins = []
    r_maxs = []
    mean_intensity_list = []

    for idx1, ab_step in enumerate(abs_steps[:-1]):
        for idx2, an_step in enumerate(angle_steps[:-1]):
            chosen_pix = (abs_pix >= ab_step) & (abs_pix < abs_steps[idx1+1]) & (angle_pix >= an_step) & (angle_pix < angle_steps[idx2+1])
            if np.sum(chosen_pix) == 0:
                continue
            mean_intensity[chosen_pix] = np.mean(image[chosen_pix])
            fi_mins.append(an_step)
            fi_maxs.append(angle_steps[idx2+1])
            r_mins.append(ab_step)
            r_maxs.append(abs_steps[idx1+1])
            mean_intensity_list.append(np.mean(image[chosen_pix]))

    fi_mins = np.asarray(fi_mins)
    fi_maxs = np.asarray(fi_maxs)
    r_mins = np.asarray(r_mins)
    r_maxs = np.asarray(r_maxs)
    mean_intensity_list = np.asarray(mean_intensity_list)



    for idx1, ab_step in enumerate(abs_steps[:-1]):
        for idx2, an_step in enumerate(angle_steps[:-1]):

            chosen_pix = (abs_pix >= ab_step) & (abs_pix < abs_steps[idx1+1]) & (angle_pix >= an_step) & (angle_pix < angle_steps[idx2+1])

            if np.sum(chosen_pix) == 0:
                continue
            elif np.sum(chosen_pix) < 2:
                mean_grad_x = 0
                mean_grad_y = 0
                grad_angle = None
            else:
                mean_grad_x = np.mean(grad_x[chosen_pix])
                mean_grad_y = np.mean(grad_y[chosen_pix])

                grad_angle = np.arctan2( mean_grad_y, mean_grad_x ) # np.rad2deg( )

            mean_intens = np.mean(image[chosen_pix])
            # max_grad_point = np.argmax(np.sqrt(grad_x[chosen_pix]**2 + grad_y[chosen_pix]**2))  #

            mean_x = np.mean(x[chosen_pix]) #   x[chosen_pix][max_grad_point]  #
            mean_y = np.mean(y[chosen_pix]) #   y[chosen_pix][max_grad_point]  #

            # уравнение для прямой градиента
            # ylingrad = tang * x + mean_x
            A = mean_intens
            B = mean_intens

            if grad_angle == None:
                pass
            elif grad_angle >= an_step and grad_angle < angle_steps[idx2+1]:
                if ab_step == abs_steps[0]:
                    A = mean_intens
                    B = mean_intens
                else:
                    A = mean_intensity_list[(r_mins == abs_steps[idx1-1])&(r_maxs == abs_steps[idx1])&(fi_mins == an_step)]

                    if abs_steps[idx1+1] == abs_steps[-1]:
                        B = mean_intens
                    else:
                        B = mean_intensity_list[(r_mins == abs_steps[idx1+1])&(r_maxs == abs_steps[idx1+2])&(fi_mins == an_step)]

                # print(A, B, mean_intens)
                # mean_intensity[chosen_pix] = mean_intens  # np.random.randint(100, 255, 1)

            elif ((grad_angle-np.pi) >= an_step and (grad_angle-np.pi) < angle_steps[idx2+1]) or ( (grad_angle+np.pi) >= an_step and (grad_angle+np.pi) < angle_steps[idx2+1]):
                if ab_step == abs_steps[0]:
                    A = mean_intens
                    B = mean_intens
                else:
                    B = mean_intensity_list[(r_mins == abs_steps[idx1-1])&(r_maxs == abs_steps[idx1])&(fi_mins == an_step)]

                    if abs_steps[idx1+1] == abs_steps[-1]:
                        A = mean_intens
                    else:
                        A = mean_intensity_list[(r_mins == abs_steps[idx1+1])&(r_maxs == abs_steps[idx1+2])&(fi_mins == an_step)]

            elif grad_angle >= 0:
                if ab_step == abs_steps[0]:
                    A = mean_intens
                    B = mean_intens
                else:
                    A = mean_intensity_list[(fi_mins == angle_steps[idx1-1])&(fi_maxs == angle_steps[idx1])&(r_mins == ab_step)]

                    if abs_steps[idx1+1] == abs_steps[-1]:
                        # print("Hello")
                        B = mean_intensity_list[(fi_mins >= angle_steps[idx1+1])&(r_mins == ab_step)] # (fi_maxs == angle_steps[-1])&
                    else:
                        B = mean_intensity_list[(fi_mins == angle_steps[idx1+1])&(fi_maxs == angle_steps[idx1+2])&(r_mins == ab_step)]

                # mean_intensity[chosen_pix] = 255
            elif grad_angle < 0:
                if ab_step == abs_steps[0]:
                    A = mean_intens
                    B = mean_intens
                else:
                    B = mean_intensity_list[(fi_mins == angle_steps[idx1-1])&(fi_maxs == angle_steps[idx1])&(r_mins == ab_step)]

                    if abs_steps[idx1+1] == abs_steps[-1]:
                        A = mean_intensity_list[(fi_mins == angle_steps[idx1+1])&(fi_maxs == angle_steps[-1])&(r_mins == ab_step)]
                    else:
                        A = mean_intensity_list[(fi_mins == angle_steps[idx1+1])&(fi_maxs == angle_steps[idx1+2])&(r_mins == ab_step)]


            # print(A, B, mean_intens)
            if B.size == 0:
                B = mean_intens

            a_tmp_arg_min = np.argmin( (mean_intensity - A)**2 )
            A_x = x.ravel()[a_tmp_arg_min]
            A_y = y.ravel()[a_tmp_arg_min]

            b_tmp_arg_min = np.argmin( (mean_intensity - B)**2 )
            B_x = x.ravel()[b_tmp_arg_min]
            B_y = y.ravel()[b_tmp_arg_min]

            # dist = np.sqrt( (A_x - B_x)**2 + (A_y - B_y)**2 )

            mean_xs.append(mean_x)
            mean_ys.append(mean_y)
            # mean_x += (B_x - A_x) / dist * np.cos(grad_angle)
            # mean_y += (B_y - A_y) / dist * np.sin(grad_angle)

            res_image[chosen_pix] = mean_intens + mean_grad_x * (x[chosen_pix] - mean_x) + mean_grad_y * (y[chosen_pix] - mean_y)
            onlygrad[chosen_pix] = mean_grad_x * (x[chosen_pix] - mean_x) + mean_grad_y * (y[chosen_pix] - mean_y)




    res_image[res_image < 0] = 0
    res_image[res_image > 255] = 255
    mean_xs = np.asarray(mean_xs)
    mean_ys = np.asarray(mean_ys)
    return res_image, mean_intensity, onlygrad, mean_xs, mean_ys, abs_steps, angle_steps



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

        x = []

        # Пересечение с лучами
        x_tmp = x_mean / (np.tan([fi_min, fi_max]) - tan_fi_grad)
        x.append(x_tmp)

        # пересечение с окружностями
        a = 1 + tan_fi_grad ** 2
        b = 2 * tan_fi_grad * x_mean
        c = x_mean**2 - np.array([abs_min, abs_max])**2
        D = b**2 - 4 * a * c

        x_tmp = (-b + np.sqrt(D)) / (2 * a)
        x.append(x_tmp)

        x_tmp = (-b - np.sqrt(D)) / (2 * a)
        x.append(x_tmp)

        x = np.asarray(x)
        y = tan_fi_grad * x + x_mean

    dists_arg = np.argsort(x**2 + y**2)[0:2]

    x = x[dists_arg]
    y = y[dists_arg]

    return x, y




