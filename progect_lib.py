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

def make_preobr(image, x, y, params):

    delta_x = x[0, 1] - x[0, 0]
    delta_y = y[1, 0] - y[0, 0]

    minmax_xy = [np.min(x), np.max(x), np.min(y), np.max(y)]
    # print(minmax_xy)
    # return
    # x = x - center_x
    # y = y - center_y

    # label = 'NRMSE = {:.4f}'

    abs_pix = np.sqrt((x**2 + y**2))
    angle_pix = np.arctan2(y, x)


    grad_y, grad_x = np.gradient(image, delta_y, delta_x)



    abs_steps = np.geomspace(0.01*(x.max() - x.min()), np.max(abs_pix)+0.0001, 30)  # np.linspace(np.min(abs_pix), np.max(abs_pix)+0.0001, 10) #
    angle_steps = np.linspace(-np.pi, np.pi+0.0001, 30)

    Len_y, Len_x = image.shape
    # xx_rete, yy_rete = get_rete(abs_steps, angle_steps, Len_x, Len_y, onesformat=True)

    # mask = np.zeros_like(image)

    res_image = np.zeros_like(image, dtype=float)
    mean_intensity = np.zeros_like(image, dtype=float)



    # mean_intensity_list = []

    mean_xs = np.empty((0), dtype=np.float)
    mean_ys = np.empty_like(mean_xs)

    if not params["use_origin"]:
        for idx1, ab_step in enumerate(abs_steps):
            for idx2, an_step in enumerate(angle_steps[:-1]):

                if idx1 == 0:
                    chosen_pix = (abs_pix <= ab_step) & (angle_pix >= an_step) & (angle_pix < angle_steps[idx2+1])

                # elif idx1+1 == abs_steps.size:
                #     chosen_pix = (abs_pix > ab_step) & (angle_pix >= an_step) & (angle_pix < angle_steps[idx2 + 1])

                else:
                    chosen_pix = (abs_pix <= ab_step) & (abs_pix > abs_steps[idx1-1]) & (angle_pix >= an_step) & (angle_pix < angle_steps[idx2+1])

                if np.sum(chosen_pix) == 0:
                    continue

                # mask[chosen_pix] = np.random.randint(0, 255)

                # mean_xs = np.append(mean_xs, np.mean(x[chosen_pix]))
                # mean_ys = np.append(mean_ys, np.mean(y[chosen_pix]))

                mean_intensity[chosen_pix] = np.mean(image[chosen_pix])
                # mean_intensity_list.append(np.mean(image[chosen_pix]))
    else:
        mean_intensity = image


    cols_x_A, cols_y_A, cols_x_B, cols_y_B = [], [], [], []
    # mean_intensity_list = np.asarray(mean_intensity_list)
    xs_AB, ys_AB = [], []

    # print(mean_intensity_list.size, np.unique(mean_intensity_list).size)

    global_counter = 0
    for idx1, ab_step in enumerate(abs_steps):
        for idx2, an_step in enumerate(angle_steps[:-1]):

            if idx1 == 0:
                chosen_pix = (abs_pix <= ab_step) & (angle_pix >= an_step) & (angle_pix < angle_steps[idx2+1])
            else:
                chosen_pix = (abs_pix <= ab_step) & (abs_pix > abs_steps[idx1-1]) & (angle_pix >= an_step) & (angle_pix < angle_steps[idx2+1])

            if np.sum(chosen_pix) == 0:
                continue
            elif np.sum(chosen_pix) < 2:
                mean_intens = np.mean(image[chosen_pix])
                res_image[chosen_pix] = mean_intens
                continue

            mean_intens = np.mean(image[chosen_pix])



            if params["use_average_grad"]:
                if params["avg_grad_over"]:
                    mean_grad_x, mean_grad_y = get_new_mean_grad(grad_x[chosen_pix], grad_y[chosen_pix], x[chosen_pix], y[chosen_pix])
                else:
                    mean_grad_x = np.mean(grad_x[chosen_pix])
                    mean_grad_y = np.mean(grad_y[chosen_pix])
            else:
                # use mode of gradient
                print("Hello")
                mean_grad_x, mean_grad_y = get_grad_mode(grad_x[chosen_pix], grad_y[chosen_pix])



            mean_x = np.mean(x[chosen_pix])
            mean_y = np.mean(y[chosen_pix])

            x_AB, y_AB = get_distance2AB(an_step, angle_steps[idx2+1], abs_steps[idx1-1], ab_step, mean_grad_x, mean_grad_y, mean_x, mean_y, minmax_xy)
            xs_AB.append(x_AB)
            ys_AB.append(y_AB)


            val_A, val_B, col_x_A, col_y_A, col_x_B, col_y_B = get_AB_value(x_AB, y_AB, x, y, mean_intensity, chosen_pix, mean_intensity)

            if not params["apply_grad_in_extrems"]:
                if (val_A > mean_intens and val_B > mean_intens) or (val_A < mean_intens and val_B < mean_intens):
                    res_image[chosen_pix] = mean_intens
                    continue


            cols_x_A.append(col_x_A)
            cols_y_A.append(col_y_A)
            cols_x_B.append(col_x_B)
            cols_y_B.append(col_y_B)

            # if  idx1+2 != abs_steps.size:
            #     print("last")

            mean_x_new = mean_x
            mean_y_new = mean_y

            if val_A - val_B != 0: #  or val_A > mean_intens and

                if params["shift_xy_mean"]:
                    mean_x_new = (val_A*x_AB[0] - val_B*x_AB[1] + mean_intens*(x_AB[1] - x_AB[0]) ) / (val_A - val_B)
                    mean_y_new = (val_A*y_AB[0] - val_B*y_AB[1] + mean_intens*(y_AB[1] - y_AB[0]) ) / (val_A - val_B)

                grad_component = mean_grad_x * (x[chosen_pix] - mean_x_new) + mean_grad_y * (y[chosen_pix] - mean_y_new)

                if params["apply_smooth_grad_dist"]:
                    grad_component_weights = np.exp( -100.0 * ( (mean_x_new - x[chosen_pix] )**2 + (mean_y_new - y[chosen_pix] )**2  )  )
                    if grad_component_weights.size == 0:
                        grad_component_weights = 1.0
                    else:
                        grad_component_weights /= np.max(grad_component_weights)
                else:
                    grad_component_weights = 1.0

                res_image[chosen_pix] = mean_intens + grad_component_weights*grad_component


                if params["apply_AB_thresh"]:
                    res_image_tmp = res_image[chosen_pix]
                    res_image_tmp[res_image_tmp <= val_A] = val_A
                    res_image_tmp[res_image_tmp >= val_B] = val_B
                    res_image[chosen_pix] = res_image_tmp

            else:
                print("Ua and Ub are equal!")
                res_image[chosen_pix] = mean_intens

            mean_xs = np.append(mean_xs, mean_x_new)
            mean_ys = np.append(mean_ys, mean_y_new)

            # print(np.sum(chosen_pix))  Антоха, все отправил?
            # z = np.ones_like(image)
            # z[chosen_pix] = res_image[chosen_pix]
            # fig, ax = plt.subplots(nrows=1, ncols=1)
            # ax.pcolor(x[0, :], y[:, 0], z, cmap='gray', vmin=0, vmax=1)
            # # print(y[:, 0])
            # ax.plot(x_AB, y_AB, color="green")
            # ax.scatter(mean_x, mean_y, color="red")
            # ax.scatter(mean_x_new, mean_y_new, color="blue")
            # ax.scatter(col_x_A, col_y_A, color="m", s=50)
            # ax.scatter(col_x_B, col_y_B, color="m", s=50)
            # nrmse = np.sqrt( np.mean( (res_image[chosen_pix] - image[chosen_pix])**2 ) )
            # ax.set_xlabel(label.format(nrmse))
            #
            # x_min2plot = np.min( x[chosen_pix] ) - delta_x
            # x_max2plot = np.max( x[chosen_pix] ) + delta_x
            #
            # x_min2plot -= 0.2*np.abs(x_min2plot)
            # x_max2plot += 0.2*np.abs(x_max2plot)
            #
            # # print(x_min2plot, x_max2plot)
            # ax.set_xlim(x_min2plot, x_max2plot)
            #
            # y_min2plot = np.min(y[chosen_pix]) - delta_y # - 0.2 * np.abs(np.min(y[chosen_pix]))
            # y_max2plot = np.max(y[chosen_pix]) + delta_y # + 0.2 * np.abs(np.max(y[chosen_pix]))
            #
            # y_min2plot -= 0.2 * np.abs(y_min2plot)
            # y_max2plot += 0.2 * np.abs(y_max2plot)
            #
            #
            # ax.set_ylim(y_min2plot, y_max2plot)
            #
            # for x_rete, y_rete in zip(xx_rete, yy_rete):
            #     ax.plot(x_rete, y_rete, color="b")
            #
            #
            # fig.savefig("./results/each_segment/" + str(global_counter)  + ".png")
            # # plt.show(block=False)
            # # # break
            # plt.close(fig)


            global_counter += 1

    # cols_x_A = np.asarray(cols_x_A) + center_x
    # cols_y_A = np.asarray(cols_y_A) + center_y
    # cols_x_B = np.asarray(cols_x_B) + center_x
    # cols_y_B = np.asarray(cols_y_B) + center_y
    # mean_xs = np.asarray(mean_xs) + center_x
    # mean_ys = np.asarray(mean_ys) + center_y

    cols_AB = [cols_x_A, cols_y_A, cols_x_B, cols_y_B, mean_xs, mean_ys]
    return res_image, mean_intensity, mean_xs, mean_ys, abs_steps, angle_steps, xs_AB, ys_AB, cols_AB   # , mask



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

def get_rete(abs_steps, angle_steps, Len_x, Len_y, centr_x = 0, centr_y = 0,  onesformat=False):
    xx = []
    yy = []

    for ab in abs_steps:
        x = np.linspace(-ab, ab, 1000)
        y = np.sqrt(ab**2 - x**2)
        xx.append( np.copy(x) )
        xx.append( np.copy(x) )
        yy.append(y + centr_y )
        yy.append(-y + centr_y)

    for an in angle_steps:
        x = 0
        # print(an)
        if np.abs(an) == np.pi:
            x = np.array([-2, 0], dtype=float)
            y = np.zeros_like(x) + centr_y
        elif an == 0:
            x = np.array([0, 2], dtype=float)
            y = np.zeros_like(x) + centr_y
        elif an == np.pi / 2:
            x = np.array([0, 0], dtype=float)
            y = np.linspace(0, 2, 2) + centr_y
        elif an == -np.pi / 2:
            x = np.array([0, 0], dtype=float)
            y = np.linspace(-2, 0, 2)  + centr_y
        elif np.abs(an) < np.pi/2:
            x = np.array([0, 2], dtype=float)
            k = np.tan(an)
            y = k * x + centr_y
        elif np.abs(an) > np.pi/2:
            x = np.array([-2, 0], dtype=float)
            k = np.tan(an)
            y = k * x  + centr_y
        if not (type(x) is int):
            xx.append(x)
            yy.append(y)


    for x in xx:
        x += centr_x

    if onesformat:
        return xx, yy


    for idx, (x, y) in enumerate(zip(xx, yy)):
        x_255, y_255 = rel2pix(x, y, Len_x, Len_y)
        xx[idx] = x_255
        yy[idx] = y_255

    return xx, yy


def get_distance2AB(fi_min, fi_max, abs_min, abs_max, grad_x, grad_y, x_mean, y_mean, minmax_xy):

    minx = minmax_xy[0]
    maxx = minmax_xy[1]
    miny = minmax_xy[2]
    maxy = minmax_xy[3]


    if grad_x == 0:
        x = np.asarray([x_mean, x_mean])

        tmp = np.array([abs_min, abs_max])**2 - x**2
        x = x[tmp >= 0 ]
        tmp = tmp[tmp >= 0 ]
        y = np.sqrt(tmp)

        x = np.append(x, x)
        y = np.append(y, -y)

        x = np.append(x, [x_mean, x_mean])
        y = np.append(y, x[0:2]*np.tan([fi_min, fi_max]))

    else:
        tan_fi_grad = grad_y / grad_x
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
    y = y.ravel()

    dists_arg = np.argsort( (x - x_mean)**2 + (y - y_mean)**2 )

    x_res = x[dists_arg[0:2]]
    y_res = y[dists_arg[0:2]]

    AB_dist = np.sqrt( (x_res[0] - x_res[1])**2 + (y_res[0] - y_res[1])**2 )
    mean_A_dist = np.sqrt( (x_res[0] - x_mean)**2 + (y_res[0] - y_mean)**2 )
    mean_B_dist = np.sqrt( (x_res[1] - x_mean)**2 + (y_res[1] - y_mean)**2 )

    if np.abs( mean_A_dist + mean_B_dist - AB_dist) >= 0.001:
        x_res = x[dists_arg[ np.array([0, 2], dtype=int) ]]
        y_res = y[dists_arg[ np.array([0, 2], dtype=int) ]]
        # print("Wrong distance!")

    # Цикл для переноса точек внутрь квадрата картинки
    for i, (xx, yy) in enumerate(zip (x_res, y_res)):

        is_x_outside = False
        if (xx < minx):
            x_res[i] = minx
            is_x_outside = True
        elif (xx > maxx):
            x_res[i] = maxx
            is_x_outside = True


        if is_x_outside:
            y_res[i] = ((y_mean - yy) * x_res[i] + yy * x_mean - xx * y_mean ) / (x_mean - xx)

        is_y_outside = False
        if (yy < miny):
            y_res[i] = miny
            is_y_outside = True

        elif (yy > maxy):
            y_res[i] = maxy
            is_y_outside = True

        if is_y_outside:
            x_res[i] = ((xx - x_mean) * y_res[i] + yy * x_mean - xx * y_mean ) / (yy - y_mean)


    # print("g", grad_x, grad_y)
    U = grad_x * (x_res - x_mean) + grad_y * (y_res - y_mean)
    Uarg = np.argsort(U)
    x_res = x_res[Uarg]
    y_res = y_res[Uarg]

    return x_res, y_res

def get_AB_value(x_AB, y_AB, pix_x, pix_y, mean_intensity, chosen_pix, original_image):

    chosen_pix_invert = np.logical_not(chosen_pix)

    for idx in range(2):

        if np.abs(x_AB[idx]) == 1 or np.abs(y_AB[idx]) == 1:
            res_idx = np.argmin( (x_AB[idx] - pix_x) ** 2 + (y_AB[idx] - pix_y) ** 2)
            if idx == 0:
                col_x_A = pix_x.ravel()[res_idx]
                col_y_A = pix_y.ravel()[res_idx]
                value_A = original_image[(pix_x == pix_x.ravel()[res_idx]) & ((pix_y == pix_y.ravel()[res_idx]))]
            else:
                col_x_B = pix_x.ravel()[res_idx]
                col_y_B = pix_y.ravel()[res_idx]
                value_B = original_image[(pix_x == pix_x.ravel()[res_idx]) & ((pix_y == pix_y.ravel()[res_idx]))]

        else:
            res_idx = np.argmin( (x_AB[idx] - pix_x[chosen_pix_invert])**2 + (y_AB[idx] - pix_y[chosen_pix_invert])**2 )
            if idx == 0:
                col_x_A = pix_x[chosen_pix_invert][res_idx]
                col_y_A = pix_y[chosen_pix_invert][res_idx]
                value_A = mean_intensity[ (pix_x == pix_x[chosen_pix_invert][res_idx])&((pix_y == pix_y[chosen_pix_invert][res_idx]))   ]
            else:
                col_x_B = pix_x[chosen_pix_invert][res_idx]
                col_y_B = pix_y[chosen_pix_invert][res_idx]
                value_B = mean_intensity[ (pix_x == pix_x[chosen_pix_invert][res_idx])&((pix_y == pix_y[chosen_pix_invert][res_idx]))   ]

    # print(value_A, value_B)
    return value_A, value_B, col_x_A, col_y_A, col_x_B, col_y_B


def get_new_mean_grad(grad_x, grad_y, x, y):

    mean_grad_x = np.mean(grad_x)
    mean_grad_y = np.mean(grad_y)

    ampl_mean = np.sqrt(mean_grad_x**2 + mean_grad_y**2)

    if ampl_mean < 0.0001:
        return mean_grad_x, mean_grad_y


    mean_grad_x /= ampl_mean
    mean_grad_y /= ampl_mean

    project_on_mean_grad = grad_x * mean_grad_x + grad_y * mean_grad_y

    if grad_x.size < 15:
        max_proj = np.max(project_on_mean_grad)
        mean_grad_x = mean_grad_x * max_proj
        mean_grad_y = mean_grad_y * max_proj

        return mean_grad_x, mean_grad_y

    grad_angle = np.arctan2(mean_grad_y, mean_grad_x)

    # print( np.rad2deg( grad_angle) )
    x_ = x * np.cos(grad_angle) + y * np.sin(grad_angle)
    bins = 10 # np.arange(np.min(x_), np.max(x_), step_x_) #!!!!!! step = np.median( np.diff( np.sort(x) ) )

    ones_hist =  np.histogram(x_, bins=bins)[0]
    weighted_hist =  np.histogram(x_, weights=project_on_mean_grad, bins=bins)[0]
    non_zeros = ones_hist != 0
    ampl = np.max( weighted_hist[non_zeros] / ones_hist[non_zeros] )

    mean_grad_x *= ampl
    mean_grad_y *= ampl

    return mean_grad_x, mean_grad_y


def circular_distribution(amples, angles, angle_step, nkernel=15):
    from scipy.ndimage.filters import convolve1d
    from scipy.signal.windows import parzen

    kernel = parzen(nkernel)
    bins = np.arange(-np.pi, np.pi+angle_step, angle_step)
    distr, _ = np.histogram(angles, bins=bins, weights=amples)


    distr = convolve1d(distr, kernel, mode="wrap")
    bins = np.convolve(bins, [0.5, 0.5], mode="valid")

    return bins, distr

def get_grad_mode(grad_x, grad_y):
    amples = np.sqrt(grad_x**2 + grad_y**2)
    angles = np.arctan2(grad_y, grad_x)
    angle_step = 0.01

    bins, distr = circular_distribution(amples, angles, angle_step, nkernel=150)

    max_idx = np.argmax(distr)

    mode_grad_x = distr[max_idx] * np.cos( bins[max_idx] )
    mode_grad_y = distr[max_idx] * np.sin( bins[max_idx] )

    return mode_grad_x, mode_grad_y