import numpy as np
from scipy.ndimage import convolve



def make_preobr(image, x, y, params):

    # delta_x = x[0, 1] - x[0, 0]
    # delta_y = y[1, 0] - y[0, 0]


    abs_pix = np.sqrt((x**2 + y**2))
    angle_pix = np.arctan2(y, x)


    
    abs_steps = np.geomspace(0.01*(x.max() - x.min()), np.max(abs_pix)+0.0001, 30)  # np.linspace(np.min(abs_pix), np.max(abs_pix)+0.0001, 10) #
    angle_steps = np.linspace(-np.pi, np.pi+0.0001, 30)

    Len_y, Len_x = image.shape

    res_image = np.zeros_like(image, dtype=float)
    mean_intensity = np.zeros_like(image, dtype=float)

    mean_xs = np.empty((0), dtype=np.float)
    mean_ys = np.empty_like(mean_xs)

 
    global_counter = 0
    for idx1, ab_step in enumerate(abs_steps):
        for idx2, an_step in enumerate(angle_steps[:-1]):

            if idx1 == 0:
                chosen_pix = (abs_pix <= ab_step) & (angle_pix >= an_step) & (angle_pix < angle_steps[idx2+1])
            else:
                chosen_pix = (abs_pix <= ab_step) & (abs_pix > abs_steps[idx1-1]) & (angle_pix >= an_step) & (angle_pix < angle_steps[idx2+1])


            field_square = np.sum(chosen_pix) #  * delta_x * delta_y
            
            if field_square == 0:
                continue
            elif field_square < 2:
                res_image[chosen_pix] = image[chosen_pix]
                continue

            field_x = x[chosen_pix]
            field_y = y[chosen_pix]
            
            
            
            
            field_center_x = np.mean(field_x)
            field_center_y = np.mean(field_y)
            
            
            field_radius = max([np.max( np.abs(field_y-field_center_y) ), np.max( np.abs(field_x-field_center_x) )])
            
            
            kernel_sigma = field_radius
            field_kernel = np.exp(  -0.5*(x - field_center_x)**2/kernel_sigma**2 - 0.5*(y - field_center_y)**2/kernel_sigma**2   )
            field_kernel = field_kernel / np.sum(field_kernel)
            receptive_field_responce = image * field_kernel
            
            sigma_long = field_radius * params["sigma_multipl"]
            sigma_short = 0.5 * sigma_long
            
            center_idx = np.argmax(field_kernel)

            grad_x, grad_y = get_gradient_by_DOG(receptive_field_responce, sigma_long, sigma_short, center_idx, angle_step=0.4)


            mean_intens = np.sum(receptive_field_responce)


            res_image[chosen_pix] = mean_intens + grad_x * (field_x - field_center_x) + grad_y * (field_y - field_center_y)

            min_th = 0
            max_th = 255

            res_image_tmp = res_image[chosen_pix]
            res_image_tmp[res_image_tmp <= min_th] = min_th
            res_image_tmp[res_image_tmp >= max_th] = max_th
            res_image[chosen_pix] = res_image_tmp

            global_counter += 1



    return res_image



def get_gradient_by_DOG(receptive_field_responce, sigma_long, sigma_short, center_idx, angle_step=0.1):
    
    a = 0.5 / sigma_short**2
    b = 0.5 / sigma_long**2
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    xx, yy = np.meshgrid(x, y)


    ample_grads = []
    angles = np.arange(-np.pi, np.pi, angle_step)
    
    for fi in angles:
        xv = xx * np.cos(fi) + yy * np.sin(fi)
        yv = -xx * np.sin(fi) + yy * np.cos(fi)

        gauss2d = np.exp(-a*xv**2 - b*yv**2 ) 
        gauss_grad_x = gauss2d * (-2*a*xv)
        # gauss_grad_y = gauss2d * (-2*b*yv)
        
        
        ample_grad = convolve(receptive_field_responce, gauss_grad_x, mode='constant')
        
        ample_grad = ample_grad.ravel()[center_idx]
        ample_grads.append(ample_grad)
    
    ample_grads = np.asarray(ample_grads)
    max_idx = np.argmax(ample_grads)
    
    grad_x = ample_grads[max_idx] * np.cos(angles[max_idx])
    grad_y = ample_grads[max_idx] * np.sin(angles[max_idx])
    return grad_x, grad_y











