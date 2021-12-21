import numpy as np
from skimage import data, filters
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.interpolate import griddata

image = data.camera().astype(np.float)

Len_y, Len_x = image.shape
# image = image[:250, :250]
# image = image / 256
# print(image.shape)


x, y = np.meshgrid(np.linspace(-1, 1, Len_y), np.linspace(1, -1, Len_x))

abs_pix = np.sqrt((x**2 + y**2)).ravel()
angle_pix = np.arctan2(y, x).ravel()

angle_array, abs_array = np.meshgrid(np.linspace(-np.pi, np.pi, 900), np.linspace(0, np.sqrt(2), 900))

polar_image = griddata( (angle_pix, abs_pix), image.ravel(), (angle_array.ravel(), abs_array.ravel()), method='nearest') #nearest cubic linear

polar_image = polar_image.reshape(900, 900)

grad_r, grad_fi = np.gradient(polar_image)

grad_r = downscale_local_mean(grad_r, (30, 30) )
grad_fi = downscale_local_mean(grad_fi, (30, 30) )
abs_array = abs_array[::30, ::30]
angle_array = angle_array[::30, ::30]
polar_image = np.zeros((30, 30), dtype=np.float)


grad_r_na = grad_r[:, :450]
grad_r_na = np.fliplr(grad_r_na)
polar_image[:, :450] = np.fliplr( cumtrapz(grad_r_na, axis=0, initial=0) )
polar_image[:, 450:] = cumtrapz(grad_r[:, 450:], axis=0, initial=0)

# polar_image = cumtrapz(grad_r, axis=0, initial=0)
# integral_over_fi = 0
# for i in range( grad_fi.shape[1] - 2 ):
#     integral_over_fi += cumtrapz(np.roll(grad_fi, i+1, axis=1), axis=1, initial=0)
# integral_over_fi /= (i+1)
# polar_image += integral_over_fi


newx_pix = (abs_array * np.cos(angle_array)).ravel()
newy_pix = (abs_array * np.sin(angle_array)).ravel()

res_image = griddata( (newx_pix, newy_pix), polar_image.ravel(), (x.ravel(), y.ravel()), method='nearest') #nearest cubic linear

res_image = res_image.reshape(Len_y, Len_x)


fig, ax = plt.subplots(nrows=1, ncols=3)
ax[0].imshow(image, cmap="gray")
ax[1].imshow(polar_image, cmap="gray")
ax[2].imshow(res_image, cmap="gray")
plt.show()





