import numpy as np
from skimage import data, filters
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.interpolate import griddata

def myrepeat(a, repf):
    sizex, sizey = a.shape
    a = np.repeat(a, repf, axis=0)
    a = np.repeat(a, repf, axis=1).reshape(sizex*repf, sizey*repf)
    return a


Nxhypercol = 30
Nyhypercol = 30
Nhypercol = Nxhypercol * Nyhypercol


image = data.camera().astype(np.float)

Len_y, Len_x = image.shape
# image = image[:250, :250]
# image = image / 256
# print(image.shape)


x, y = np.meshgrid(np.linspace(-1, 1, Len_y), np.linspace(1, -1, Len_x))

abs_pix = np.sqrt((x**2 + y**2)).ravel()
angle_pix = np.arctan2(y, x).ravel()

angle_array, abs_array = np.meshgrid(np.linspace(-np.pi, np.pi, Nhypercol), np.linspace(0, np.sqrt(2), Nhypercol))

polar_image = griddata( (angle_pix, abs_pix), image.ravel(), (angle_array.ravel(), abs_array.ravel()), method='nearest') #nearest cubic linear
polar_image = polar_image.reshape(Nhypercol, Nhypercol)

grad_r, grad_fi = np.gradient(polar_image)


grad_r = downscale_local_mean(grad_r, (Nxhypercol, Nyhypercol) )
grad_r = myrepeat(grad_r, Nxhypercol)
grad_fi = downscale_local_mean(grad_fi, (Nxhypercol, Nyhypercol) )
grad_fi = myrepeat(grad_fi, Nxhypercol)
polar_image_down_sampled = downscale_local_mean(polar_image, (Nxhypercol, Nyhypercol) )
polar_image_down_sampled = myrepeat(polar_image_down_sampled, Nxhypercol)



polar_image = grad_r * abs_array + grad_fi * angle_array + polar_image_down_sampled


"""
abs_array = abs_array[::Nxhypercol, ::Nyhypercol]
angle_array = angle_array[::Nxhypercol, ::Nyhypercol]
polar_image = cumtrapz(grad_r, axis=0, initial=0) + downscale_local_mean(polar_image, (Nxhypercol, Nyhypercol) )
"""

newx_pix = (abs_array * np.cos(angle_array)).ravel()
newy_pix = (abs_array * np.sin(angle_array)).ravel()

res_image = griddata( (newx_pix, newy_pix), polar_image.ravel(), (x.ravel(), y.ravel()), method='nearest') #nearest cubic linear

res_image = res_image.reshape(Len_y, Len_x)


fig, ax = plt.subplots(nrows=1, ncols=3)
ax[0].imshow(image, cmap="gray")
ax[1].imshow(polar_image, cmap="gray")
ax[2].imshow(res_image, cmap="gray")
plt.show()






