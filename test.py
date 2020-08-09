import numpy as np
import progect_lib2 as lib
import matplotlib.pyplot as plt

params = {}


x = np.linspace(-1, 1, 500)
y = np.linspace(-1, 1, 500)

xx, yy = np.meshgrid(x, y)


sine = 255 * 0.5 * (np.cos(2 * np.pi * xx * 6) + 1)


res_image = lib.make_preobr(sine, xx, yy, params)

fig, axes = plt.subplots(nrows=2, ncols=1)

axes[0].pcolor(x, y, sine, cmap='gray', vmin=0, vmax=255)
axes[1].pcolor(x, y, res_image, cmap='gray', vmin=0, vmax=255)

fig.savefig("/home/ivan/PycharmProjects/Vision/results/dog/test_dog.png")


plt.show()

"""
import numpy as np
import progect_lib as lib
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

sigma_x = 15 * 0.15
sigma_y = 15 * 0.3
r_xy = 0
fi = np.deg2rad(0)


a = 0.5 / sigma_x**2
b = 0.5 / sigma_y**2


x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)




xx, yy = np.meshgrid(x, y)

xv = xx * np.cos(fi) + yy * np.sin(fi)
yv = -xx * np.sin(fi) + yy * np.cos(fi)

gauss2d = np.exp(-a*xv**2 - b*yv**2 ) 
gauss_grad_x = gauss2d * (-2*a*xv)
gauss_grad_y = gauss2d * (-2*b*yv)


sine = np.cos(2 * np.pi * xx * 5) 


grad_sine_x = convolve(sine, gauss_grad_x, mode='constant')
grad_sine_y = convolve(sine, gauss_grad_y, mode='constant')

grad_ampl = np.sqrt(grad_sine_x**2 + grad_sine_y**2)

fig, axes = plt.subplots(nrows=1, ncols=4)

axes[0].pcolor(x, y, sine)

axes[1].pcolor(x, y, grad_sine_x)
axes[2].pcolor(x, y, grad_sine_y)
axes[3].pcolor(x, y, grad_ampl)



plt.show()
"""



"""
import numpy as np
import progect_lib as lib
import matplotlib.pyplot as plt


deg = np.linspace(0, 40, 100)  # градусов
acc = 20 * 10**(-deg / 40)     # циклов на градус

exp_hypec_size = 0.5 / acc
my_hypec_size = np.geomspace(0.01, 1, 30 ) * 40  # 40 потому что считаем всесь обзор только 40 градусов

my_deg = np.linspace(0, 40, my_hypec_size.size)
my_acc = 0.5 / my_hypec_size  # полцикла на градус



fig, ax = plt.subplots(nrows=1, ncols=2)

ax[0].plot(deg, acc, label="Experiment")
ax[0].plot(my_deg, my_acc, label="Our model")


ax[1].plot(deg, exp_hypec_size, label="Experiment")
ax[1].plot(my_deg, my_hypec_size, label="Our model")





ax[0].legend()
ax[1].legend()

ax[0].set_title("Resolution")
ax[1].set_title("Hypercoloumn size")


ax[0].set_ylabel("Spatial frequency (Cyc/Deg)")
ax[1].set_ylabel("Size (Deg)")

for a in ax:
    a.set_xlabel("Eccentricity (Deg)")

plt.show()
"""

# image = np.random.rand(5, 5) # -x**2 + 1
#
# result = ndimage.maximum_filter(image, size=3, mode="constant", cval=1)
#
# print(image)
# print(result)

# A = -x**2 # np.random.rand(11)
# K = 45
# rollingmax = np.array([max(A[j:j+K]) for j in range(len(A)-K+1)])
#
# plt.subplot(211)
# plt.plot(x, A)
# plt.subplot(212)
# plt.plot( x[K//2 : -K//2+1], rollingmax)
#
#
# plt.show()


# print (rollingmax)
