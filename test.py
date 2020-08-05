import numpy as np
import progect_lib as lib
import matplotlib.pyplot as plt

N = 1000
amples = np.ones(N)
angles = 0.2 * np.random.randn(N) + np.pi
angles[angles >= np.pi] -= 2*np.pi




angle_step = 0.01

x, y = lib.circular_distribution(amples, angles, angle_step, nkernel=150)

x[0] = -np.pi
x[-1] = np.pi


plt.polar(x, y)
plt.show()

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
