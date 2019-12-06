import numpy as np
import matplotlib.pyplot as plt
import progect_lib as lib


r = 1
tan_grad_angl = -0.5
x_mean = -0.1



xring = np.linspace(-r, r, 1000)
yring = np.sqrt(r**2 - xring**2)
plt.plot(xring, yring, color="b")
plt.plot(xring, -yring, color="b")

yline = xring
xline = np.zeros_like(yline) + x_mean


plt.plot(xline, yline, color="b")

y1 = np.sqrt(r**2 - x_mean**2)
plt.scatter([x_mean, x_mean], [y1, -y1], color="r")


plt.show()



# abs_steps = np.geomspace(0.01, np.sqrt(2), 6)  # np.linspace(np.min(abs_pix), np.max(abs_pix)+0.0001, 30)
# angle_steps = np.linspace(-np.pi, np.pi, 6)
# xx, yy = lib.get_rete(abs_steps, angle_steps, 255, 255)
# for x, y in zip(xx, yy):
#     plt.plot(x, y, color="b")
# plt.show()





