import numpy as np
import matplotlib.pyplot as plt



theta_deg = 135
theta_rad = np.deg2rad(theta_deg)

Len_x = 500
Len_y = 500


x = np.linspace(-1, 1, Len_x)
y = np.linspace(-1, 1, Len_y)

xx, yy = np.meshgrid( x, y ) 


xx_rot = xx * np.cos(theta_rad) + yy * np.sin(theta_rad)
# yy_rot = yy * np.cos(theta_rad) - xx * np.sin(theta_rad)


sine = np.cos(2 * np.pi * xx_rot * 2)


grad_y, grad_x = np.gradient(sine)
grad_abs = np.sqrt(grad_x**2 + grad_y**2)
grad_angle = np.arctan2(grad_y, grad_x)


grad_hist, bin_angles = np.histogram(grad_angle, bins=70, range=[-np.pi, np.pi], density=True, weights=grad_abs  )
# bin_angles = np.rad2deg(bin_angles)


fig = plt.figure()
ax1 = plt.subplot(121)
ax2 = plt.subplot(122, projection='polar')
ax1.pcolor(x, y, sine, cmap="gray")

ax2.plot(bin_angles[1:], grad_hist)

plt.show()

