import numpy as np
import matplotlib.pyplot as plt
from HyperColomn2D import HyperColumn


def Background(x_, y_):
    A = (x_-1.3)*0.5 + (y_-0.7)*0.8
    return A


def AmplitudesOfGratings(x_, y_, sigma=0.5, cent_x=0.2, cent_y=0.2):
    A = np.exp( -0.5*((x_ - cent_x)/sigma)**2 - 0.5*((y_ - cent_y)/sigma)**2 )
    return A


def Gratings(freq, xx, yy, sigma=0.5, cent_x=0.2, cent_y=0.2, direction=2.3, phi0=0):
    xx_rot = xx * np.cos(direction) - yy * np.sin(direction)
    #yy_rot = xx * np.sin(direction) + yy * np.cos(direction)
    image = np.cos(2 * np.pi * xx_rot * freq + phi0) * AmplitudesOfGratings(xx, yy, sigma, cent_x, cent_y) + Background(xx, yy)
    return image

image_shift_x = 0 # 0.2
image_shift_y = 0 # 0.2
Nx = 200
Ny = 200

yy, xx = np.meshgrid(np.linspace(-1, 1, Ny), np.linspace(-1, 1, Nx))

sgmGauss = 0.1
r = np.sqrt(image_shift_x**2 + image_shift_y**2)
delta_x = xx[1, 0] - xx[0, 0]
delta_y = yy[0, 1] - yy[0, 0]
freq_teor_max = 0.5 / (np.sqrt(delta_x**2 + delta_y**2))
sigma_teor_min = 1 / (2 * np.pi * freq_teor_max)
nsigmas = 4
sigminimum = sigma_teor_min + 0.01 * r  # + 0.2*r
sigmaximum = 100 * sigminimum  # ??????? ?? r
sigmas = np.geomspace(sigminimum, sigmaximum, nsigmas)
directions = np.linspace(-np.pi, np.pi, 32, endpoint=False)
frequencies = np.asarray([5, ])
hc = HyperColumn(image_shift_x, image_shift_y, xx, yy, directions, sigmas, sgmGauss, frequencies=frequencies, params={})



# Freq = 8
#Direction = np.pi/5 #  np.pi / 2
# ph_0 = 0
Sgm = 0.1

for iter_test in range(10):

    real_encoded = {'peak_freq': np.random.uniform(1.0, 15.0), # 8,
                    'phase_0': np.random.uniform(-np.pi, np.pi), #1.5,
                    'dominant_direction': np.random.uniform(0, np.pi), #0.25*np.pi,
                    }

    Freq = real_encoded['peak_freq']
    Direction = real_encoded['dominant_direction']
    phase_0 = real_encoded['phase_0']

    image = Gratings(Freq, xx, yy, sigma=Sgm, cent_x=image_shift_x, cent_y=image_shift_y, direction=Direction, phi0=phase_0)

    #image += np.random.normal(0, 0.2, image.shape) # !!!!

    calc_encoded = hc.encode(image)

    for key, val in real_encoded.items():
        print(key, val, calc_encoded[0][key])


    print("#######################################################")



    # plt.imshow(image, cmap='gray')
    # plt.show()