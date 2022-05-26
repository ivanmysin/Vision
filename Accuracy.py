import numpy as np
import matplotlib.pyplot as plt

from skimage import color, data, filters
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# import sys
# sys.path.append("../")

from HyperColomn2D import HyperColumn


def Background(x_, y_):
    A = (x_-1.3)*0.5 + (y_-0.7)*0.8
    return A


def AmplitudesOfGratings(x_, y_, sigma=0.5, cent_x=0.2, cent_y=0.2):
    A = np.exp( -0.5*((x_ - cent_x)/sigma)**2 - 0.5*((y_ - cent_y)/sigma)**2 )
    return A


def Gratings(freq, xx, yy, sigma=0.5, cent_x=0.2, cent_y=0.2, direction=2.3):
    xx_rot = xx * np.cos(direction*(1+2*xx)) - yy * np.sin(direction*(1+2*xx))
    #yy_rot = xx * np.sin(direction) + yy * np.cos(direction)
    image = np.cos(2 * np.pi * xx_rot * freq) * AmplitudesOfGratings(xx, yy, sigma, cent_x, cent_y) + Background(xx, yy)
    return image


##def decode(xx, yy, freq, direction, ph_0, xc, yc):
  ##  xx_ = xx * np.cos(direction) - yy * np.sin(direction)
    ##xhc_ = xc * np.cos(direction) - yc * np.sin(direction)
    ##image_restored = np.cos(2 * np.pi * (xx_ - xhc_) * freq + ph_0)

    # coeff4delout = ( (xx - xc)**2 + (yy - yc)**2 ) / ( np.sum( (xx - xc)**2 + (yy - yc)**2) )
    # image_restored = image_restored / coeff4delout # !!!! ?????????? ?????????? ??? !!!!!

    ##turn image_restored


######################################################

Freq = 8
Direction = np.pi/5 #  np.pi / 2
ph_0 = 0
Sgm = 0.3

errs = {
    "freq_error": 1.5 * 0,  # Hz
    "dir_error": 0.2 * 0,  # Rad
    "ph_0_error": 2.5 * 0,  # Rad
}

##### Stimulus ################################################################
image_shift_x = 0.1 
image_shift_y = 0.3
Nx = 200
Ny = 200
sgmGauss = 0.1
# yy, xx = np.meshgrid(np.linspace(-1, 1, Ny), np.linspace(-1, 1, Nx))
# image = Gratings(Freq, xx, yy, sigma=Sgm, cent_x=image_shift_x, cent_y=image_shift_y, direction=Direction)

image = data.camera() # color.rgb2gray(data.astronaut())
image = image.astype(np.float64)
image = image / np.max(image)
# image = filters.gaussian(image, sigma=5)
image = image[::-2, ::2]
Ny, Nx = image.shape
yy, xx = np.meshgrid(np.linspace(-1, 1, Ny), np.linspace(-1, 1, Nx))

xx[xx == 0] += 0.00000001
yy[yy == 0] += 0.00000001

# fig, axes = plt.subplots(ncols=1, figsize=(30, 15), sharex=True, sharey=True)
# axes.pcolor(xx[:, 0], yy[0, :], image, cmap='gray', shading='auto')
# plt.show()


delta_x = xx[1, 0] - xx[0, 0]
delta_y = yy[0, 1] - yy[0, 0]



params = {
    "use_circ_regression": False,
}
radiuses = np.geomspace(0.05, 0.95, 30)  # np.geomspace(0.1, 0.5, 10) #  #np.asarray([0.01, 0.05, 0.1]) #
# radiuses = radiuses[:5]

angles = np.linspace(-np.pi, np.pi, 30, endpoint=False)  # np.linspace(0, 0.5*np.pi, 10, endpoint=True) #
NGHs = int(radiuses.size * angles.size)  # ????? ????????????

hc_centers_x = np.zeros(NGHs, dtype=np.float64)
hc_centers_y = np.zeros(NGHs, dtype=np.float64)
hc_sigma_rep_field = np.zeros(NGHs, dtype=np.float64)
freq_c = np.zeros(NGHs, dtype=np.float64)
dir_c = np.zeros(NGHs, dtype=np.float64)
ph_c = np.zeros(NGHs, dtype=np.float64)
ampl_c = np.zeros(NGHs, dtype=np.float64)
bgrd_c = np.zeros(NGHs, dtype=np.float64)
grdX_c = np.zeros(NGHs, dtype=np.float64)
grdY_c = np.zeros(NGHs, dtype=np.float64)
image_restored_by_HCs = np.zeros((Nx, Ny, NGHs), dtype=np.float64)
receptive_fields = np.zeros((Nx, Ny, NGHs), dtype=np.float64)
Freq_restored_by_HCs = np.zeros((Nx, Ny, NGHs), dtype=np.float64)
Dir_restored_by_HCs = np.zeros((Nx, Ny, NGHs), dtype=np.float64)
Phi_restored_by_HCs = np.zeros((Nx, Ny, NGHs), dtype=np.float64)
Ampl_restored_by_HCs = np.zeros((Nx, Ny, NGHs), dtype=np.float64)
Grad_restored_by_HCs = np.zeros((Nx, Ny, NGHs), dtype=np.float64)
Bgrd_restored_by_HCs = np.zeros((Nx, Ny, NGHs), dtype=np.float64)

freq_teor_max = 0.5 / (np.sqrt(delta_x**2 + delta_y**2))
sigma_teor_min = 1 / (2 * np.pi * freq_teor_max)
directions = np.linspace(-np.pi, np.pi, 32, endpoint=False)  # 28

##### Prepare geometry of HCs #################################################
idx = 0
for r in radiuses:
    nsigmas = 4
    sigminimum = sigma_teor_min + 0.01 * r  # + 0.2*r
    sigmaximum = 100 * sigminimum  # ??????? ?? r

    sigmas = np.geomspace(sigminimum, sigmaximum, nsigmas)  # 28

    freq_max = freq_teor_max * 0.8 * (1 - r)  # 28
    freq_min = 0.05 * freq_max  # 28
    frequencies = np.asarray([2.0, ]) #np.asarray([5.0, ])  # np.geomspace(freq_min, freq_max, 5) #28

    for ia in range(len(angles)):
        an = angles[ia]
        xc = r * np.cos(an)
        yc = r * np.sin(an)
        hc_centers_x[idx] = xc
        hc_centers_y[idx] = yc
        hc_sigma_rep_field[idx] = sigmaximum / 10

        idx += 1
N_of_HC = idx

isc = range(N_of_HC + 1) # range(36) # [22,23,24,25,26,30,36,37,38,39]

##### Encoding ################################################################
for idx in range(NGHs):
    xc = hc_centers_x[idx]
    yc = hc_centers_y[idx]

    ##### Features ########################################################
    # freq_c[idx] = Freq + np.random.randn() * errs["freq_error"]
    # dir_c[idx] = Direction + np.random.randn() * errs["dir_error"]
    # xhc_ = xc * np.cos(dir_c[idx]) - yc * np.sin(dir_c[idx])
    # ph_c[idx] = 2 * np.pi * Freq * xhc_ + np.random.randn() * errs["ph_0_error"]
    # ampl_c[idx] = AmplitudesOfGratings(xc, yc, Sgm, cent_x=image_shift_x, cent_y=image_shift_y)
    # bgrd_c[idx] = Background(xc, yc)
    # grdX_c[idx] = (Background(xc, yc) - Background(xc - 0.01, yc)) / 0.01
    # grdY_c[idx] = (Background(xc, yc) - Background(xc, yc - 0.01)) / 0.01
    sgmGauss = sigmas[0]

    if True: #idx in isc:
        hc = HyperColumn(xc, yc, xx, yy, directions, sigmas, sgmGauss, frequencies=frequencies, params=params)  # 28


        ######################################
        Encoded = hc.encode(image)  
        ######################################

        freq_c[idx] = Encoded[0]['peak_freq']  # Freq                  + np.random.randn() * errs["freq_error"]
        dir_c[idx] = Encoded[0]['dominant_direction']  # Direction             + np.random.randn() * errs["dir_error"]
        xhc_ = xc * np.cos(dir_c[idx]) - yc * np.sin(dir_c[idx])
        ph_c[idx] = Encoded[0]['phase_0']  # 2*np.pi * Freq * xhc_  + np.random.randn() * errs["ph_0_error"]
        ampl_c[idx] = Encoded[0]['abs']  # AmplitudesOfGratings(xc, yc, Sgm, cent_x=0.1, cent_y=0.1)
        bgrd_c[idx] = Encoded[-1]['mean_intensity']  # Background(xc, yc)
        grdX_c[idx] = Encoded[-1]['Grad_x']
        grdY_c[idx] = Encoded[-1]['Grad_y']

        #print(idx,": dir_c=", dir_c[idx], "; freq_c=",freq_c[idx], "; ph_c=",ph_c[idx], "; ampl_c=",ampl_c[idx], "; drdX_c=",grdX_c[idx])
        print(idx,": bgrd_c=", bgrd_c[idx])

##### Decoding ################################################################
for idx in range(NGHs):

    xc = hc_centers_x[idx]
    yc = hc_centers_y[idx]

    ##########################################image_restored_by_HCs[:, :, idx] = decode(xx, yy, freq_c[idx], direction_c[idx], ph_c[idx], xc, yc)
    ###Freq_restored_by_HCs[:, :, idx]  = freq_c[idx]
    ###Dir_restored_by_HCs[:, :, idx]   = direction_c[idx]

    xx_ =  xx * np.cos(dir_c[idx]) - yy * np.sin(dir_c[idx])
    xhc_ = xc * np.cos(dir_c[idx]) - yc * np.sin(dir_c[idx])

    #### Each HC restores features across entire image
    Phi_restored_by_HCs[:, :, idx] = (ph_c[idx] + 2 * np.pi * (xx_ - xhc_) * freq_c[idx])
    Ampl_restored_by_HCs[:, :, idx] = ampl_c[idx]
    Grad_restored_by_HCs[:, :, idx] = grdX_c[idx] * (xx - xc) + grdY_c[idx] * (yy - yc)
    Bgrd_restored_by_HCs[:, :, idx] = bgrd_c[idx]

    #### Define RFs for HCs ###############################################
    receptive_field = np.exp(
        - 0.5 * ((yy - yc) / (0.5 * hc_sigma_rep_field[idx]))**2
        - 0.5 * ((xx - xc) / (0.5 * hc_sigma_rep_field[idx]))**2)
    # receptive_field = 1/((xx-xc)**2 + (yy-yc)**2)
    receptive_fields[:, :, idx] = receptive_field


#### Normalize RFs ############################################################
summ = np.sum(receptive_fields, axis=2)
summ[summ == 0] = 0.001
for i in range(NGHs):
    receptive_fields[:, :, i] /= summ
    #if i not in [isc]: 
        #receptive_fields[:, :, i] = 0

##########################################image_restored = np.sum(image_restored_by_HCs*receptive_fields, axis=2)
###Freq_restored  = np.sum(Freq_restored_by_HCs *receptive_fields, axis=2)
###Dir_restored   = np.sum(Dir_restored_by_HCs  *receptive_fields, axis=2)

##### Weight with RFs the contributions of HCs into features ##################
#Phi_restored = np.sum(Phi_restored_by_HCs * receptive_fields, axis=2)
#Ampl_restored = np.sum(Ampl_restored_by_HCs * receptive_fields, axis=2)
#Bgrd_restored = np.sum(Bgrd_restored_by_HCs * receptive_fields, axis=2)

##### Decode image from the fields of features ################################
#image_restored = np.cos(Phi_restored) * Ampl_restored + Bgrd_restored
#image_restored = np.sum((np.cos(Phi_restored_by_HCs[:, :, :]) * Ampl_restored_by_HCs[:, :, :] + Bgrd_restored_by_HCs[:, :, :]) * receptive_fields[:, :, :], axis=2)


image_restored_1 = np.sum( Bgrd_restored_by_HCs * receptive_fields, axis=2)
image_restored_2 = np.sum( (Bgrd_restored_by_HCs + Grad_restored_by_HCs) * receptive_fields, axis=2)
image_restored_3 = np.sum( (np.cos(Phi_restored_by_HCs) * 0.2*Ampl_restored_by_HCs + Bgrd_restored_by_HCs + Grad_restored_by_HCs) * receptive_fields, axis=2)

image_restored_list = [image_restored_1, image_restored_2, image_restored_3]

# image_restored = image_restored_by_HCs[:, :, 10]
# image_restored = receptive_fields[:, :, 30] + receptive_fields[:, :, 13]

metrics_template = "psnr = {:.4f}, ssim = {:.4f}"

##### Drawing #################################################################
fig, axes = plt.subplots(ncols=len(image_restored_list)+1, figsize=(40, 10), sharex=True, sharey=True)
axes[0].pcolor(xx[:, 0], yy[0, :], image, cmap='gray', shading='auto')

for ir_idx, image_restored in enumerate(image_restored_list):
    psnr = peak_signal_noise_ratio(image, image_restored)
    ssim = structural_similarity(image, image_restored)
    axes[ir_idx+1].pcolor(xx[:, 0], yy[0, :], image_restored, cmap='gray', shading='auto')
    axes[ir_idx+1].set_xlabel( metrics_template.format(psnr, ssim) )

for ax in axes:
    ax.scatter(hc_centers_x, hc_centers_y, s=2, color="green")
    ax.hlines([0, ], xmin=-1, xmax=1, color="blue")
    ax.vlines([0, ], ymin=-1, ymax=1, color="blue")

#axes[1].pcolor(xx,yy, receptive_fields[:, :, isc], cmap='gray', shading='auto')
# axes[1].scatter(hc_centers_x, hc_centers_y, s=15, color="red")
fig.savefig("./results/complex_result/accuracy.png")
plt.show()

