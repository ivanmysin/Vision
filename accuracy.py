import numpy as np
import matplotlib.pyplot as plt

def get_gratings(freq, sigma=0.1, cent_x=0.2, cent_y=0.2, Len_x=200, Len_y=200, direction=2.3):
    xx, yy = np.meshgrid(np.linspace(-1, 1, Len_x), np.linspace(1, -1, Len_y))
    xx_rot = xx * np.cos(direction) - yy * np.sin(direction)
    image = np.cos(2*np.pi * xx_rot * freq) #0.5*(+1) #* np.exp( -0.5*((xx - cent_x)/sigma)**2 - 0.5*((yy - cent_y)/sigma)**2 )
    return image

def decode(xx, yy, freq, direction, phi_0, xc, yc):
    xx_  = xx * np.cos(direction) - yy * np.sin(direction)
    xhc_ = xc * np.cos(direction) - yc * np.sin(direction)
    image_restored = np.cos(2*np.pi * (xx_ - xhc_) * freq + phi_0)

    #coeff4delout = ( (xx - xc)**2 + (yy - yc)**2 ) / ( np.sum( (xx - xc)**2 + (yy - yc)**2) )
    #image_restored = image_restored / coeff4delout # !!!! ?????????? ?????????? ??? !!!!!

    return image_restored

######################################################

Freq = 5
Direction = 2.3
phi_0=0

errs = {
    "freq_error" : 1.5,      # Hz
    "dir_error" : 0.2,       # Rad
    "phi_0_error" :  1.5,    # Rad
}

##### Stimulus ################################################################
image = get_gratings(Freq, sigma=0.1, cent_x=0.1, cent_y=0.1, direction=Direction)
Len_y, Len_x = image.shape
xx, yy = np.meshgrid(np.linspace(-1, 1, Len_x), np.linspace(1, -1, Len_y))
delta_x = xx[0, 1] - xx[0, 0]
delta_y = yy[1, 0] - yy[0, 0]

radiuses = np.geomspace(0.05, 0.95, 15) # np.geomspace(0.1, 0.5, 10) #  #np.asarray([0.01, 0.05, 0.1]) #
angles = np.linspace(-np.pi, np.pi, 10, endpoint=True) # np.linspace(0, 0.5*np.pi, 10, endpoint=True) #
NGHs = int(radiuses.size*angles.size) # ????? ????????????

hc_centers_x = np.zeros(NGHs, dtype=np.float)
hc_centers_y = np.zeros(NGHs, dtype=np.float)
freq_c       = np.zeros(NGHs, dtype=np.float)
direction_c  = np.zeros(NGHs, dtype=np.float)
phi_c        = np.zeros(NGHs, dtype=np.float)
image_restored_by_HCs = np.zeros( (Len_y, Len_x, NGHs), dtype=np.float )
receptive_fields      = np.zeros( (Len_y, Len_x, NGHs), dtype=np.float )
Freq_restored_by_HCs  = np.zeros( (Len_y, Len_x, NGHs), dtype=np.float )
Dir_restored_by_HCs   = np.zeros( (Len_y, Len_x, NGHs), dtype=np.float )
Phi_restored_by_HCs   = np.zeros( (Len_y, Len_x, NGHs), dtype=np.float )

freq_teor_max = 0.5 / (np.sqrt(delta_x**2 + delta_y**2))
sigma_teor_min = 1 / (2*np.pi*freq_teor_max)

idx = 0
for r in radiuses:
    nsigmas = 8

    sigminimum = sigma_teor_min + 0.01*r #+ 0.2*r
    sigmaximum = 10 * sigminimum # ??????? ?? r

    for an in angles:
        xc = r * np.cos(an)
        yc = r * np.sin(an)

        hc_centers_x[idx] = xc
        hc_centers_y[idx] = yc

        # hc = HyperColomn([xc, yc], xx, yy, directions, sigmas, frequencies=frequencies, params=params)
        #Encoded = hc.encode(image)

        ##### Encoding ########################################################
        freq_c[idx]      = Freq                  + np.random.randn() * errs["freq_error"]
        direction_c[idx] = Direction             + np.random.randn() * errs["dir_error"]
        xhc_ = xc * np.cos(Direction) - yc * np.sin(Direction)
        phi_c[idx]      = 2*np.pi * Freq * xhc_  + np.random.randn() * errs["phi_0_error"]

        ##### Decoding ########################################################
        image_restored_by_HCs[:, :, idx] = decode(xx, yy, freq_c[idx], direction_c[idx], phi_c[idx], xc, yc)
        ###Freq_restored_by_HCs[:, :, idx]  = freq_c[idx]
        ###Dir_restored_by_HCs[:, :, idx]   = direction_c[idx]
        xx_  = xx * np.cos(direction_c[idx]) - yy * np.sin(direction_c[idx])
        xhc_ = xc * np.cos(direction_c[idx]) - yc * np.sin(direction_c[idx])
        Phi_restored_by_HCs[:, :, idx]   = phi_c[idx] + 2*np.pi * (xx_ - xhc_) * freq_c[idx]

        ##### Smoothing #######################################################
        sigma_rep_field = sigmaximum/1 # sigminimum
        # receptive_field = np.exp( -0.5*((yy - yc)/sigma_rep_field)**2 - 0.5*((xx - xc)/sigma_rep_field)**2 )
        receptive_field = 1/((xx-xc)**2 + (yy-yc)**2)
        receptive_fields[:, :, idx] = receptive_field
        idx += 1

summ = np.sum(receptive_fields, axis=2)
summ[summ == 0] = 0.001
for i in range(NGHs):
    receptive_fields[:, :, i] /= summ

# Phi_restored_by_HCs = Phi_restored_by_HCs%(2*np.pi)


# image_restored = np.sum(image_restored_by_HCs*receptive_fields, axis=2)
###Freq_restored  = np.sum(Freq_restored_by_HCs *receptive_fields, axis=2)
###Dir_restored   = np.sum(Dir_restored_by_HCs  *receptive_fields, axis=2)
Phi_restored   = np.sum(Phi_restored_by_HCs * receptive_fields, axis=2)

# Phi_restored   = np.angle( np.sum( receptive_fields*np.exp(1j * Phi_restored_by_HCs), axis=2) )

# sin_phi = np.sum( receptive_fields*np.sin(Phi_restored_by_HCs), axis=2)
# cos_phi = np.sum( receptive_fields*np.cos(Phi_restored_by_HCs), axis=2)
# Phi_restored = np.arctan2( sin_phi, cos_phi )

#image_restored = image_restored_by_HCs[:, :, 10]
#image_restored = receptive_fields[:, :, 30] + receptive_fields[:, :, 13]
image_restored = np.cos(Phi_restored)

fig, axes = plt.subplots(ncols=2, figsize=(30, 15), sharex=True, sharey=True)
axes[0].pcolor(xx[0, :], yy[:, 0], image, cmap='gray', shading='auto')
axes[1].pcolor(xx[0, :], yy[:, 0], image_restored, cmap='gray', shading='auto')
axes[1].scatter(hc_centers_x, hc_centers_y, s=5, color="red")
axes[1].hlines([0, ], xmin=-1, xmax=1, color="blue")
axes[1].vlines([0, ], ymin=-1, ymax=1, color="blue")

# fig.savefig("./results/accuracy_new.png")
plt.show()
