import numpy as np
import matplotlib.pyplot as plt

from skimage import color, data, filters

import sys
sys.path.append("../")

from HyperColumn2D_010722_ import HyperColumn


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

def Transform_with_Root(A):
    B = 0*A
    for i in range(200):
        for j in range(200):
            x1 = i/99-1
            y1 = j/99-1
            r1 = np.sqrt(x1**2 + y1**2)+0.00001
            cs = x1/r1
            sn = y1/r1
            if r1<1:
                r2 = r1**2
            else: 
                r2 = r1
            i2 = np.int(99*(r2*cs+1))
            j2 = np.int(99*(r2*sn+1))
            B[i,j] = A[i2,j2]
    return B
    
def Transform_with_Sqr(A):
    B = 0*image
    for i in range(200):    
        for j in range(200):
            x1 = i/99-1
            y1 = j/99-1
            r1 = np.sqrt(x1**2 + y1**2)+0.00001
            cs = x1/r1
            sn = y1/r1
            if r1<1:
                r2 = np.sqrt(r1)
            else:
                r2 = r1
            i2 = np.int(99*(r2*cs+1))
            j2 = np.int(99*(r2*sn+1))
            B[i,j] = A[i2,j2]
    return B

def Transform_X_with_Sqr(x1,y1):
    r1 = np.sqrt(x1**2 + y1**2)+0.00001
    cs = x1/r1
    if r1<1:
        r2 = r1**2
    else:
        r2 = r1
    x2 = r2*cs
    return x2

def Transform_Y_with_Sqr(x1,y1):
    r1 = np.sqrt(x1**2 + y1**2)+0.00001
    sn = y1/r1
    if r1<1:
        r2 = r1**2
    else:
        r2 = r1
    y2 = r2*sn
    return y2

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
yy, xx = np.meshgrid(np.linspace(-1, 1, Ny), np.linspace(-1, 1, Nx))
################
image = Gratings(Freq, xx, yy, sigma=Sgm, cent_x=image_shift_x, cent_y=image_shift_y, direction=Direction)
################

image = data.camera() # color.rgb2gray(data.astronaut())
image = image[::-2, ::2]
image = image[56:, 46:246]
image = np.transpose(image)
image = image.astype(np.float64)

image = (image - 0.5*np.min(image) - 0.5*np.max(image))/(np.max(image)-np.min(image))*2
image[199,199] = np.max(image)
image[0,  199] = np.min(image)

# sguare-root tranformation of image
image_sqroot = Transform_with_Root(image)
image_sqrootback = Transform_with_Sqr(image_sqroot)

fig, axes = plt.subplots(ncols=3, figsize=(30, 15), sharex=True, sharey=True)
axes[0].pcolor(xx,yy, image,            cmap='gray')#, shading='auto')
axes[1].pcolor(xx,yy, image_sqroot,     cmap='gray')#, shading='auto')
axes[2].pcolor(xx,yy, image_sqrootback, cmap='gray')#, shading='auto')
fig.savefig("iii.png")
plt.show()

delta_x = xx[1, 0] - xx[0, 0]
delta_y = yy[0, 1] - yy[0, 0]

params = {
    "use_circ_regression": False,
}
Nxy = 30 # number for either the grid in polar coordinates or the square grid
NGHs = Nxy*Nxy # Number of hypercolumns

hc_centers_x = np.zeros(NGHs, dtype=np.float64)
hc_centers_y = np.zeros(NGHs, dtype=np.float64)
xxx_ = np.zeros(NGHs, dtype=np.float64)
yyy_ = np.zeros(NGHs, dtype=np.float64)
hc_sigma_rep_field = np.zeros(NGHs, dtype=np.float64)
freq_c = np.zeros(NGHs, dtype=np.float64)
dir_c = np.zeros(NGHs, dtype=np.float64)
ph_c = np.zeros(NGHs, dtype=np.float64)
ampl_c = np.zeros(NGHs, dtype=np.float64)
bgrd_c = np.zeros(NGHs, dtype=np.float64)
grdX_c = np.zeros(NGHs, dtype=np.float64)
grdY_c = np.zeros(NGHs, dtype=np.float64)
aBreak_c = np.zeros(NGHs, dtype=np.float64)
xBreak_c = np.zeros(NGHs, dtype=np.float64)
image_restored_by_HCs = np.zeros((Nx, Ny, NGHs), dtype=np.float64)
receptive_fields = np.zeros((Nx, Ny, NGHs), dtype=np.float64)
Freq_restored_by_HCs = np.zeros((Nx, Ny, NGHs), dtype=np.float64)
Dir_restored_by_HCs = np.zeros((Nx, Ny, NGHs), dtype=np.float64)
Phi_restored_by_HCs = np.zeros((Nx, Ny, NGHs), dtype=np.float64)
Ampl_restored_by_HCs = np.zeros((Nx, Ny, NGHs), dtype=np.float64)
Bgrd_restored_by_HCs = np.zeros((Nx, Ny, NGHs), dtype=np.float64)
Grad_restored_by_HCs = np.zeros((Nx, Ny, NGHs), dtype=np.float64)
Break_restored_by_HCs = np.zeros((Nx, Ny, NGHs), dtype=np.float64) # 14.07.2022

directions = np.linspace(-np.pi, np.pi, 32, endpoint=False)  # 03.07.2022 | 32

##### Prepare geometry of HCs #################################################

### Square mech ###
sgmGauss = 1/Nxy ##0.2  # 01.07.2022: 0.1
sigmas = np.geomspace(0.01, 0.2, 4)  # 28
freqMH = 2 #uencies = np.geomspace(0.1, 10, 5) ## np.asarray([3, ]) # 03.07.2022 |

idx = 0
for i in range(Nxy):
    for j in range(Nxy):
        hc_centers_x[idx] = -1 + 2.0/Nxy*(i+1/2)
        hc_centers_y[idx] = -1 + 2.0/Nxy*(j+1/2)
        hc_sigma_rep_field[idx] = 0.5/Nxy ## 0.05 # 01.07.2022 | 1/Nxy
        idx += 1
N_of_HC = idx

### Selection ###
isc = range(N_of_HC)#[28,29,33,40]#range(N_of_HC) #[22,23,24,25,26,30,36,37,38,39]

##### Encoding ################################################################
for idx in range(N_of_HC):
    xc = hc_centers_x[idx]
    yc = hc_centers_y[idx]

    ##### Features ########################################################
    freq_c[idx] = Freq + np.random.randn() * errs["freq_error"]
    dir_c[idx] = Direction + np.random.randn() * errs["dir_error"]
    xhc_ = xc * np.cos(dir_c[idx]) - yc * np.sin(dir_c[idx])
    ph_c[idx] = 2 * np.pi * Freq * xhc_ + np.random.randn() * errs["ph_0_error"]
    ampl_c[idx] = 0#AmplitudesOfGratings(xc, yc, Sgm, cent_x=image_shift_x, cent_y=image_shift_y)
    bgrd_c[idx] = 0#Background(xc, yc)
    grdX_c[idx] = 0#(Background(xc, yc) - Background(xc - 0.01, yc)) / 0.01
    grdY_c[idx] = 0#(Background(xc, yc) - Background(xc, yc - 0.01)) / 0.01
    aBreak_c[idx] = 0# 14.07.2022

    if idx in isc:
        hc = HyperColumn(xc, yc, xx, yy, directions, sigmas, sgmGauss, freqMH, params=params)  # 28
        ######################################
        Encoded = hc.encode(image_sqroot)  
        ######################################

        freq_c[idx] = Encoded['peak_freq']  # Freq                  + np.random.randn() * errs["freq_error"]
        dir_c[idx] = Encoded['dominant_direction']  # Direction             + np.random.randn() * errs["dir_error"]
        #xhc_ = xc * np.cos(dir_c[idx]) - yc * np.sin(dir_c[idx])
        ph_c[idx] = Encoded['phase_0']  # 2*np.pi * Freq * xhc_  + np.random.randn() * errs["ph_0_error"]
        ampl_c[idx] = Encoded['abs']  # AmplitudesOfGratings(xc, yc, Sgm, cent_x=0.1, cent_y=0.1)
        bgrd_c[idx] = Encoded['mean_intensity']  # Background(xc, yc)
        grdX_c[idx] = Encoded['Grad_x']
        grdY_c[idx] = Encoded['Grad_y']
        aBreak_c[idx] = Encoded['aBreak'] # 14.07.2022
        xBreak_c[idx] = Encoded['xBreak'] # 14.07.2022

        print(idx,": bgrd=",f"{ bgrd_c[idx] :.2f}", "; dir=", f"{ dir_c[idx]/np.pi*180 :.0f}", "; xBreak=",f"{ xBreak_c[idx] :.2f}", "; aBreak=",f"{ aBreak_c[idx] :.2f}") #, "; freq=",f"{ freq_c[idx] :.2f}", "; ph=",f"{ ph_c[idx] :.3f}", "; ampl=",f"{ ampl_c[idx] :.3f}", "; drdX=",f"{ grdX_c[idx] :.2f}", "; drdY=",f"{ grdY_c[idx] :.2f}")

##### Decoding ################################################################
for idx in range(N_of_HC):

    xc = hc_centers_x[idx]
    yc = hc_centers_y[idx]

    xx_ =  xx * np.cos(dir_c[idx]) - yy * np.sin(dir_c[idx])
    xhc_ = xc * np.cos(dir_c[idx]) - yc * np.sin(dir_c[idx])

    #### Each HC restores features across entire image
    Phi_restored_by_HCs[:, :, idx] =  ph_c[idx] + 2 * np.pi * (xx_ - xhc_) * freq_c[idx]
    Ampl_restored_by_HCs[:, :, idx] = ampl_c[idx]
    Bgrd_restored_by_HCs[:, :, idx] = bgrd_c[idx]
    Grad_restored_by_HCs[:, :, idx] = bgrd_c[idx] + grdX_c[idx] * (xx - xc) + grdY_c[idx] * (yy - yc)
    Break_restored_by_HCs[:, :, idx] = (0.5-np.heaviside(xx_-xhc_-xBreak_c[idx],0.5))*aBreak_c[idx] # 14.07.2022

    #### Define RFs for HCs ###############################################
    receptive_field = np.exp(
        - 0.5 * ((yy - yc) / hc_sigma_rep_field[idx]) ** 2 
        - 0.5 * ((xx - xc) / hc_sigma_rep_field[idx]) ** 2)
    # receptive_field = 1/((xx-xc)**2 + (yy-yc)**2)
    receptive_fields[:, :, idx] = receptive_field
    #for i in range(Nx):
    #    for j in range(Ny):
    #        receptive_fields[i, j, idx] = 0
    #        if ((np.abs(xx[i,j]-xc)<0.95/Nxy)and(np.abs(yy[i,j]-yc)<0.95/Nxy)):
    #            receptive_fields[i, j, idx] = 1
                


#### Normalize RFs ############################################################
summ = np.sum(receptive_fields, axis=2)
summ[summ == 0] = 0.001
for i in range(NGHs):
    receptive_fields[:, :, i] /= summ


##### Decode image from the fields of features ################################
for i in range(NGHs):
    ###########################################################################
    image_restored_by_HCs[:,:,i] = Bgrd_restored_by_HCs[:,:,i] + Break_restored_by_HCs[:,:,i] # 14.07.2022
    # 14.07.2022: image_restored_by_HCs[:,:,i] = Grad_restored_by_HCs[:,:,i] + np.cos(Phi_restored_by_HCs[:,:,i]) * Ampl_restored_by_HCs[:,:,i] ## * 0.74  # 03.07.2022 | it was Bgrd instead of Grad
    ###########################################################################
    
image_restored_Bg = np.sum(  Bgrd_restored_by_HCs[:, :, :] * receptive_fields[:, :, :], axis=2) 
image_restored_Gr = np.sum(  Grad_restored_by_HCs[:, :, :] * receptive_fields[:, :, :], axis=2) 
image_restored    = np.sum( image_restored_by_HCs[:, :, :] * receptive_fields[:, :, :], axis=2) 

for i in range(200) :
    for j in range(200):
        image_restored[i,j] = np.min([image_restored[i,j], np.max(image)])
        image_restored[i,j] = np.max([image_restored[i,j], np.min(image)])
        image_restored_Bg[i,j] = np.min([image_restored_Bg[i,j], np.max(image)])
        image_restored_Bg[i,j] = np.max([image_restored_Bg[i,j], np.min(image)])
        image_restored_Gr[i,j] = np.min([image_restored_Gr[i,j], np.max(image)])
        image_restored_Gr[i,j] = np.max([image_restored_Gr[i,j], np.min(image)])
image_restored[195:199,195:199] = np.max(image)
image_restored[0:10,  195:199] = np.min(image)
image_restored_Bg[195:199,195:199] = np.max(image)
image_restored_Bg[0:10,  195:199] = np.min(image)
image_restored_Gr[195:199,195:199] = np.max(image)
image_restored_Gr[0:10,  195:199] = np.min(image)

##### Drawing #################################################################
fig, axes = plt.subplots(ncols=2, figsize=(30, 15), sharex=True, sharey=True)
axes[0].pcolor(xx,yy, image, cmap='gray')#, shading='auto')
axes[1].pcolor(xx,yy, Transform_with_Sqr(image_restored), cmap='gray')#, shading='auto')
#axes[2].pcolor(xx,yy, receptive_fields[:, :, 8], cmap='gray')

#axes[0].scatter(hc_centers_x, hc_centers_y, s=150, color="red")
#axes[1].scatter(hc_centers_x, hc_centers_y, s=150, color="red")
for i in isc:
    xxx_[i] = Transform_X_with_Sqr(hc_centers_x[i], hc_centers_y[i])
    yyy_[i] = Transform_Y_with_Sqr(hc_centers_x[i], hc_centers_y[i])
#axes[1].scatter(hc_centers_x[isc], hc_centers_y[isc], s=100, color="green")
axes[0].scatter(xxx_[isc], yyy_[isc],                 s=100, color="green")
axes[1].scatter(xxx_[isc], yyy_[isc],                 s=50, color="green")
axes[1].hlines([0, ], xmin=-1, xmax=1, color="blue")
axes[1].vlines([0, ], ymin=-1, ymax=1, color="blue")
axes[0].hlines([0, ], xmin=-1, xmax=1, color="blue")
axes[0].vlines([0, ], ymin=-1, ymax=1, color="blue")
fig.savefig("aaa.png")
plt.show()

fig, axes = plt.subplots(ncols=2, figsize=(30, 15), sharex=True, sharey=True)
axes[0].pcolor(xx,yy, Transform_with_Sqr(image_restored_Bg), cmap='gray')#, shading='auto')
axes[1].pcolor(xx,yy, Transform_with_Sqr(image_restored_Gr), cmap='gray')#, shading='auto')
axes[0].scatter(xxx_[isc], yyy_[isc], s=50, color="green")
axes[1].scatter(xxx_[isc], yyy_[isc], s=50, color="green")
axes[1].hlines([0, ], xmin=-1, xmax=1, color="blue")
axes[1].vlines([0, ], ymin=-1, ymax=1, color="blue")
axes[0].hlines([0, ], xmin=-1, xmax=1, color="blue")
axes[0].vlines([0, ], ymin=-1, ymax=1, color="blue")
fig.savefig("bbb.png")
plt.show()