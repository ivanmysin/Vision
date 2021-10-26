import numpy as np
import matplotlib.pyplot as plt
pltparams = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 18,
         'axes.titlesize': 18,
         'xtick.labelsize': 14,
         'ytick.labelsize': 14,
         }
plt.rcParams.update(pltparams)


sigmas = [0.02, 0.2]#  0.05, 0.1, 0.2]

fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(18, 9))
for sigma in sigmas:

    t = np.linspace(0, 1, 900)
    dt = t[1] - t[0]

    t = np.arange(-0.5, 0.5, dt)

    a = 1 / sigma**2


    freqs = np.fft.fftshift( np.fft.fftfreq(t.size, dt) )


    kernel = np.exp(-a * t**2) * -2 * a * t


    theor_spec = np.abs( freqs * 1j * np.exp(-(np.pi * freqs)**2 / a) * np.sqrt(np.pi / a) )
    theor_spec = theor_spec / np.sum(theor_spec)


    fft_spec = np.fft.fftshift( np.abs(np.fft.fft(kernel) ) ) # / kernel.size 3.2
    fft_spec = fft_spec / np.sum(fft_spec)




    ax[0, 0].plot(t, kernel, label=r"$\sigma="+str(sigma)+"$") #
    ax[0, 0].set_title("Производная гауссианы", fontsize=22)
    #ax[0, 1].plot(freqs, theor_spec, color="blue", label="theor", linewidth=2)
    ax[0, 1].plot(freqs, theor_spec, label=r"$\sigma="+str(sigma)+"$", linewidth=1)
    # ax[0, 1].plot(freqs, fft_spec, color="green", label="fft")
    ax[0, 1].set_xlim(0, 100)
    ax[0, 1].set_xlabel("Частота, Гц")

    ax[0, 1].legend()
    ax[0, 0].legend()


    kernel = 2 * np.exp(-a * t**2) * ( 1 - 2 * a * t**2)
    theor_spec = np.abs( -freqs**2 * np.exp(-(np.pi * freqs)**2 / a) * np.sqrt(np.pi / a) )
    theor_spec = theor_spec / np.sum(theor_spec)
    fft_spec = np.fft.fftshift( np.abs(np.fft.fft(kernel) ) )
    fft_spec = fft_spec / np.sum(fft_spec)

    # fig, ax = plt.subplots(ncols=2)

    ax[1, 0].plot(t, kernel, label=r"$\sigma="+str(sigma)+"$")
    ax[1, 0].set_title("Вторая производная гауссианы", fontsize=22)
    ax[1, 1].plot(freqs, theor_spec, label=r"$\sigma="+str(sigma)+"$", linewidth=1) # color="blue",
    # ax[1, 1].plot(freqs, fft_spec, color="green", label="fft")
    ax[1, 1].set_xlim(0, 100)
    ax[1, 1].set_xlabel("Частота, Гц")
    #
    ax[1, 1].legend()
    ax[1, 0].legend()

ax[0, 1].set_ylim(0, None)
ax[1, 1].set_ylim(0, None)

# sigma1 = 0.05
# sigma2 = 0.08
# a1 = 0.5/sigma1**2
# a2 = 0.5/sigma2**2
#
# kernel = np.exp(-0.5 * (t/sigma1)**2)/sigma1 - np.exp(-0.5 * (t/sigma2)**2)/sigma2
# theor_spec = np.abs( np.exp(-(np.pi * freqs)**2 / a1) * np.sqrt(np.pi / a1)/sigma1 -  np.exp(-(np.pi * freqs)**2 / a2) * np.sqrt(np.pi / a2)/sigma2 )
# theor_spec = theor_spec / np.sum(theor_spec)
# fft_spec = np.fft.fftshift( np.abs(np.fft.fft(kernel) ) )
# fft_spec = fft_spec / np.sum(fft_spec)
#
# # fig, ax = plt.subplots(ncols=2)
#
# ax[2, 0].plot(t, kernel)
# ax[2, 0].set_title("Разность гауссиан")
# ax[2, 1].plot(freqs, theor_spec, color="blue", label="theor", linewidth=2)
# ax[2, 1].plot(freqs, fft_spec, color="green", label="fft")
# ax[2, 1].set_xlim(0, None)
# ax[2, 1].set_ylim(0, None)
# ax[2, 1].legend()

fig.savefig("./results/kernel_spectrums.png", dpi=500)
plt.show()
