import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.signal import hilbert

def get_gaussian_derivative(x, sigma, order=1):

    a = 1 / sigma**2
    if order == 1:
        kernel = np.exp(-0.5 * x**2 / sigma**2) * -1 *  x / sigma**2 / np.sqrt(2 * np.pi * sigma) 
    elif order == 2:
        kernel = 2 * np.exp(-a * x**2) * ( 1 - 2 * a * x**2)

    # kernel = zscore(kernel)
    return kernel

def get_signal():
    t = np.linspace(0, 1, 900)
    sigma_ends = 0.01
    ends = np.ones_like(t)
    cuted = t - t[0] - 5*sigma_ends
    ends[cuted<0] *=  np.exp( -0.5*(cuted[cuted<0] /sigma_ends)**2 )
    cuted = t - t[-1] + 5*sigma_ends
    ends[cuted>0] *=  np.exp( -0.5*(cuted[cuted>0] /sigma_ends)**2 )

    w = 5 #  np.linspace(2, 25, t.size)
    u = np.cos(2*np.pi*w*t) #+ 0.9*np.cos(2*np.pi*3*t)
    u *= ends
    
    return t, u

def myfft_convolve(signal, kernel, mode="same", isnormkernel=False):
    
    n1 = signal.size
    n2 = kernel.size
    N = n1 + n2 - 1


    x1 = np.append(signal, np.zeros(N - n1)  )
    x2 = np.append(kernel, np.zeros(N - n2)  )
    
    kernel_fft = np.fft.fft(x2)
    if isnormkernel:
        kernel_fft = kernel_fft / np.sqrt( np.mean(kernel_fft.real**2 + kernel_fft.imag**2) )
        pass
        
    conv = np.fft.ifft( np.fft.fft(x1) * kernel_fft ).real
    
    if mode == "same":
        Nres = n1 # max([n1, n2])
        Ndelited = int((N - Nres)//2)
        conv = conv[Ndelited : Ndelited+Nres]
    
    elif mode == "valid":
        nmax = max([n1, n2])
        nmin = min([n1, n2])
        Nres = nmax - nmin + 1
        Ndelited = int((N - Nres)//2)
        conv = conv[Ndelited : Ndelited+Nres]
        
    return conv



t, signal = get_signal()
dt = t[1] - t[0]
print(dt)
x = np.arange(-0.5, 0.5, dt) # np.linspace(-0.1, 0.1, 900)
sigma = 0.02
dg = get_gaussian_derivative(x, sigma, order=1)
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(x, dg)

# dg = dg / np.sqrt(np.sum(dg**2)) / 7.5 # * np.sqrt(2*sigma)
print(np.sum(dg**2))

sig_imag = myfft_convolve(signal, dg, mode="same")
# sig_imag = sig_imag / 7.5


res = np.sqrt(signal**2 + sig_imag**2)

theor_res = np.abs(hilbert(signal)  )

fig, ax = plt.subplots(ncols=1, nrows=2)
ax[0].plot(t, signal, color="green")
ax[0].plot(t, sig_imag, color="m")

ax[1].plot(t, res, color="blue")
ax[1].plot(t, theor_res, color="red")
plt.show()
