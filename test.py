import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt

def myfft_convolve(x1, x2, mode="full"):
    
    n1 = x1.size
    n2 = x2.size
    N = n1 + n2 - 1


    x1 = np.append(x1, np.zeros(N - n1)  )
    x2 = np.append(x2, np.zeros(N - n2)  )

    conv = np.fft.ifft( np.fft.fft(x1) * np.fft.fft(x2) ).real
    
    if mode == "same":
        Nres = max([n1, n2])
        Ndelited = int((N - Nres)//2)
        conv = conv[Ndelited : Ndelited+Nres]
    
    elif mode == "valid":
        nmax = max([n1, n2])
        nmin = min([n1, n2])
        Nres = nmax - nmin + 1
        Ndelited = int((N - Nres)//2)
        conv = conv[Ndelited : Ndelited+Nres]
        
    return conv

a = np.linspace(0.2, 0.5, 15) 
b = np.arange(1.0, 10.0, 0.5) 


c1 = np.convolve(a, b, mode="valid")
c2 = myfft_convolve(a, b, mode="valid")

print(c1)
print(c2)



"""
points = 10000

xvect = np.linspace(-10, 10, points)    #np.arange(0, points) - (points - 1.0) / 2
wp = 2 * np.pi * 0.03

dx = xvect[1] - xvect[0]
sigma = 0.5 # 1.0 / wp
a = 0.5 / sigma**2


kernel = np.sqrt(a / np.pi) * np.exp(-a * xvect**2)  * (-2.0 * a * xvect)

# kernel = hilbert(kernel).imag
# kernel = kernel / ( np.max(kernel) - np.min(kernel) )

# print(np.sum(kernel))
# print(np.sum(kernel**2))
kernel_fft = np.fft.fft(kernel)
kernel_fft = np.fft.fftshift(kernel_fft)
#kernel_fft = np.abs(kernel_fft) / np.sum(np.abs(kernel_fft) )
freqs = np.fft.fftshift( np.fft.fftfreq(xvect.size, xvect[1]-xvect[0]) )


theor_spec = freqs * 1j * np.exp(-(np.pi * freqs)**2 / a) * np.sqrt(np.pi / a) 
#theor_spec = np.abs(theor_spec) / np.sum(np.abs(theor_spec) )

plt.plot(freqs, kernel_fft.real, color="green", linewidth=5)
# plt.plot(freqs, kernel_fft.real)
# plt.plot(freqs, kernel_fft.imag)
plt.plot(freqs, theor_spec.real, color="blue")
# plt.plot(freqs, theor_spec.real)
# plt.plot(freqs, theor_spec.imag)


plt.figure()
plt.plot(xvect, kernel)



plt.show()
"""

"""
points = 100

xvect = np.arange(0, points) - (points - 1.0) / 2
wp = 2 * np.pi * 0.05

a = np.sqrt(2) / wp
A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
wsq = a**2
vec = np.arange(0, points) - (points - 1.0) / 2
xsq = vec**2
mod = (1 - xsq / wsq)
gauss = np.exp(-xsq / (2 * wsq))

ricker = A * mod * gauss

ricker_fft = 2 * np.abs( np.fft.rfft(ricker) ) / ricker.size

freqs = np.fft.rfftfreq(ricker.size, xvect[1]-xvect[0])
print( np.mean(np.abs(np.fft.fft(ricker)**2) ))
print( np.sum(ricker**2) )


wp = wp / 2 / np.pi
R = 2 * (freqs**2) / (np.sqrt(np.pi) * wp**3) * np.exp(-(freqs**2/wp**2) )
R = R / np.sum(R)
ricker_fft = ricker_fft / np.sum(ricker_fft)


plt.plot(freqs, ricker_fft)
plt.plot(freqs, R)
plt.show()
"""
