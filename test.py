import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt


x = np.array([1, 2, 3, 4])

y = np.fft.fft(x)

print(y)
"""
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
