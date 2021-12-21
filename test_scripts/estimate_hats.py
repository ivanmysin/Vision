import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import sys
sys.path.append("../")
import pycwt.pycwt as pycwt


filename = "./results/mexican_hats/estimate_hats.png"


N = 900
dt = 1/N
freqs = np.fft.rfftfreq(N, dt)

wavelet = pycwt.mothers.MexicanHat()

sj = 1 / (wavelet.flambda() * freqs[1:])

ftfreqs = 2*np.pi*freqs
sj_col = sj[:, np.newaxis]


psi_ft_bar = ((sj_col * ftfreqs[1] * N)**.5 * np.conjugate(wavelet.psi_ft(sj_col * ftfreqs)))



fig, axes = plt.subplots()
for psi_idx, psi in enumerate(psi_ft_bar):
    psi = np.abs(psi)
    if psi_idx == 0:
        current_psi = psi
        
        max_freq = freqs[ np.argmax(psi) ]
        axes.plot(freqs, psi, label=max_freq)
        print(max_freq)
        continue
        
    
    psi_comp = np.sum(psi * current_psi )/ (np.sum(psi + current_psi ))
    if psi_comp < 0.1:
        max_freq = freqs[ np.argmax(psi) ]
        current_psi = psi
        axes.plot(freqs, psi, label=max_freq)
        print(max_freq)
        continue
        
axes.set_xlim(0, 450)
axes.set_ylim(0, None)
axes.legend()   
    
fig.savefig(filename, dpi=250)
plt.show()
