import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import sys
sys.path.append("../")
import pycwt.pycwt as pycwt

filename = "./results/mexican_hats/fixed_frequencies_and_times/demo_sine.png"

t = np.linspace(0, 1, 900)
dt = (t[1] - t[0])

sigma_ends = 0.01
ends = np.ones_like(t)
cuted = t - t[0] - 5*sigma_ends
ends[cuted<0] *=  np.exp( -0.5*(cuted[cuted<0] /sigma_ends)**2 )
cuted = t - t[-1] + 5*sigma_ends
ends[cuted>0] *=  np.exp( -0.5*(cuted[cuted>0] /sigma_ends)**2 )

w = 100 # np.linspace(2, 25, t.size) # 15 # 15 #
u = np.cos(2*np.pi*w*t) # + 0.9*np.cos(2*np.pi*5*t)
u *= ends


# 1.0, 9.0, 53.0, 225.0
frequencies = np.arange(1, 250, 1) # np.array([1.0, 9.0, 53.0, 255.0]) # np.geomspace(1, 450, num=8)
wavelet_u_list = pycwt.cwt(u, dt, freqs=frequencies, wavelet='mexicanhat') 


wavelet_u = np.copy(wavelet_u_list[0])
wave_u_abs = np.abs(wavelet_u) # wavelet_u.real #
wavelet_u_restored = np.zeros_like(wavelet_u)

fig, axes = plt.subplots(nrows=4, sharex=True, figsize=(15, 15) )
axes[0].plot(t, u)

tcentsd = t[15::30]
axes[0].scatter(tcentsd, np.zeros_like(tcentsd), color="red", s=15)
axes[0].set_xlim(0, 1)
axes[0].set_title("Сигнал (10 Гц)")


axes[1].pcolor(t, frequencies, wave_u_abs, shading='auto')
axes[1].scatter(tcentsd, np.zeros_like(tcentsd)+10, color="red", s=15)
axes[1].set_ylabel("Частота, Гц")
axes[1].set_title("Вейвлет спектр по мексиканским шляпам")


# wavelet_u = hilbert( wavelet_u_list[0].real, axis=1)
"""
wavelet_u = hilbert( wavelet_u_list[0].real, axis=1)
axes[2].pcolor(t, frequencies, np.abs(wavelet_u), shading='auto')
axes[2].scatter(tcentsd, np.zeros_like(tcentsd)+10, color="red", s=15)
axes[2].set_ylabel("Частота, Гц")
axes[2].set_title("Амплитуда клмплексного вейвлет спектра по мексиканским шляпам")

wavelet_u_list[0].imag = 0
wavelet_u_list[0].real = np.abs(wavelet_u) * np.cos(np.angle(wavelet_u))
wave_u_abs = np.abs(wavelet_u_list[0].real)
axes[3].pcolor(t, frequencies, wave_u_abs, shading='auto')
axes[3].scatter(tcentsd, np.zeros_like(tcentsd)+10, color="red", s=15)
axes[3].set_ylabel("Частота, Гц")
axes[3].set_title("Вейвлет спектр по мексиканским шляпам")
"""


for idx in range(30, u.size+30, 30):
    sl = np.s_[idx-30:idx]
    
    tcol = t[sl]
    max_idx = tcol.size//2
    t_col_centered = tcol - tcol[max_idx]
    wave_u_col = wavelet_u[:, sl]
    wave_abs_cols = wave_u_abs[:, sl] 
    
    # print( frequencies[max_idx[0]], t[sl][max_idx[1]] )
    
    axes[0].vlines(tcol[-1], -2, 2, color="black")
    # axes[0].scatter( tcol[max_idx], [1.2, ], color="red", s=5 )
    
   
    tcol_max = tcol[max_idx]
    for freq_idx, freq_col in enumerate(frequencies):
        
        phi_col_cent = np.angle(wave_u_col[freq_idx, max_idx])
        # len_col_cent = wave_abs_cols[freq_idx, max_idx]
        len_col_cent = np.mean( wave_abs_cols[freq_idx] )
        # print(len_col_cent)

        # print( np.sum (np.cos( np.angle(wave_u_col[freq_idx, :] ) - freq_col*2*np.pi*(tcol-tcol_max) + phi_col_cent) ) ) 
        # print("#############################################")
        
        phases = freq_col*2*np.pi*(tcol-tcol_max) + phi_col_cent
        # wavelet_u[freq_idx, sl] = len_col_cent * (np.cos(phases) + 1j * np.sin(phases))
        wavelet_u_restored[freq_idx, sl] = len_col_cent * np.cos(phases) # (tcol-tcol_max)
        # np.exp(1j*(freq_col*2*np.pi*(tcol-tcol_max) + phi_col_cent))

    axes[3].vlines(tcol[-1], -2, 2, color="black")
    axes[3].scatter( tcol[max_idx], [1.2, ], color="red", s=5 )

wave_u_abs = wavelet_u_restored.real
axes[2].pcolor(t, frequencies, wave_u_abs, shading='auto')
axes[2].set_ylabel("Frequency, Hz")

# axes[2].plot(t, wavelet_u.real[frequencies==100, :].ravel(), label="restored")
# axes[3].plot(t, wavelet_u_list[0].real[frequencies==100, :].ravel(), label="sourse")
# axes[3].plot(t, wave_u_abs[frequencies==100, :].ravel(), label="sourse")
#
# axes[2].legend()
# axes[3].legend()
# wavelet_u = np.copy(wavelet_u_list[0])
# wavelet_u.imag = 0
decode = pycwt.icwt(wavelet_u_restored, wavelet_u_list[1], dt, wavelet='mexicanhat')
# decode = np.abs(decode) # * np.cos(np.angle(decode))
axes[3].plot(t, decode)


# fig.savefig(filename, dpi=250)
plt.show()
