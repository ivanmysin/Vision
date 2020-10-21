import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../")
import pycwt.pycwt as pycwt


t = np.linspace(0, 1, 900)
dt = (t[1] - t[0])

sigma_ends = 0.01
ends = np.ones_like(t)
cuted = t - t[0] - 5*sigma_ends
ends[cuted<0] *=  np.exp( -0.5*(cuted[cuted<0] /sigma_ends)**2 )
cuted = t - t[-1] + 5*sigma_ends
ends[cuted>0] *=  np.exp( -0.5*(cuted[cuted>0] /sigma_ends)**2 )

u = np.cos(2*np.pi*15*t)
u *= ends

frequencies = np.arange(1, 51, 1)
wavelet_u_list = pycwt.cwt(u, dt, freqs=frequencies)

wavelet_u = wavelet_u_list[0]

wave_u_abs = np.abs(wavelet_u) #*2/ u.size



fig, axes = plt.subplots(nrows=4, sharex=True)
axes[0].plot(t, u)
axes[0].set_xlim(0, 1)

axes[1].pcolor(t, frequencies, wave_u_abs)



for idx in range(30, u.size+30, 30):
    sl = np.s_[idx-30:idx]
    
    tcol = t[sl]
    tcol_c = tcol[tcol.size//2]

    t_col_centered = tcol - tcol_c
    
   
    wave_abs_cols = wave_u_abs[:, sl] * np.exp(-1000 * t_col_centered**2)
    max_idx = np.unravel_index(np.argmax(wave_abs_cols), wave_abs_cols.shape)
    # print( frequencies[max_idx[0]], t[sl][max_idx[1]] )
    
    axes[0].vlines(tcol[-1], -2, 2)
    axes[0].scatter( t[sl][max_idx[1]], [1, ], color="red", s=5 )
    
    # freq_col = frequencies[max_idx[0]]
    
 
    wavelet_u[:max_idx[0], sl] = 0 + 0*1j
    wavelet_u[max_idx[0]+1:, sl] = 0 + 0*1j

    
    wavelet_u[max_idx[0], sl] = wavelet_u[max_idx[0], sl] / wave_u_abs[max_idx[0], sl] * wave_abs_cols[max_idx[0], max_idx[1]]
    
    
    
    axes[3].vlines(tcol[-1], -2, 2)
    axes[3].scatter( t[sl][max_idx[1]], [1, ], color="red", s=5 )

wave_u_abs = np.abs(wavelet_u) #*2/ u.size
axes[2].pcolor(t, frequencies, wave_u_abs)

decode = pycwt.icwt(wavelet_u, wavelet_u_list[1], dt)
axes[3].plot(t, decode)



plt.show()
