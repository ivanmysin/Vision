import numpy as np
import matplotlib.pyplot as plt
from elephant.signal_processing import wavelet_transform

filename = "./results/wavelets/algo1/high_sine.png"


t = np.linspace(0, 1, 900)
fs = 1 / (t[1] - t[0])
sigma_ends = 0.01
ends = np.ones_like(t)
cuted = t - t[0] - 5*sigma_ends
ends[cuted<0] *=  np.exp( -0.5*(cuted[cuted<0] /sigma_ends)**2 )
cuted = t - t[-1] + 5*sigma_ends
ends[cuted>0] *=  np.exp( -0.5*(cuted[cuted>0] /sigma_ends)**2 )


w = 400 # np.linspace(2, 25, t.size) # 15
u = np.cos(2*np.pi*w*t)
u *= ends



frequencies = np.arange(1, 449, 1)
wavelet_u = wavelet_transform(u, frequencies, fs=fs)

wave_u_abs = np.abs(wavelet_u) #*2/ u.size



fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(20, 10) )
axes[0].plot(t, u)
axes[0].set_xlim(0, 1)

axes[1].pcolor(t, frequencies, wave_u_abs)



for idx in range(30, u.size+30, 30):
    sl = np.s_[idx-30:idx]
    
    tcol = t[sl]
    tcol_c = tcol[tcol.size//2]

    t_col_centered = tcol - tcol_c
    
   
    wave_abs_cols = wave_u_abs[:, sl] 
    max_idx = np.unravel_index(np.argmax(wave_abs_cols * np.exp(-100000 * t_col_centered**2) ), wave_abs_cols.shape)
    # print( frequencies[max_idx[0]], t[sl][max_idx[1]] )
    
    axes[0].vlines(tcol[-1], -2, 2)
    axes[0].scatter( t[sl][max_idx[1]], [1, ], color="red", s=5 )
    
    freq_col = frequencies[max_idx[0]]
    
    amp_col = wave_abs_cols[max_idx[0], max_idx[1]]
    phase_col = np.angle( wavelet_u[:, sl][max_idx[0], max_idx[1]] )
    # print(amp_col)
    decoded = amp_col * np.cos(2 * np.pi * t_col_centered * freq_col + phase_col) 
    
    # decoded = wave_abs_cols * np.cos(2 * np.pi * tcol * freq_col + phase_col) 
    
    
    axes[2].plot(tcol, decoded)
    axes[2].vlines(tcol[-1], -2, 2)
    axes[2].scatter( t[sl][max_idx[1]], [1, ], color="red", s=5 )


fig.savefig(filename, dpi=250)
plt.show()
