import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import sys
sys.path.append("../")
import pycwt.pycwt as pycwt
from hilb_via_dgs import HilbertByGaussianDerivative
filename = "./results/demo_sine_2p.png"
file_saving_ws = "./results/weights_for_gaussian_derivatives_sum.npy"
###                        Исходный сигнал
t = np.linspace(0, 1, 900)
dt = (t[1] - t[0])
N = t.size

sigma_ends = 0.01
ends = np.ones_like(t)
cuted = t - t[0] - 5*sigma_ends
ends[cuted<0] *=  np.exp( -0.5*(cuted[cuted<0] /sigma_ends)**2 )
cuted = t - t[-1] + 5*sigma_ends
ends[cuted>0] *=  np.exp( -0.5*(cuted[cuted>0] /sigma_ends)**2 )

w = 47 # np.linspace(2, 25, t.size) # 15 # 15 #
u =  10*np.cos(2*np.pi*w*t*t) + 7*np.cos(2*np.pi*25.6*t) + 3*np.cos(2*np.pi*5*t)
u *= ends


myHilbert = HilbertByGaussianDerivative(30, file_saving_ws, nsigmas=8, isplotarox=False)

u = u + 1j*myHilbert.hilbert(u)
###                        Кодирование: вейвлет-преобразование сигнала
# 1.0, 9.0, 53.0, 225.0
#frequencies = np.arange(0.1, 200, 20) # np.array([1.0, 9.0, 53.0, 255.0]) # np.geomspace(1, 450, num=8)
frequencies = np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4])
#frequencies = np.array([0.1, 0.4, 1.6, 6.4, 25.6, 102.4])
####################################################################################
wave, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(u, dt, freqs=frequencies, wavelet=#'morlet')
                                                                                      'mexicanhat') 
####################################################################################

###                        Кодирование: достраивание мнимой части сигнала
##########################################
# Hilbert_wave = hilbert(wave.real, axis=1)
# wave.imag = Hilbert_wave.imag
##########################################

###                        Кодирование и раскодирование для каждой ГК
nPW = u.size//30                        # число ГК
phi_col_cent = np.zeros((scales.size, nPW)) # to be memorized in ГК
len_col_cent = np.zeros((scales.size, nPW)) # to be memorized in ГК
peak_freq    = np.zeros((scales.size, nPW)) # to be memorized in ГК
phi_col_edge = np.zeros((scales.size, nPW)) # to be memorized in ГК

for iPW in range(1, nPW, 1):            # номер ГК
    idx = iPW*30                        # индекс правой границы ГК
    print('номер ГК', iPW)
    sl = np.s_[idx-30:idx]              # индексы внутри ГК
    max_idx = t[sl].size//2             # локальный индекс центра ГК
    
    tcol = t[sl]                        # координата внутри ГК
    #wave_u_col = wave[:, sl]                # вейвлет-преобразование в лок. координатах
    #wave_abs_cols = np.abs(wave[:, sl])     # амплитуда аналитического сигнала в лок. координатах
    

    # axes[0].scatter( tcol[max_idx], [1.2, ], color="red", s=5 )
   
    tcol_max = tcol[max_idx]    # локальная координата центра ГК 
### Кодирование: редукция - запоминание фаз и амплитуд для каждой частоты и номера главной частоты
    for freq_idx, freq_col in enumerate(frequencies): # индекс частоты и частота 
        
        phi_col_cent[freq_idx, iPW] = np.angle(wave[freq_idx, sl][max_idx+1])  # фаза в центре ГК
        len_col_cent[freq_idx, iPW] = np.mean(np.abs(wave[freq_idx, sl]))    # среднее по ГК
        phi_col_edge[freq_idx, iPW] = np.angle(wave[freq_idx, sl][max_idx-1])        # фаза на левой границе ГК
        
        ### Находим частоту колебаний разложения по вэйвлетам одного из масштабов
        #x=wave[freq_idx, sl]
        #sp = np.fft.fft(x)
        ##sp = pycwt.cwt(x, dt, freqs=frequencies, wavelet='mexicanhat')
        #freq = np.linspace(0, 1 / (tcol[1] - tcol[0]), t[sl].size)
        #peak_freq[freq_idx, iPW] = freq[np.argmax( np.abs(sp) )]

        phase_diff = phi_col_cent[freq_idx, iPW] - phi_col_edge[freq_idx, iPW]
        if phase_diff > np.pi: phase_diff -= 2*np.pi
        if phase_diff < -np.pi: phase_diff += 2*np.pi
        peak_freq[freq_idx, iPW] = phase_diff  / (tcol[max_idx+1] - tcol[max_idx-1]) / 2 / np.pi
        #print(freq_idx, peak_freq[freq_idx, iPW])
        
    #imain_len_col_cent = np.argmax(len_col_cent)                        # номер главной частоты
    #print(frequencies[imain_len_col_cent])
    
#################################################################################################    
### Раскодирование
wavelet_u_restored = np.zeros_like(wave)
for iPW in range(1, nPW, 1):     # номер ГК
    idx = iPW*30                        # индекс правой границы ГК
    sl = np.s_[idx-30:idx]              # индексы внутри ГК
    for freq_idx, freq_col in enumerate(frequencies): # индекс частоты и частота 
        #phases = frequencies[min(freq_idx, imain_len_col_cent)]*2*np.pi*(t[sl]-tcol_max) + phi_col_cent[freq_idx] # фаза вдоль ГК
        phases = peak_freq[freq_idx, iPW]*2*np.pi*(t[sl]-t[sl][t[sl].size//2]) + phi_col_cent[freq_idx, iPW] # фаза вдоль ГК
        phases_true=np.angle(wave[freq_idx, sl])
        #print(phases[max_idx-1],phases[max_idx],phases_true[max_idx-1],phases_true[max_idx],freq_col)
        #if freq_idx != imain_len_col_cent: len_col_cent[freq_idx] = 0
        wavelet_u_restored[freq_idx, sl].real = len_col_cent[freq_idx, iPW] * np.cos(phases) 
        wavelet_u_restored[freq_idx, sl].imag = len_col_cent[freq_idx, iPW] * np.sin(phases)


###############################################################################
decode = pycwt.icwt(wavelet_u_restored, scales, dt, wavelet='mexicanhat')

# decode = np.abs(decode) # * np.cos(np.angle(decode))
decode2 = pycwt.icwt(wave, scales, dt, wavelet='mexicanhat')

###############################################################################


fig, axes = plt.subplots(nrows=4, sharex=True, figsize=(15, 15) )
axes[0].plot(t, u.real)

tcentsd = t[15::30]
axes[0].scatter(tcentsd, np.zeros_like(tcentsd), color="red", s=15)
axes[0].set_xlim(0, 1)
axes[0].set_title("Сигнал (10 Гц)")
axes[0].vlines(t[30::30], -2, 2, color="black")



c = axes[1].pcolor(t, frequencies, wave.real, shading='auto')
axes[1].scatter(tcentsd, np.zeros_like(tcentsd)+10, color="red", s=15)
axes[1].set_ylabel("Частота, Гц")
axes[1].set_title("Вейвлет спектр по мексиканским шляпам")
#fig.colorbar(c, ax=axes[1])
axes[2].pcolor(t, frequencies, wavelet_u_restored.real, shading='auto')
axes[2].scatter(tcentsd, np.zeros_like(tcentsd)+10, color="red", s=15)
axes[2].set_ylabel("Frequency, Hz")
axes[2].set_title("Восстановленный вейвлет спектр")

axes[3].plot(t, decode, t, decode2, t, u.real)
#axes[3].plot(tcol, np.cos(phases), tcol, np.cos(phases_true))
axes[3].set_title("Оранжевый - точно восстановленный, синий - восстановленный с редукцией по ГК, зелёный - исходный")

axes[3].vlines(t[30::30], -2, 2, color="black")
axes[0].scatter(tcentsd, np.zeros_like(tcentsd), color="red", s=15)


fig.savefig(filename, dpi=250)
plt.show()
