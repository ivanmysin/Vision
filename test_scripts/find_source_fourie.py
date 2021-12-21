import numpy as np
# from scipy.signal import hilbert
import matplotlib.pyplot as plt
nums = np.arange(2, 11)
detMspecs = np.ones(nums.size, dtype=np.float)

f_u = 400
for i, n in enumerate(nums):
    # print(n)
    freqs = np.geomspace(1, 440, num=n) #np.arange(1, 50, 4) #


    Mspec = np.zeros( [freqs.size, freqs.size], dtype=np.float)
    for idx, f_M in enumerate(freqs):
        F_ricker = 2 / np.sqrt(np.pi) * freqs**2 / f_M**3 * np.exp( - (freqs/f_M)**2 )
        Mspec[idx, :] = F_ricker

    detMspecs[i] = np.linalg.det(Mspec)
    print(detMspecs[i])

fig, axes = plt.subplots(ncols=2)
axes[0].plot(nums, detMspecs )
axes[1].plot(nums, np.log10(detMspecs) )

axes[0].set_xlabel("Число частот")
axes[1].set_xlabel("Число частот")
axes[0].set_ylabel("Определитель матрицы")
axes[1].set_ylabel("Логарифм определителя матрицы")
plt.show()

"""
Mspec_inv = np.linalg.inv(Mspec)


f_M = np.min(freqs)
t_R = np.sqrt(2) / np.pi / f_M
x = np.arange(-t_R * 2.5, t_R * 2.5, 0.0001)

u = np.cos(2*np.pi*f_u*x)
u_ort = np.sin(2*np.pi*f_u*x)

coded_spec = np.zeros(freqs.size, dtype=np.float)
for idx, f_M in enumerate(freqs):
    f_ricker = (1 - 2*(np.pi*f_M*x)**2)*np.exp(-(np.pi*f_M*x)**2)

    coded_spec[idx] = np.sqrt(np.sum( u * f_ricker )**2 + np.sum( u_ort * f_ricker )**2)

theor_decoded_spec = np.zeros(freqs.size, dtype=np.float)
theor_decoded_spec[np.argmin( np.abs(freqs - f_u)) ] = 1
theor_coded_spec = 10000 * np.dot(Mspec, theor_decoded_spec)


#Mspec_inv = Mspec_inv / np.linalg.norm(Mspec_inv)
decoded_spec = 0.0001*np.dot(Mspec_inv, coded_spec)
# print(decoded_spec.shape)

plt.figure()
plt.plot(freqs, coded_spec)
plt.plot(freqs, theor_coded_spec)


plt.figure()
plt.plot(freqs, decoded_spec)
plt.plot(freqs, theor_decoded_spec)
plt.show()
"""