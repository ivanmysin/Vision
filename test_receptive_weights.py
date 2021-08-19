import numpy as np
import matplotlib.pyplot as plt


N = 5
Len_y = 200
Len_x = 200
receptive_fields = np.empty( (Len_x, Len_y, N), dtype=np.float )

xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, Len_y), np.linspace(0.5, -0.5, Len_x))


sigma = 0.05


for i in range(N):
    xc = np.random.rand() - 0.5
    yc = np.random.rand() - 0.5
    receptive_field = np.exp(-(yy-yc)**2 / (2 * sigma**2) - (xx-xc)**2 / (2 * sigma**2))

    receptive_fields[:, :, i] = receptive_field

summ = np.sum(receptive_fields, axis=2)

for i in range(N):
    receptive_fields[:, :, i] /= summ

print( np.sum(receptive_fields, axis=2) )