import numpy as np
import matplotlib.pyplot as plt

def get_gaussian(sigma, x, x_shift):
    xs = x + x_shift
    kernel = np.exp(-0.5 * ( xs/sigma )**2)

    return kernel

def get_gaussian_derivarive(sigma, x, x_shift):

    xs = x + x_shift
    kernel = np.exp(-0.5 * (xs/sigma)**2) * -xs / sigma**2
    return kernel

def get_mexican_hat(sigma, x, x_shift):
    xs = x + x_shift
    kernel = np.exp(-0.5 * ( xs/sigma )**2) * (1 - 2 * (xs/sigma)**2)

    return kernel


Npoits = 100
x = np.linspace(0, 1, Npoits)
K = np.empty( (0, Npoits), dtype=np.float ) # matrix of kernels

sigmas = np.linspace(0.05, 2, 10)
x_shifts = np.linspace(-0.5, 0.5, 10)

kernel_funcs = [get_gaussian, get_gaussian_derivarive, get_mexican_hat]

# print(K.shape)
for func in kernel_funcs:
    for sigma in sigmas:
        for shift in x_shifts:
            kernel = func(sigma, x, shift)
            kernel = kernel.reshape(1, -1)
            # print(kernel.shape)
            K = np.append(K, kernel, axis=0)

Xres = np.dot(K, x)

plt.plot(x, K[2, :])
plt.show()
# print(Xres.shape)