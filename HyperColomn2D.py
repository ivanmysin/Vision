import numpy as np
from scipy.optimize import minimize
from scipy.signal import oaconvolve as convolve2d

class HyperColomn:
    def __init__(self, centers, xx, yy, angles, sigmas, frequencies=None):

        self.cent_x = centers[0]
        self.cent_y = centers[1]
        self.xx = xx - self.cent_x
        self.yy = yy - self.cent_y



        self.x = self.xx[0, :]
        self.y = self.yy[:, 0]

        self.cent_x_idx = np.argmin( np.abs(self.x) )
        self.cent_y_idx = np.argmin( np.abs(self.y) )

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        self.angles = angles
        self.sigmas = sigmas

        self.aproximate_hilbert_kernel()

        self.rot_xx = []
        self.rot_yy = []

        if frequencies is None:
            self.frequencies = np.geomspace(0.1, 102.4, num=11)
        else:
            self.frequencies = frequencies

        self.mexican_hats = []
        self.hilbert_aproxed = []


        self.rotate_xy()
        self.compute_kernels()

    def aproximate_hilbert_kernel(self):
        w0 = np.ones(self.sigmas.size, dtype=np.float)
        opt_res = minimize(self._hilbert_dgs_diff, x0=w0, args=(self.xx, self.yy, self.sigmas))
        self.dg_weiths = opt_res.x
        # np.save(self.file_saving_ws, self.dg_weiths)
        self.H_aproxed = self.sum_gaussian_derivaries(self.dg_weiths, self.xx, self.yy, self.sigmas)  #
        self.H_aproxed_normed = self.H_aproxed / np.sqrt(np.sum(self.H_aproxed ** 2))

    def rotate_xy(self):
        for phi in self.angles:
            xx = self.xx * np.cos(phi) - self.yy * np.sin(phi)
            yy = self.xx * np.sin(phi) + self.yy * np.cos(phi)

            self.rot_xx.append(xx)
            self.rot_yy.append(yy)

    def compute_kernels(self):
        for phi_idx in range(len(self.angles)):

            xx = self.rot_xx[phi_idx]
            yy = self.rot_yy[phi_idx]
            H_aprox = self.sum_gaussian_derivaries(self.dg_weiths, xx, yy, self.sigmas)
            H_aprox = H_aprox / np.sqrt(np.sum(H_aprox**2))

            self.hilbert_aproxed.append(H_aprox)


            self.mexican_hats.append([])
            for freq in self.frequencies:
                sigma = 1 / (np.sqrt(2) * np.pi * freq)
                hat = self.get_rickers(sigma, xx, yy)
                self.mexican_hats[phi_idx].append(hat)


    def get_rickers(self, sigma, xx, yy):
        ricker = 0.005 * (4 - xx**2 / sigma**2) * np.exp(-(xx**2 + 4 * yy**2) / (8 * sigma**2)) / sigma**4
        return ricker

    def get_gaussian_derivative(self, sigma, xx, yy):
        gaussian = np.exp(-(xx**2 + 4 * yy**2) / (8 * sigma**2)) / (4 * np.pi * sigma**2)
        gaussian_dx = gaussian * -0.25 * xx / (sigma**2)
        return gaussian_dx

    def sum_gaussian_derivaries(self, w, xx, yy, sigmas):
        dgsum = np.zeros_like(xx)
        for idx, sigma in enumerate(sigmas):
            gaussian_dx = self.get_gaussian_derivative(sigma, xx, yy)
            dgsum += w[idx] * gaussian_dx
        return dgsum

    def _hilbert_dgs_diff(self, w, xx, yy, sigmas):
        H =  1 / (np.pi * self.x)
        sum_dg = self.sum_gaussian_derivaries(w, xx, yy, sigmas)
        E = np.sum( (H - sum_dg[self.x.size//2, :])**2 )

        return E

    def encode(self, U):
        
        # U_restored = np.zeros_like(U)
        # print(self.dx)
        Encoded = []

        for phi_idx in range(len(self.angles)):
            Encoded.append([])

            u_imag = convolve2d(U, self.hilbert_aproxed[phi_idx], mode="same")

            u = U + 1j*u_imag

            dx = self.dx*np.cos(self.angles[phi_idx])
            dy = self.dx*np.sin(self.angles[phi_idx])

            # print(dx, dy)

            cent_x_idx_b = np.argmin( (self.xx - dx)**2 + (self.yy - dy)**2 )
            cent_x_idx_f = np.argmin( (self.xx + dx)**2 + (self.yy + dy)**2 )

            dx_dist = 2 * np.sqrt( dx**2 + dy**2 )
            # plt.figure()
            # # plt.pcolormesh(self.x, self.y, U_restored, cmap="rainbow", shading="auto")
            # plt.scatter(self.xx.ravel()[cent_x_idx_b], self.yy.ravel()[cent_x_idx_b])
            # plt.scatter(self.xx.ravel()[cent_x_idx_f], self.yy.ravel()[cent_x_idx_f])
            # plt.vlines(0, -0.1, 0.1)
            # plt.hlines(0, -0.1, 0.1)
            # plt.show()

            #peak_freqs = []
            for freq_idx, freq in enumerate(self.frequencies):
                if freq > 1 / (2*dx): continue

                Ucoded = convolve2d(u, self.mexican_hats[phi_idx][freq_idx], mode="same")
                phase2 = np.angle(Ucoded.ravel()[cent_x_idx_b])
                phase1 = np.angle(Ucoded.ravel()[cent_x_idx_f])

                phase_diff = phase1 - phase2
                if phase_diff < -np.pi: phase_diff += 2*np.pi
                if phase_diff >  np.pi: phase_diff -= 2*np.pi

                peak_freq = phase_diff / dx_dist # (2 * dx) # peak_freq
                # peak_freqs.append(peak_freq)

                encoded_dict = {
                    "peak_freq" : peak_freq,
                    "phi_0" : np.angle(Ucoded[self.cent_y_idx, self.cent_x_idx]),
                    "abs" : np.abs(Ucoded[self.cent_y_idx, self.cent_x_idx])
                }
                Encoded[-1].append(encoded_dict)

                # U_restored += np.abs(Ucoded[self.cent_y_idx, self.cent_x_idx]) * np.cos( peak_freq * self.rot_xx[phi_idx] + phase2 )
                # break
            #break
        return Encoded




    def decode(self, Encoded):
        U_restored = np.zeros_like(self.xx)

        for phi_idx, angle_encoded in enumerate(Encoded):
            for freq_idx, freq_encoded in enumerate(angle_encoded):

                A = freq_encoded["abs"]
                peak_freq = freq_encoded["peak_freq"]
                phi_0 = freq_encoded["phi_0"]

                U_restored += A * np.cos(peak_freq * self.rot_xx[phi_idx] + phi_0)

        return U_restored




if __name__ == '__main__':
    import matplotlib.pyplot as plt

    centers = [0, 0]
    centers = [0, 0]
    # angles = [0.01*np.pi, 0.45*np.pi]#np.linspace(-np.pi, np.pi, 6, endpoint=False)
    angles = np.linspace(-np.pi, np.pi, 2, endpoint=False)

    print(angles)
    Len_y = 100
    Len_x = 100

    nsigmas = 8
    sigminimum = 0.05
    sigmaximum = 0.005
    sigmas = np.linspace(sigminimum, sigmaximum, nsigmas)
    xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, Len_y), np.linspace(0.5, -0.5, Len_x))

    image = np.zeros_like(xx)
    for idx in range(1):
        f = 6.4  # np.random.rand() * 20
        # an = np.random.rand() * 2*np.pi - np.pi
        an = np.pi  # np.random.choice(angles)

        xx_ = xx * np.cos(an) - yy * np.sin(an)
        image += np.cos(2 * np.pi * xx_ * f)

    hc = HyperColomn(centers, xx, yy, angles, sigmas)

    Encoded = hc.encode(image)
    image_restored = hc.decode(Encoded)

    # for hats in hc.mexican_hats:
    #     for hat in hats:
    #         plt.figure()
    #         plt.pcolormesh(hc.x, hc.y, hat, cmap="rainbow", shading="auto")
    #         plt.show()

    # for h_apriximated in hc.hilbert_aproxed:
    #     plt.figure()
    #     plt.pcolormesh(hc.x, hc.y, h_apriximated, cmap="rainbow", shading="auto")
    #     plt.show()

    plt.figure()
    plt.pcolormesh(hc.x, hc.y, image, cmap="rainbow", shading="auto")
    plt.figure()
    plt.pcolormesh(hc.x, hc.y, image_restored, cmap="rainbow", shading="auto")

    plt.show()