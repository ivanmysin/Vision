import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import oaconvolve as convolve2d
from scipy.signal import hilbert
from scipy.ndimage import convolve1d
from pprint import pprint
import matplotlib.pyplot as plt


class HyperColumn:
    def __init__(self, cent_x, cent_y, xx, yy, angles, sigmas, sgmGauss, frequencies=None, params={}):

        self.cent_x = cent_x
        self.cent_y = cent_y
        self.sgmGauss = sgmGauss
        self.params = params
        self.xx = np.copy(xx)  #
        self.yy = np.copy(yy)  #
        self.Nx, self.Ny = xx.shape

        self.x = np.copy(self.xx[:, 0])
        self.y = np.copy(self.yy[0, :])

        self.cent_x_idx = np.argmin(np.abs(self.x - self.cent_x))
        self.cent_y_idx = np.argmin(np.abs(self.y - self.cent_y))

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        self.angles = angles
        self.sigmas = sigmas

        self.approximate_hilbert_kernel()

        self.rot_xx = []
        self.rot_yy = []

        if frequencies is None:
            self.frequencies = np.geomspace(0.1, 102.4, num=11)
        else:
            self.frequencies = frequencies

        self.kotelnikov_limit = 0.5 / np.sqrt(self.dx ** 2 + self.dy ** 2)
        self.frequencies = self.frequencies[self.frequencies < self.kotelnikov_limit]
        self.params = params

        self.mexican_hats = []
        self.H_approxed_phi = []
        self.gaussian = np.ones_like(self.xx)

        self.rotate_xy()
        self.compute_kernels()

        self.normalization_factor_H = 1

    def approximate_hilbert_kernel(self):
        w0 = np.ones(self.sigmas.size, dtype=np.float64)
        opt_res = minimize(self._hilbert_dgs_diff, x0=w0, args=(self.xx, self.yy, self.sigmas))
        self.dg_weigths = opt_res.x
        #### np.save(self.file_saving_ws, self.dg_weigths)
        self.H_approxed = self.sum_gaussian_derivatives(self.dg_weigths, self.xx, self.yy, self.sigmas)  #

        # print( np.sum(self.H_approxed[self.x.size//2, :]**2) ) # !!!!!!!
        self.normalization_factor_H = np.sqrt(np.sum(self.H_approxed[self.x.size // 2, :] ** 2))
        self.H_approxed_normed = self.H_approxed / self.normalization_factor_H
        H = 1 / (np.pi * self.yy) * self.dy
        # H = H / np.sqrt( np.sum(H**2) )
        Gauss = np.exp(-self.xx ** 2 / (2 * self.sgmGauss ** 2)) / np.sqrt(2 * np.pi) / self.sgmGauss
        H *= Gauss * self.dx
        plt.plot(self.y, H[self.x.size // 2, :], label="Hyperb", linewidth=3)
        plt.plot(self.y, self.H_approxed[self.x.size // 2, :], label="Approxed", linewidth=1)
        plt.legend()
        plt.show()

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
            H_approx = self.sum_gaussian_derivatives(self.dg_weigths, yy, -xx, self.sigmas)

            # print( np.sum(  H_approx**2 ) )
            # H_approx = H_approx / np.sqrt( np.sum(  H_approx**2 ) )
            self.H_approxed_phi.append(H_approx)

            self.mexican_hats.append([])
            for freq in self.frequencies:
                sigma = 1 / (np.sqrt(2) * np.pi * freq)
                hat = self.get_rickers(sigma, xx, yy)
                self.mexican_hats[phi_idx].append(hat)

        sigma = np.max(self.sigmas)  # !!!!
        self.gaussian = np.exp(
            -(self.yy - self.cent_y) ** 2 / (2 * sigma ** 2) - (self.xx - self.cent_x) ** 2 / (2 * sigma ** 2))
        self.gaussian /= np.sum(self.gaussian)
        self.d_gauss_dx = -(self.xx - self.cent_x) / (sigma ** 2) * self.gaussian
        # self.d_gauss_dx /= np.sum(self.d_gauss_dx)
        self.d_gauss_dy = -(self.yy - self.cent_y) / (sigma ** 2) * self.gaussian
        # self.d_gauss_dy /= np.sum(self.d_gauss_dy)

    def get_rickers(self, sigma, xx, yy):
        sigma_x = sigma
        sigma_y = 0.5 * sigma_x
        ricker = (-1 + xx ** 2 / sigma_x ** 2) * np.exp(
            -yy ** 2 / (2 * sigma_y ** 2) - xx ** 2 / (2 * sigma_x ** 2)) / sigma_x ** 2

        ricker = ricker / np.sqrt(np.sum(ricker ** 2))  # -ricker / np.sqrt(np.sum(ricker**2))
        return ricker

    #    def derivate_gaussian_by_x(self, sigma, xx, yy):
    #        sigma_x = sigma
    #        sigma_y = 0.1 * sigma_x
    #        gaussian_dx = -xx * np.exp(-yy**2 / (2 * sigma_y**2) - xx**2 / (2 * sigma_x**2)) / sigma_x**2
    #        return gaussian_dx

    def derivate_gaussian_by_y(self, sigma, xx, yy):
        gaussian_dy = -yy * np.exp(-yy ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi) / sigma ** 2
        Gauss = np.exp(- self.xx ** 2 / (2 * self.sgmGauss ** 2)) / np.sqrt(2 * np.pi) / self.sgmGauss
        gaussian_dy *= Gauss * self.dx
        return gaussian_dy

    def sum_gaussian_derivatives(self, w, xx, yy, sigmas):
        dgsum = np.zeros_like(xx)
        for idx, sigma in enumerate(sigmas):
            gaussian_dy = self.derivate_gaussian_by_y(sigma, xx, yy)
            dgsum += w[idx] * gaussian_dy
        return dgsum

    def _hilbert_dgs_diff(self, w, xx, yy, sigmas):
        H = 1 / (np.pi * self.yy) * self.dy
        # H = H / np.sqrt( np.sum(H**2) )
        Gauss = np.exp(- self.xx ** 2 / (2 * self.sgmGauss ** 2)) / np.sqrt(2 * np.pi) / self.sgmGauss
        H *= Gauss * self.dx
        sum_dg = self.sum_gaussian_derivatives(w, xx, yy, sigmas)
        E = np.sum((H - sum_dg) ** 2)

        return E

    def find_dominant_direction(self, U):
        direction_max_resps = np.zeros_like(self.frequencies)

        for freq_idx in range(len(self.frequencies)):
            responses = np.zeros_like(self.angles)  # ответы по углам
            for an_idx, kernel in enumerate(self.mexican_hats):
                responses[an_idx] = np.abs(np.sum(kernel[freq_idx] * U))
            direction_max_resp = self.angles[np.argmin(responses)]  # возвращаем направление с максимальным ответом
            #direction_max_resp += 0.5*np.pi

            direction_max_resps[freq_idx] = direction_max_resp  # возвращаем направление после векторного усреднения ответов

        return direction_max_resps

    def _get_phi_0(self, slope, phi_train, x_train):
        s = np.sum(np.cos(phi_train - 2 * np.pi * slope * x_train))
        s += 1j * np.sum(np.sin(phi_train - 2 * np.pi * slope * x_train))
        phi_0 = np.angle(s)
        return phi_0

    def _get_Dist(self, slope, phi_train, x_train):
        phi_0 = self._get_phi_0(slope, phi_train, x_train)
        D = 2 * (1 - np.mean(np.cos(phi_train - 2 * np.pi * slope * x_train - phi_0)))
        if slope < 0.5:
            D += 100

        return D

    def find_peak_freq(self, phases_train, x_train, freq, y_train):
        # res = minimize_scalar(self._get_Dist, args=(phases_train, x_train), method='Golden')
        # np.savez("./results/saved.npz", phases_train, x_train)

        if self.params["use_circ_regression"]:
            slopes = np.linspace(0.5 * freq, 1.5 * freq, 200)
            D = np.zeros_like(slopes)
            for idx, slope in enumerate(slopes):
                D[idx] = self._get_Dist(slope, phases_train, x_train)
            slope = slopes[np.argmin(D)]

            res = minimize_scalar(self._get_Dist, args=(phases_train, x_train), bounds=[slope - 2, slope + 2],
                                  method='bounded')
            slope = float(res.x)
        else:
            # y_train
            idx1 = np.argmin(np.abs(x_train - 2 * self.dx) + np.abs(y_train))
            idx2 = np.argmin(np.abs(x_train + 2 * self.dx) + np.abs(y_train))
            dist_x = np.abs(x_train[idx2] - x_train[idx1])

            phase_diff = phases_train[idx2] - phases_train[idx1]
            phase_diff = phase_diff % (2 * np.pi)
            if phase_diff < 0:
                phase_diff += 2 * np.pi
            slope = phase_diff / (2 * np.pi * dist_x)

        return slope

    def encode(self, U):

        Encoded = []

        for i in range(len(self.frequencies)):
            Encoded.append({})

        directions = self.find_dominant_direction(U)  # [1.57, ] #
        mean_intensity = np.sum(self.gaussian * U)
        Grad_x = -np.sum(self.d_gauss_dx * U)
        Grad_y = -np.sum(self.d_gauss_dy * U)

        for direc_idx, direction in enumerate(directions):
            if direction < 0: direction += np.pi
            phi_idx = np.argmax(np.cos(direction - self.angles))
            U_noGrad = U - mean_intensity - (self.xx - self.cent_x) * Grad_x - (self.yy - self.cent_y) * Grad_y
            H = 1 / (np.pi * self.yy) * self.dy
            u_imag1d = convolve1d(U_noGrad[self.Nx // 2, :], H[self.Nx // 2, :], axis=0)
            Gauss = np.exp(- self.xx ** 2 / (2 * self.sgmGauss ** 2)) / np.sqrt(2 * np.pi) / self.sgmGauss
            H *= Gauss * self.dx

            u_H = convolve2d(U_noGrad, H, mode="same")
            u_imag = convolve2d(U_noGrad, self.H_approxed, mode="same")
            u_H_phi = convolve2d(U_noGrad, self.H_approxed_phi[phi_idx], mode="same")

            uu = hilbert(U_noGrad, axis=1)
            # uu = hilbert(U-mean_intensity, axis=0)

            # H = H / np.sqrt( np.sum(H**2) )

            fig, (ax0, ax1) = plt.subplots(nrows=2)
            ax1.plot(self.yy[self.Nx // 2, :], H[self.Nx // 2, :], label="H", linewidth=3)
            ax1.plot(self.yy[self.Nx // 2, :], self.H_approxed[self.Nx // 2, :], label="H_approxed", linewidth=1)
            ax1.plot(self.yy[self.Nx // 2, :], self.H_approxed_phi[phi_idx][self.Nx // 2, :], label="H_approxed_phi",
                     linewidth=1)
            ax1.legend()
            ax0.plot(self.yy[self.Nx // 2, :], U_noGrad[self.Nx // 2, :], label='U_noGrad')
            ax0.plot(self.yy[self.Nx // 2, :], np.imag(uu)[self.Nx // 2, :], label='hilbert')
            ax0.plot(self.yy[self.Nx // 2, :], u_imag1d, label='convolve1d')
            ax0.plot(self.yy[self.Nx // 2, :], u_H[self.Nx // 2, :], label='2d with H')
            ax0.plot(self.yy[self.Nx // 2, :], u_imag[self.Nx // 2, :], label='2d with H_approxed')
            ax0.plot(self.yy[self.Nx // 2, :], u_H_phi[self.Nx // 2, :], label='2d with H_approxed_phi')
            ax0.legend()
            plt.show()

            uu = U_noGrad + 1j * u_imag


            dx = self.dx * np.cos(self.angles[phi_idx])
            dy = self.dx * np.sin(self.angles[phi_idx])

            dx_dist = np.sqrt(dx ** 2 + dy ** 2)

            for freq_idx, freq in enumerate(self.frequencies):
                Ucoded = convolve2d(uu, self.mexican_hats[phi_idx][freq_idx], mode="same")
                Ucoded_normal = Ucoded.real / np.sqrt(np.sum(Ucoded.real ** 2)) + 1j * Ucoded.imag / np.sqrt(
                    np.sum(Ucoded.imag ** 2))

                # phase_0 = np.angle(Ucoded_normal[self.cent_x_idx, self.cent_y_idx])
                phase_0 = np.angle(uu[self.cent_x_idx, self.cent_y_idx])

                selected_vals = (np.abs(self.rot_xx[phi_idx]) < 15 * dx_dist) & (
                            np.abs(self.rot_yy[phi_idx]) < 15 * dx_dist)

                phases_train = np.angle(Ucoded_normal[selected_vals]).ravel()
                x_train = self.rot_xx[phi_idx][selected_vals].ravel()
                y_train = self.rot_yy[phi_idx][selected_vals].ravel()
                peak_freq = self.find_peak_freq(phases_train, x_train, freq, y_train)

                encoded_dict = {
                    "peak_freq": peak_freq,
                    "phi_0": phase_0,
                    "abs": np.abs(uu[self.cent_x_idx, self.cent_y_idx]),
                    # np.abs(Ucoded[self.cent_x_idx, self.cent_y_idx]),
                    "dominant_direction": direction,  # self.angles[phi_idx], # direction,
                    "direction_idx": phi_idx,
                    "central_wavelet_freq": freq,
                }
                Encoded[freq_idx] = encoded_dict

                # print(encoded_dict["abs"], peak_freq, self.cent_x, self.cent_y)

        hc_data = {
            "mean_intensity": mean_intensity,
            "Grad_x": Grad_x,
            "Grad_y": Grad_y,
            "cent_x": self.cent_x,
            "cent_y": self.cent_y,
            "intensity_at_cent": U[self.cent_x_idx, self.cent_y_idx]
        }

        Encoded.append(hc_data)

        # pprint(Encoded)
        return Encoded

    def decode(self, Encoded):
        U_restored = np.zeros_like(self.xx)

        Abs_HC = np.zeros_like(self.xx)
        Phases_HC = np.zeros_like(self.xx)
        Mean_U_HC = np.zeros_like(self.xx)

        for freq_idx, freq_encoded in enumerate(Encoded):
            if freq_idx == len(Encoded) - 1:
                Umean = freq_encoded["mean_intensity"]
            else:
                A = freq_encoded["abs"]
                peak_freq = freq_encoded["peak_freq"]
                phi_0 = freq_encoded["phi_0"]
                phi_idx = freq_encoded["direction_idx"]

                xhc_ = self.cent_x * np.cos(self.angles[phi_idx]) - self.cent_y * np.sin(self.angles[phi_idx])
                U_restored += A * np.cos(peak_freq * (self.rot_xx[phi_idx] - xhc_) * 2 * np.pi + phi_0)
                Phases_HC = peak_freq * (self.rot_xx[phi_idx] - xhc_) * 2 * np.pi + phi_0
                Abs_HC += A
        Mean_U_HC += Umean

        U_restored += Umean

        return U_restored, Abs_HC, Phases_HC, Mean_U_HC


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    params = {
        "use_circ_regression": False,
    }

    # angles = [0.01*np.pi, 0.45*np.pi]#np.linspace(-np.pi, np.pi, 6, endpoint=False)
    angles = np.linspace(-np.pi, np.pi, 16, endpoint=False)

    # print(angles)
    Nx = 200
    Ny = 200
    sgmGauss = 0.1

    nsigmas = 8
    sigminimum = 0.05
    sigmaximum = 0.005
    sigmas = np.linspace(sigminimum, sigmaximum, nsigmas)
    yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, Ny), np.linspace(1.0, -1.0, Nx))

    image = np.zeros_like(xx)
    frequencies = np.asarray([12.0, ])  # np.asarray([1.5, 8.0, 16.0, 80])  # np.geomspace(1.5, 25, num=5) #

    f = 15.0
    an = 0.25 * np.pi

    xx_ = xx * np.cos(an) - yy * np.sin(an)
    image = 0.5 * (np.cos(2 * np.pi * xx_ * f + 0.5 * np.pi) + 1) * np.exp(
        -0.5 * ((xx - 0.5) / 0.05) ** 2 - 0.5 * ((yy - 0.5) / 0.05) ** 2)

    hc = HyperColumn(0, 0, xx, yy, angles, sigmas, sgmGauss, frequencies=frequencies, params=params)



    for H_approxed_phi in hc.H_approxed_phi:
        image_im = convolve2d(image, H_approxed_phi, mode='same')
        fig, axes = plt.subplots(ncols=3, figsize=(20, 5), sharex=True, sharey=True)
        # plt.pcolormesh(hc.x, hc.y, hc.H_approxed_normed, cmap="rainbow", shading="auto")
        axes[0].pcolormesh(hc.x, hc.y, image, cmap="gray", shading="auto")
        axes[1].pcolormesh(hc.x, hc.y, hc.H_approxed_normed, cmap="gray", shading="auto")

        axes[2].pcolormesh(hc.x, hc.y, image_im, cmap="gray", shading="auto")

        # plt.plot(hc.x, hc.H_approxed_normed[100, :])

        # plt.figure()
        # plt.pcolormesh(hc.x, hc.y, image, cmap="rainbow", shading="auto")
        # plt.figure()
        # plt.pcolormesh(hc.x, hc.y, image_restored, cmap="rainbow", shading="auto")

        plt.show()