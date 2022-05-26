import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import oaconvolve as convolve2d
from scipy.signal import hilbert
from scipy.ndimage import convolve1d
from pprint import pprint
from sympy import diff, lambdify, exp, Symbol
import matplotlib.pyplot as plt


class HyperColumn:
    def __init__(self, cent_x, cent_y, xx, yy, angles, sigmas, sgmGauss, frequencies=None, params={}):

        self.sgmGauss = sgmGauss
        self.params = params
        self.xx = np.copy(xx)  #
        self.yy = np.copy(yy)  #
        self.Nx, self.Ny = xx.shape

        self.x = np.copy(self.xx[:, 0])
        self.y = np.copy(self.yy[0, :])

        self.cent_x_idx = np.argmin(np.abs(self.x - cent_x))
        self.cent_y_idx = np.argmin(np.abs(self.y - cent_y))
        self.cent_x = self.x[self.cent_x_idx]  # corrected coordinates of HC center on the mesh
        self.cent_y = self.y[self.cent_y_idx]

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        self.angles = angles
        self.dir = []
        self.sigmas = sigmas

        self.approximate_hilbert_kernel()

        self.rot_xx = []
        self.rot_yy = []
        self.rot_xc = []
        self.rot_yc = []

        if frequencies is None:
            self.frequencies = np.geomspace(0.1, 102.4, num=11)
        else:
            self.frequencies = frequencies

        self.kotelnikov_limit = 0.5 / np.sqrt(self.dx ** 2 + self.dy ** 2)
        self.frequencies = self.frequencies[self.frequencies < self.kotelnikov_limit]
        self.params = params

        self.mexican_hats = []
        self.mexican_hats_y = []
        self.dGauss_x = []
        self.H_approxed_phi = []
        self.gaussian = np.ones_like(self.xx)

        self.rotate_xy()
        self.compute_kernels()

        self.normalization_factor_H = 1

    #    def derivate_gaussian_by_x(self, sigma, xx, yy):
    #        sigma_x = sigma
    #        sigma_y = 0.1 * sigma_x
    #        gaussian_dx = -xx * np.exp(-yy**2 / (2 * sigma_y**2) - xx**2 / (2 * sigma_x**2)) / sigma_x**2
    #        return gaussian_dx

    def derivate_gaussian_by_x(self, sigma, xx, yy):
        gaussian_dx = -xx * np.exp(-xx ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi) / sigma ** 2
        Gauss = np.exp(- self.yy ** 2 / (2 * self.sgmGauss ** 2)) / np.sqrt(2 * np.pi) / self.sgmGauss
        gaussian_dx *= Gauss * self.dy
        return gaussian_dx

    def sum_gaussian_derivatives(self, w, xx, yy, sigmas):
        dgsum = np.zeros_like(xx)
        for idx, sigma in enumerate(sigmas):
            gaussian_dx = self.derivate_gaussian_by_x(sigma, xx, yy)
            dgsum += w[idx] * gaussian_dx
        return dgsum

    def _hilbert_dgs_diff(self, w, xx, yy, sigmas):
        H = 1 / (np.pi * self.xx) * self.dx
        # H = H / np.sqrt( np.sum(H**2) )
        Gauss = np.exp(- self.yy ** 2 / (2 * self.sgmGauss ** 2)) / np.sqrt(2 * np.pi) / self.sgmGauss
        H *= Gauss * self.dy
        sum_dg = self.sum_gaussian_derivatives(w, xx, yy, sigmas)
        E = np.sum((H - sum_dg) ** 2)
        return E

    def approximate_hilbert_kernel(self):
        w0 = np.ones(self.sigmas.size, dtype=np.float64)
        opt_res = minimize(self._hilbert_dgs_diff, x0=w0, args=(self.xx, self.yy, self.sigmas))
        self.dg_weigths = opt_res.x
        #### np.save(self.file_saving_ws, self.dg_weigths)
        self.H_approxed = self.sum_gaussian_derivatives(self.dg_weigths, self.xx, self.yy, self.sigmas)  #

        self.normalization_factor_H = np.sqrt(np.sum(self.H_approxed[:, self.y.size // 2] ** 2))
        self.H_approxed_normed = self.H_approxed / self.normalization_factor_H
        H = 1 / (np.pi * self.xx) * self.dx
        # H = H / np.sqrt( np.sum(H**2) )
        Gauss = np.exp(-self.yy ** 2 / (2 * self.sgmGauss ** 2)) / np.sqrt(2 * np.pi) / self.sgmGauss
        H *= Gauss * self.dy

        # plt.plot(self.x, H[:, self.y.size // 2], label="Hyperb", linewidth=3)
        # plt.plot(self.x, self.H_approxed[:, self.y.size // 2], label="Approxed", linewidth=1)
        # plt.legend()
        # plt.show()

    def get_rickers(self, sigma, xx, yy):
        sigma_x = sigma
        sigma_y = 0.5 * sigma_x
        ricker = (-1 + xx ** 2 / sigma_x ** 2) * np.exp(
            -yy ** 2 / (2 * sigma_y ** 2) - xx ** 2 / (2 * sigma_x ** 2)) / sigma_x ** 2

        ricker = ricker / np.sqrt(np.sum(ricker ** 2))  # -ricker / np.sqrt(np.sum(ricker**2))
        return ricker

    def rotate_xy(self):
        for phi in self.angles:
            xx = self.xx * np.cos(phi) - self.yy * np.sin(phi)
            yy = self.xx * np.sin(phi) + self.yy * np.cos(phi)
            xc = self.cent_x * np.cos(phi) - self.cent_y * np.sin(phi)
            yc = self.cent_x * np.sin(phi) + self.cent_y * np.cos(phi)

            self.rot_xx.append(xx)
            self.rot_yy.append(yy)
            self.rot_xc.append(xc)
            self.rot_yc.append(yc)

    def x_rot_of_xy(self, x, y, phi):
        x_rot = x * np.cos(phi) - y * np.sin(phi)
        return x_rot

    def y_rot_of_xy(self, x, y, phi):
        y_rot = x * np.sin(phi) + y * np.cos(phi)
        return y_rot

    def compute_kernels(self):

        ## find receptive field form
        x = Symbol("x")
        y = Symbol("y")
        sigma_x = Symbol("sigma_x")
        sigma_y = Symbol("sigma_y")
        gaussian = exp(- (x ** 2 / (2 * sigma_x ** 2) + y ** 2 / (2 * sigma_y ** 2)))  # гауссиана
        kernel_expr = diff(gaussian, y, 2)  # дифференцируем по х n раз
        dGauss_x_expr = diff(gaussian, x, 1)  # дифференцируем по х n раз
        # компилируем выражение в выполняемую функцию
        kernel_func = lambdify([x, y, sigma_x, sigma_y], kernel_expr)
        dGauss_x_func = lambdify([x, y, sigma_x, sigma_y], dGauss_x_expr)
        ####################################################################

        for phi_idx in range(len(self.angles)):

            xx = self.rot_xx[phi_idx]
            yy = self.rot_yy[phi_idx]
            H_approx = self.sum_gaussian_derivatives(self.dg_weigths, xx, yy, self.sigmas)

            # print( np.sum(  H_approx**2 ) )
            # H_approx = H_approx / np.sqrt( np.sum(  H_approx**2 ) )

            ####################################
            self.H_approxed_phi.append(H_approx)
            ####################################

            self.mexican_hats.append([])
            for freq in self.frequencies:
                sigma = 1 / (np.sqrt(2) * np.pi * freq)
                hat = self.get_rickers(sigma, xx, yy)
                self.mexican_hats[phi_idx].append(hat)

        # Define mexican hats to find the direction
        for phi_idx in range(len(self.angles) // 2):
            self.dir.append(self.angles[phi_idx])
            xx = self.rot_xx[phi_idx] - self.rot_xc[phi_idx]
            yy = self.rot_yy[phi_idx] - self.rot_yc[phi_idx]
            freq = self.frequencies[len(self.frequencies) - 1]
            sigma_y = 0.5 / (np.sqrt(2) * np.pi * freq)
            sigma_x = 0.2 * sigma_y
            # hat_y = (-2/sigma_y**2 + 4*yy**2/sigma_y**4)  * np.exp(-xx**2/(2*sigma_x**2) - yy**2/(2*sigma_y**2)) / (2*np.pi*sigma_x*sigma_y)
            # hat_y = hat_y - np.sum(hat_y)/self.Nx/self.Ny

            hat_y = kernel_func(xx, yy, sigma_x, sigma_y)  # Вычисляем ядро
            hat_y = hat_y / np.sqrt(np.sum(hat_y ** 2))  # Нормируем ядро

            dGauss_x = dGauss_x_func(xx, yy, sigma_y, sigma_y)
            dGauss_x = dGauss_x / np.sqrt(np.sum(dGauss_x ** 2))

            #################################
            self.mexican_hats_y.append(hat_y)
            self.dGauss_x.append(dGauss_x)
            #################################

        sigma = 0.1  # np.max(self.sigmas)  # !!!!
        self.gaussian = np.exp(
            -(self.yy - self.cent_y)**2 / (2* sigma**2) - (self.xx - self.cent_x)**2 / (2 * sigma**2))
        self.gaussian /= np.sum(self.gaussian)

        self.d_gauss_dx = -(self.xx - self.cent_x) / (sigma ** 2) * self.gaussian
        # self.d_gauss_dx /= np.sum(self.d_gauss_dx)
        self.d_gauss_dy = -(self.yy - self.cent_y) / (sigma ** 2) * self.gaussian
        # self.d_gauss_dy /= np.sum(self.d_gauss_dy)

    ######################################
    def find_dominant_direction(self, U):
        ######################################

        responses = np.zeros_like(self.dir)  # ответы по углам

        for an_idx in range(len(self.angles) // 2):
            a = self.dGauss_x[an_idx]
            b = self.mexican_hats_y[an_idx]
            kernel = a - b
            responses[an_idx] = np.abs(np.sum(a * U)) - np.abs(np.sum(b * U))

            # fig, axes = plt.subplots(ncols=2, figsize=(30, 15), sharex=True, sharey=True)
            # axes[0].pcolor(self.xx,self.yy, U, cmap="rainbow", shading='auto')
            # axes[1].pcolor(self.xx,self.yy, kernel, cmap="rainbow", shading='auto')
            # axes[1].set_title(str(responses[an_idx])+"  "+str(self.angles[an_idx]))
            # plt.show()
            # kernel=kernel

        direction_min_resp = self.angles[np.argmax(responses)]  # возвращаем направление с минимальным ответом

        return direction_min_resp

    def _get_phi_0(self, slope, phi_train, x_train):
        # s = np.sum(np.cos(phi_train - 2 * np.pi * slope * x_train))
        # s += 1j * np.sum(np.sin(phi_train - 2 * np.pi * slope * x_train))

        s = np.sum(np.exp(1j * (phi_train - 2 * np.pi * slope * x_train)))
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
            idx1 = np.argmin(np.abs(x_train - 2 * self.dx) + np.abs(y_train))
            idx2 = np.argmin(np.abs(x_train + 2 * self.dx) + np.abs(y_train))
            dist_x = np.abs(x_train[idx2] - x_train[idx1])

            phase_diff = phases_train[idx2] - phases_train[idx1]
            phase_diff = phase_diff % (2 * np.pi)
            if phase_diff < 0:
                phase_diff += 2 * np.pi
            slope = phase_diff / (2 * np.pi * dist_x)

        return slope

    ######################################
    def encode(self, U):
        ######################################

        Encoded = []

        for i in range(len(self.frequencies)):
            Encoded.append({})

        ## 1 ## Extract mean and gradient

        mean_intensity = np.sum(self.gaussian * U)
        Grad_x = -np.sum(self.d_gauss_dx * U)
        Grad_y = -np.sum(self.d_gauss_dy * U)
        UnoGrad = U - mean_intensity - (self.xx - self.cent_x) * Grad_x - (self.yy - self.cent_y) * Grad_y

        ## 2 ## Find direction

        direction = self.find_dominant_direction(UnoGrad)  # [1.57, ] #
        # for direc_idx, direction in enumerate(directions):
        if direction < 0: direction += np.pi
        phi_idx = np.argmax(np.cos(direction - self.angles))  # for self.H_approxed_phi[phi_idx]
        phi = self.angles[phi_idx]

        # fig, axes = plt.subplots(ncols=2, figsize=(30, 15), sharex=True, sharey=True)
        # axes[0].pcolor(self.xx,self.yy, self.d_gauss_dx, cmap='gray', shading='auto')
        # axes[1].pcolor(self.xx,self.yy, mean_intensity + (self.xx - self.cent_x) * Grad_x + (self.yy - self.cent_y) * Grad_y, cmap='gray', shading='auto')
        # plt.show()

        ## 3 ## Analytical signal

        H = 1 / (np.pi * self.xx) * self.dx
        u_imag1d = convolve1d(UnoGrad[:, self.cent_y_idx], H[:, self.cent_y_idx], axis=0)
        Gauss = np.exp(- self.yy ** 2 / (2 * self.sgmGauss ** 2)) / np.sqrt(2 * np.pi) / self.sgmGauss
        H *= Gauss * self.dy

        u_H = convolve2d(UnoGrad, H, mode="same")
        u_imag = convolve2d(UnoGrad, self.H_approxed, mode="same")
        u_H_phi = convolve2d(UnoGrad, self.H_approxed_phi[phi_idx], mode="same")

        H_UnoGrad = np.imag(hilbert(UnoGrad, axis=0))
        # uu = hilbert(U-mean_intensity, axis=0)
        # H = H / np.sqrt( np.sum(H**2) )

        ###########################
        uu = UnoGrad + 1j * u_H_phi  # u_imag
        ###########################

        """
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
        ax1.plot(self.xx[:, self.cent_y_idx], H[:, self.cent_y_idx], label="H", linewidth=3)
        ax1.plot(self.xx[:, self.cent_y_idx], self.H_approxed[:, self.cent_y_idx], label="H_approxed", linewidth=1)
        ax1.plot(self.rot_xx[phi_idx][:, self.cent_y_idx], self.H_approxed_phi[phi_idx][:, self.cent_y_idx], label="H_approxed_phi", linewidth=1)
        ax1.legend()
        ax0.plot(self.xx[:, self.cent_y_idx],         UnoGrad[:, self.cent_y_idx], label='UnoGrad')
        ax0.plot(self.xx[:, self.cent_y_idx],       H_UnoGrad[:, self.cent_y_idx], label='Hilbert(UnoGrad)')
        ax0.plot(self.xx[:, self.cent_y_idx],        u_imag1d,                  label='convolve1d')
        ax0.plot(self.xx[:, self.cent_y_idx],             u_H[:, self.cent_y_idx], label='2d with H')
        ax0.plot(self.xx[:, self.cent_y_idx],          u_imag[:, self.cent_y_idx], label='2d with H_approxed')
        ax0.plot(self.xx[:, self.cent_y_idx],         u_H_phi[:, self.cent_y_idx], label='2d with H_approxed_phi')
        ax0.legend()
        """

        dxy_dist = np.sqrt(self.dx ** 2 + self.dy ** 2)
        # rot_x1 = self.rot_xx[phi_idx][self.cent_x_idx+2, self.cent_y_idx+2]
        # rot_y1 = self.rot_yy[phi_idx][self.cent_x_idx+2, self.cent_y_idx+2]

        for i_freq, freq in enumerate(self.frequencies):
            # Ucoded = convolve2d(uu, self.mexican_hats[phi_idx][i_freq], mode="same")
            # Ucoded_normal = Ucoded.real / np.sqrt(np.sum(Ucoded.real ** 2)) + 1j * Ucoded.imag / np.sqrt(
            #        np.sum(Ucoded.imag ** 2))

            ## 4 ## Find phase

            # phase_0 = np.angle(Ucoded_normal[self.cent_x_idx, self.cent_y_idx])
            phase_0 = np.angle(uu[self.cent_x_idx, self.cent_y_idx])
            ###phase_1 = np.angle(uu[self.cent_x_idx+2, self.cent_y_idx+2])
            ###phase_diff = phase_1 - phase_0
            ###phase_diff = phase_diff % (2 * np.pi)
            ###if phase_diff < 0:
            ###    phase_diff += 2 * np.pi
            ###peak_freq = phase_diff/(2*np.pi*np.abs(rot_x1-rot_xc))

            ## 5 ## Find frequency

            # selected_vals = (np.abs(self.rot_xx[phi_idx]-self.rot_xc[phi_idx]) < 8 * dxy_dist) & (  #!!!
            #                 np.abs(self.rot_yy[phi_idx]-self.rot_yc[phi_idx]) < 1 * dxy_dist)      #!!!
            selected_vals = (np.abs(
                self.x_rot_of_xy(self.xx - self.cent_x, self.yy - self.cent_y, phi)) < 5 * dxy_dist) & (
                                    np.abs(self.y_rot_of_xy(self.xx - self.cent_x, self.yy - self.cent_y,
                                                            phi)) < 0.7 * dxy_dist)
            """                         
            fig, axes = plt.subplots(ncols=2, figsize=(30, 15), sharex=True, sharey=True)
            axes[0].pcolor(self.xx,self.yy, np.real(uu), cmap='gray', shading='auto')
            axes[0].scatter(self.xx[selected_vals],self.yy[selected_vals], s=150, color="green")
            plt.show()
            """

            # phases_train = np.angle(Ucoded_normal[selected_vals]).ravel()
            x_train = self.rot_xx[phi_idx][selected_vals].ravel()
            phases_train = np.angle(uu[selected_vals]).ravel()
            ampl_train = np.abs(uu[selected_vals]).ravel()
            ###y_train = self.rot_yy[phi_idx][selected_vals].ravel()
            ##peak_freq = self.find_peak_freq(phases_train, x_train, freq, y_train)
            ii = 0
            peak_freq = 0
            ampl = 0
            for i in range(0, len(x_train) - 1):
                x1 = x_train[i]
                i1 = np.argmin(np.abs(x_train - x1 - dxy_dist))
                x2 = x_train[i1]
                p1 = phases_train[i]
                p2 = phases_train[i1]

                if (p1 * p2 > 0) & (x2 != x1) & ((p2 - p1) * (x2 - x1) > 0) & (abs(x2 - x1) < 2 * dxy_dist) & (
                        abs(x2 - x1) > 0.5 * dxy_dist):
                    ii = ii + 1
                    peak_freq = peak_freq * (ii - 1) / ii + (p2 - p1) / (x2 - x1) / (2 * np.pi) / ii
                    ampl = ampl * (ii - 1) / ii + ampl_train[i] / ii

            # phase_0_aproxed = self._get_phi_0(peak_freq, phases_train, x_train)
            # print(phase_0, phase_0_aproxed, np.cos(phase_0 - phase_0_aproxed)  )

            """
            ax2.plot(self.xx[:, self.cent_y_idx],     np.angle(uu[:, self.cent_y_idx]), label='phase')
            ax2.plot(x_train,     phases_train, label='phases_train')
            ax2.legend()
            plt.show()
            """

            ## 6 ## Find amplitude

            # ampl = np.abs(uu[self.cent_x_idx, self.cent_y_idx])

            encoded_dict = {
                "peak_freq": peak_freq,
                "phase_0": phase_0,  # phase_0,
                "abs": ampl,
                # np.abs(Ucoded[self.cent_x_idx, self.cent_y_idx]),
                "dominant_direction": direction,  # self.angles[phi_idx], # direction,
                "direction_idx": phi_idx,
                "central_wavelet_freq": freq,
            }
            Encoded[i_freq] = encoded_dict

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

    ###############################################################################
    ###############################################################################

    def decode(self, Encoded):
        U_restored = np.zeros_like(self.xx)

        Abs_HC = np.zeros_like(self.xx)
        Phases_HC = np.zeros_like(self.xx)
        Mean_U_HC = np.zeros_like(self.xx)

        for i_freq, freq_encoded in enumerate(Encoded):
            if i_freq == len(Encoded) - 1:
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