import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import oaconvolve as convolve2d
from pprint import pprint


class HyperColomn:
    def __init__(self, centers, xx, yy, angles, sigmas, frequencies=None, params={}):

        self.cent_x = centers[0]
        self.cent_y = centers[1]
        self.params = params
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

        kotelnikov_limit = 0.5 / self.dx
        self.frequencies = self.frequencies[self.frequencies < kotelnikov_limit]



        self.mexican_hats = []
        self.hilbert_aproxed = []


        self.rotate_xy()
        self.compute_kernels()

        self.normalization_factor_H = 1

    def aproximate_hilbert_kernel(self):
        w0 = np.ones(self.sigmas.size, dtype=np.float)
        opt_res = minimize(self._hilbert_dgs_diff, x0=w0, args=(self.xx, self.yy, self.sigmas))
        self.dg_weiths = opt_res.x
        #### np.save(self.file_saving_ws, self.dg_weiths)
        self.H_aproxed = self.sum_gaussian_derivaries(self.dg_weiths, self.xx, self.yy, self.sigmas)  #
        self.normalization_factor_H = np.sqrt(np.sum(self.H_aproxed[self.x.size//2, :]**2))
        self.H_aproxed_normed = self.H_aproxed / self.normalization_factor_H


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
            H_aprox = H_aprox / np.sqrt( np.sum(  H_aprox**2 ) )
            self.hilbert_aproxed.append(H_aprox)

            self.mexican_hats.append([])
            for freq in self.frequencies:
                sigma = 1 / (np.sqrt(2) * np.pi * freq)
                hat = self.get_rickers(sigma, xx, yy)
                self.mexican_hats[phi_idx].append(hat)


    def get_rickers(self, sigma, xx, yy):
        sigma_x = sigma
        sigma_y = 0.5 * sigma_x
        ricker = (-1 + xx**2 / sigma_x**2) * np.exp(-yy**2 / (2 * sigma_y**2) - xx**2 / (2 * sigma_x**2)) / sigma_x**2

        ricker = -ricker / np.sqrt(np.sum(ricker**2))
        return ricker

    def get_gaussian_derivative(self, sigma, xx, yy):
        sigma_x = sigma
        sigma_y = 0.1 * sigma_x
        gaussian_dx = -xx * np.exp(-yy**2 / (2 * sigma_y**2) - xx**2 / (2 * sigma_x**2)) / sigma_x**2
        return gaussian_dx

    def sum_gaussian_derivaries(self, w, xx, yy, sigmas):
        dgsum = np.zeros_like(xx)
        for idx, sigma in enumerate(sigmas):
            gaussian_dx = self.get_gaussian_derivative(sigma, xx, yy)
            dgsum += w[idx] * gaussian_dx
        return dgsum

    def _hilbert_dgs_diff(self, w, xx, yy, sigmas):
        H = 1 / (np.pi * self.x)
        sum_dg = self.sum_gaussian_derivaries(w, xx, yy, sigmas)
        E = np.sum( (H - sum_dg[self.x.size//2, :])**2 )

        return E

    def find_dominant_direction(self, U):
        direction_max_resps = np.zeros_like(self.frequencies)

        for freq_idx in range(len(self.frequencies)):
            responses = np.zeros_like(self.angles)  # ответы по углам
            for an_idx, kernel in enumerate(self.mexican_hats):
                responses[an_idx] = np.sum( kernel[freq_idx] * U)

            direction_max_resp = self.angles[np.argmax(np.abs(responses))]  # возвращаем направление с максимальным ответом
            vectors_responses = responses * np.exp(1j * self.angles)
            near_angles_idxs = np.argsort(np.cos(direction_max_resp - self.angles))

            direction_max_resps[freq_idx] = np.angle(np.sum(vectors_responses[near_angles_idxs[:self.angles.size // 2]]))  # возвращаем направление после векторного усреднения ответов

        # direction_max_resps[direction_max_resps < 0] += np.pi
        return direction_max_resps

    def _get_phi_0(self, slope, phi_train, x_train):
        s = np.sum(np.cos(phi_train - 2 * np.pi * slope * x_train))
        s += 1j * np.sum( np.sin(phi_train - 2 * np.pi * slope * x_train))
        phi_0 = np.angle(s)
        return phi_0

    def _get_Dist(self, slope, phi_train, x_train):
        phi_0 = self._get_phi_0(slope, phi_train, x_train)
        D = 2 * (1 - np.mean(np.cos(phi_train - 2 * np.pi * slope * x_train - phi_0)))

        return D


    def find_peak_freq(self, phases_train, x_train, freq):
        # res = minimize_scalar(self._get_Dist, args=(phases_train, x_train), method='Golden')
        res = minimize_scalar(self._get_Dist, args=(phases_train, x_train), bounds=[0.8*freq, 1.5*freq], method='Bounded')
        slope = float(res.x)
        return slope


    def encode(self, U):

        Encoded = []
        for i in range(len(self.frequencies)):
            Encoded.append({})


        directions = self.find_dominant_direction(U)

        for direc_idx, direction in enumerate(directions):
            if direction < 0: direction += np.pi
            phi_idx = np.argmax( np.cos(direction - self.angles) )
            u_imag = convolve2d(U, self.hilbert_aproxed[phi_idx], mode="same")

            u = U + 1j * u_imag
            u.real = u.real / np.sqrt( np.mean(u.real**2) )
            u.imag = u.imag / np.sqrt( np.mean(u.imag**2) )

            dx = self.dx*np.cos(self.angles[phi_idx])
            dy = self.dx*np.sin(self.angles[phi_idx])

            dx_dist = np.sqrt( dx**2 + dy**2 )

            for freq_idx, freq in enumerate(self.frequencies):

                Ucoded = convolve2d(u, self.mexican_hats[phi_idx][freq_idx], mode="same")

                phase_0 = np.angle(Ucoded[self.cent_y_idx, self.cent_x_idx])

                selected_vals = (np.abs(self.rot_xx[phi_idx]) < 5*dx_dist)&(np.abs( self.rot_yy[phi_idx]) < 5*dx_dist)

                phases_train = np.angle(Ucoded[selected_vals]).ravel()
                x_train = self.rot_xx[phi_idx][selected_vals].ravel()
                peak_freq = self.find_peak_freq(phases_train, x_train, freq)


                encoded_dict = {
                    "peak_freq" : peak_freq,
                    "phi_0" : phase_0,
                    "abs" : np.abs(Ucoded[self.cent_y_idx, self.cent_x_idx]),
                    "dominant_direction" : self.angles[phi_idx], # direction,
                    "direction_idx" : phi_idx,
                }
                Encoded[freq_idx] = encoded_dict

        pprint(Encoded)
        return Encoded




    def decode(self, Encoded):
        U_restored = np.zeros_like(self.xx)

        for freq_idx, freq_encoded in enumerate(Encoded):
            A = freq_encoded["abs"]
            peak_freq = freq_encoded["peak_freq"]
            phi_0 = freq_encoded["phi_0"]
            phi_idx = freq_encoded["direction_idx"]
            U_restored += A * np.cos(peak_freq * self.rot_xx[phi_idx] * 2 * np.pi + phi_0)
        return U_restored




if __name__ == '__main__':
    import matplotlib.pyplot as plt

    params = {}

    centers = [0, 0]
    centers = [0, 0]
    # angles = [0.01*np.pi, 0.45*np.pi]#np.linspace(-np.pi, np.pi, 6, endpoint=False)
    angles = np.linspace(-np.pi, np.pi, 16, endpoint=False)

    # print(angles)
    Len_y = 200
    Len_x = 200

    nsigmas = 8
    sigminimum = 0.05
    sigmaximum = 0.005
    sigmas = np.linspace(sigminimum, sigmaximum, nsigmas)
    xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, Len_y), np.linspace(0.5, -0.5, Len_x))

    image = np.zeros_like(xx)
    frequencies = np.asarray([1.5, 8.0, 16.0, 80])  # np.geomspace(1.5, 25, num=5) #
    for idx in range(1):
        f = 80 # np.random.rand() * 20 # 10 # frequencies[2] #
        # print(f)
        an = np.random.rand() * np.pi  # np.random.rand() * 2*np.pi - np.pi
        # an = np.pi * 0.5 # np.pi  # np.random.choice(angles)

        xx_ = xx * np.cos(an) - yy * np.sin(an)
        image += np.cos(2 * np.pi * xx_ * f + 0.5*np.pi)


    hc = HyperColomn(centers, xx, yy, angles, sigmas, frequencies=frequencies, params=params)

    #main_direction = hc.find_dominant_direction(image)

    print(an, f)
    Encoded = hc.encode(image)
    image_restored = hc.decode(Encoded)

    # fig, axes = plt.subplots(nrows=len(hc.angles), ncols=len(hc.frequencies))
    # for an_idx, (hats, an) in enumerate(zip(hc.mexican_hats, hc.angles)):
    #     for freq_idx, (hat, fr) in enumerate(zip(hats, hc.frequencies)):
    #
    #         pr = image * hat
    #         axes[an_idx, freq_idx].pcolormesh(hc.x, hc.y, pr, cmap="rainbow", shading="auto")
    #
    #         resp = np.sum(pr)
    #         axes[an_idx, freq_idx].axis('off')
    #
    #         axes[an_idx, freq_idx].set_title( str(np.around(resp, 1)) )

    # for h_apriximated in hc.hilbert_aproxed:
    #     plt.figure()
    #     plt.pcolormesh(hc.x, hc.y, h_apriximated, cmap="rainbow", shading="auto")
    #     plt.show()

    plt.figure()
    plt.pcolormesh(hc.x, hc.y, image, cmap="rainbow", shading="auto")
    plt.figure()
    plt.pcolormesh(hc.x, hc.y, image_restored, cmap="rainbow", shading="auto")

    plt.show()