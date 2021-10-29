import numpy as np
import h5py
from HyperColomn2D import HyperColomn

class HyperColomns:
    def __init__(self, radiuses, angles, directions, xx, yy, hcparams):
        NGHs = int(radiuses.size * angles.size)  # число гиперколонок

        self.hc_centers_x = np.zeros(NGHs, dtype=np.float)
        self.hc_centers_y = np.zeros(NGHs, dtype=np.float)
        self.Columns = []
        self.hdf4XYset = None
        nsigmas = 8

        delta_x = xx[0, 1] - xx[0, 0]
        delta_y = yy[1, 0] - yy[0, 0]
        Len_y, Len_x = xx.shape
        freq_teor_max = 0.5 / (np.sqrt(delta_x ** 2 + delta_y ** 2))
        sigma_teor_min = 1 / (2 * np.pi * freq_teor_max)

        idx = 0
        image_restored_by_HCs = np.zeros((Len_y, Len_x, NGHs), dtype=np.float)
        receptive_fields = np.zeros((Len_y, Len_x, NGHs), dtype=np.float)


        for r in radiuses:


            sigminimum = sigma_teor_min + 0.01*r # 0.05 - этот коэффициент важен, его нужно подбирать
            sigmaximum = 10 * sigminimum # 0.005 # функция от r

            sigmas = np.linspace(sigminimum, sigmaximum, nsigmas)

            frequencies = np.asarray([0.5, 2.0, ])  #1 / (2*np.pi*np.geomspace(sigminimum, sigmaximum, 15))  #

            for an in angles:
                xc = r * np.cos(an)
                yc = r * np.sin(an)

                self.hc_centers_x[idx] = xc
                self.hc_centers_y[idx] = yc


                hc = HyperColomn([xc, yc], xx, yy, directions, sigmas, frequencies=frequencies, params=hcparams)

                self.Columns.append(hc)
                sigma_rep_field = sigminimum
                receptive_field = np.exp(-0.5 * ((yy - yc) / sigma_rep_field) ** 2 - 0.5 * ((xx - xc) / sigma_rep_field) ** 2)
                receptive_fields[:, :, idx] = receptive_field
                idx += 1
        summ = np.sum(receptive_fields, axis=2)
        summ[summ == 0] = 0.001
        for i in range(NGHs):
            receptive_fields[:, :, i] /= summ
        self.receptive_fields = receptive_fields

    def encode(self, image):
        Encoded_full = []

        for hc in self.Columns:
            encoded = hc.encode(image)
            Encoded_full.append(encoded)

        return Encoded_full

    def decode(self, Encoded_full):
        image_restored = np.zeros_like(self.Columns[0].xx)
        for hc_idx, hc in enumerate(self.Columns):
            image_restored += self.receptive_fields[:, :, hc_idx] * hc.decode(Encoded_full[hc_idx])
        return image_restored


    #########################################################################
    def encoded2vector(self, Encoded_full):
        Nfreqs = len(Encoded_full[0]) - 1
        Nfreq_feasures = 6
        N = len(Encoded_full) * (Nfreqs * Nfreq_feasures + 1)
        X_train = np.zeros(N, dtype=np.float64)

        gidx = 0
        for hc in Encoded_full:
            for freq_data in hc:
                try:
                    X_train[gidx] = freq_data["abs"]
                    gidx += 1
                    X_train[gidx] = freq_data["peak_freq"]
                    gidx += 1
                    X_train[gidx] = np.sin(freq_data["dominant_direction"])
                    gidx += 1
                    X_train[gidx] = np.cos(freq_data["dominant_direction"])
                    gidx += 1
                    X_train[gidx] = np.sin(freq_data["phi_0"])
                    gidx += 1
                    X_train[gidx] = np.cos(freq_data["phi_0"])
                    gidx += 1
                except KeyError:
                    X_train[gidx] = freq_data["mean_intensity"]
                    gidx += 1

        return X_train