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
    def encoded2vector(self, Encoded_full, return_feasures_names=False):
        Nfreqs = len(Encoded_full[0]) - 1
        Nfreq_feasures = 6
        N = len(Encoded_full) * (Nfreqs * Nfreq_feasures + 1)
        X_train = np.zeros(N, dtype=np.float64)

        if return_feasures_names:
            feasures_names = []
            # datatype, xc, yc, cent_freq,
            name_template = "{datatype}, {cent_freq}, {xc}, {yc}"

        gidx = 0
        for hc in Encoded_full:
            for freq_data in hc:
                try:
                    X_train[gidx] = freq_data["abs"]
                    gidx += 1
                    if return_feasures_names:
                        name = name_template.format( datatype="abs", cent_freq=freq_data["central_wavelet_freq"], xc=hc[-1]["cent_x"], yc=hc[-1]["cent_y"]   )
                        feasures_names.append(name)

                    X_train[gidx] = freq_data["peak_freq"]
                    gidx += 1
                    if return_feasures_names:
                        name = name_template.format( datatype="peak_freq", cent_freq=freq_data["central_wavelet_freq"], xc=hc[-1]["cent_x"], yc=hc[-1]["cent_y"]   )
                        feasures_names.append(name)

                    X_train[gidx] = np.sin(freq_data["dominant_direction"])
                    gidx += 1
                    if return_feasures_names:
                        name = name_template.format( datatype="sin of direction", cent_freq=freq_data["central_wavelet_freq"], xc=hc[-1]["cent_x"], yc=hc[-1]["cent_y"]   )
                        feasures_names.append(name)

                    X_train[gidx] = np.cos(freq_data["dominant_direction"])
                    gidx += 1
                    if return_feasures_names:
                        name = name_template.format( datatype="cos of direction", cent_freq=freq_data["central_wavelet_freq"], xc=hc[-1]["cent_x"], yc=hc[-1]["cent_y"]   )
                        feasures_names.append(name)

                    X_train[gidx] = np.sin(freq_data["phi_0"])
                    gidx += 1
                    if return_feasures_names:
                        name = name_template.format( datatype="sin of phi_0", cent_freq=freq_data["central_wavelet_freq"], xc=hc[-1]["cent_x"], yc=hc[-1]["cent_y"]   )
                        feasures_names.append(name)

                    X_train[gidx] = np.cos(freq_data["phi_0"])
                    gidx += 1
                    if return_feasures_names:
                        name = name_template.format( datatype="cos of phi_0", cent_freq=freq_data["central_wavelet_freq"], xc=hc[-1]["cent_x"], yc=hc[-1]["cent_y"]   )
                        feasures_names.append(name)

                except KeyError:
                    X_train[gidx] = freq_data["mean_intensity"]
                    gidx += 1
                    if return_feasures_names:
                        name = name_template.format( datatype="mean_U", cent_freq=0, xc=hc[-1]["cent_x"], yc=hc[-1]["cent_y"]   )
                        feasures_names.append(name)

        if return_feasures_names:
            return X_train, feasures_names
        else:
            return X_train

    def get_feature_description(self, Encoded_full):
        feasures_names = []

        # type_data, xc, yc, cent_freq,
        name = "abs, {cent_freq}, {xc}, {yc}".format(0.1, 0.1)


        feasures_names.append(name)

        return feasures_names