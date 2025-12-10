import SPECFEM3D_interface
import seismograms_handler
import numpy as np
from matplotlib import pyplot as plt
import pickle


strain_seismograms = SPECFEM3D_interface.read_SU_seismograms("test", "r")
observed_seismograms = seismograms_handler.calculate_DAS_seismograms(strain_seismograms)
with open("/DATA/eyal/specfem2d/test/observed_seismograms.pk", "wb") as f:
    pickle.dump(observed_seismograms, f)

noise_levels = []
if __name__ == "__main__":
    observed_seismograms = []
    max_vals = []
    for i in range(1, 21):
        strain_seismograms = SPECFEM3D_interface.read_SU_seismograms(f"mtinv/run{i:04d}", "r")
        observed_seismograms.append(seismograms_handler.calculate_DAS_seismograms(strain_seismograms))
        curr_max = np.median(np.abs(seismograms_handler.ordered_stream_into_ordered_matrix(observed_seismograms[i - 1])).max(axis=1))
        max_vals.append(curr_max)
    for i in range(20):
        noise_level = np.sqrt(max_vals[i] / np.median(max_vals)) * np.median(max_vals) / 12
        noise_levels.append(noise_level)
        noise = seismograms_handler.generate_correlated_noise(observed_seismograms[i][0].stats.npts,
                                                              len(observed_seismograms[i]),
                                                              noise_level,
                                                              int(np.round(125.0 / 2400.0 / np.pi / observed_seismograms[i][0].stats.delta)),
                                                              int(np.round(125.0 / 5.0 / np.pi)))
        zs = np.arange(0, 2001, 5)
        noise[:, :401] *= np.exp(-0.4*zs / 125.0)[np.newaxis, :]
        noise[:, 401:802] *= np.exp(-0.4*zs / 125.0)[np.newaxis, :]
        for j in range(len(observed_seismograms[i])):
            observed_seismograms[i][j].data += noise[:, j]
        with open(f"/DATA/eyal/specfem2d/mtinv/observed_seismograms{i+1:04d}.pk", "wb") as f:
            pickle.dump(observed_seismograms[i], f)
        plt.imshow(seismograms_handler.ordered_stream_into_ordered_matrix(observed_seismograms[i]).T,
                   extent=[0, 4403, 12, 0], aspect='auto', cmap="cmc.vik", vmin=-np.abs(
                seismograms_handler.ordered_stream_into_ordered_matrix(observed_seismograms[i]).T).max(),
                   vmax=np.abs(seismograms_handler.ordered_stream_into_ordered_matrix(observed_seismograms[i]).T).max())
        plt.savefig(f"obseisnoise{i+1}.png")
        plt.close()
    with open("noise_levels.pk", "wb") as f:
        pickle.dump(noise_levels, f)