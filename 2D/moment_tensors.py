import numpy as np
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
import obspy

import SPECFEM3D_interface


def calc_moment_tensor_from_time_dependent(raw_mom_tensor, name="mtinv/run0001"):
    raw_mom_tensor_mat = np.empty((len(raw_mom_tensor), len(raw_mom_tensor[0].data)))
    dt = raw_mom_tensor[0].stats.delta
    for i in range(len(raw_mom_tensor)):
        raw_mom_tensor_mat[i, :] = raw_mom_tensor[i].data

    smooth_mom_tensor = np.empty(raw_mom_tensor_mat.shape)
    for i in range(len(raw_mom_tensor)):
        smooth_mom_tensor[i, :] = savgol_filter(raw_mom_tensor_mat[i, :], int(1 / (10 * dt)), 2)

    for i in range(3):
        plt.plot(smooth_mom_tensor[i, :])

    suspected_time = np.argmax(np.sum(np.abs(smooth_mom_tensor) / np.repeat(np.sum(np.abs(smooth_mom_tensor), axis=1)[:, np.newaxis], smooth_mom_tensor.shape[1], axis=1), axis=0))
    real_t0_obs = float(SPECFEM3D_interface.read_source_time_function(name).stats.starttime)
    green_t0_obs = float(SPECFEM3D_interface.read_source_time_function(name + "/green1").stats.starttime)
    time_shift = suspected_time * dt - 2 + real_t0_obs - green_t0_obs

    moment_tensor = np.empty((len(raw_mom_tensor)))
    for i in range(len(smooth_mom_tensor)):
        moment_tensor[i] = smooth_mom_tensor[i][np.argmax(np.abs(
            smooth_mom_tensor[i][int(suspected_time-1/(10*dt)):int(suspected_time+1/(10*dt))]))+int(suspected_time-1/(10*dt))]

    stf = np.zeros(len(raw_mom_tensor[0]))

    for i in range(len(smooth_mom_tensor)):
        stf += moment_tensor[i] * smooth_mom_tensor[i]

    stf = obspy.Trace(stf).taper(type="hann", max_percentage=0.005, side="both").data
    zero_crossings = np.where(np.diff(np.sign(stf - 1/np.e * stf.max())))[0]
    half_duration = (min((x for x in zero_crossings if x > stf.argmax()), default=None) - max((x for x in zero_crossings if x < stf.argmax()), default=None)) * dt / 2
    f0 = 1250/1221 / half_duration
    plt.close()

    return moment_tensor, time_shift, f0