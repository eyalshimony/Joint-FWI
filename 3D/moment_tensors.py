import numpy as np
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
import obspy


def calc_moment_tensor_from_time_dependent(raw_mom_tensor):
    raw_mom_tensor_mat = np.empty((len(raw_mom_tensor), len(raw_mom_tensor[0].data)))
    dt = raw_mom_tensor[0].stats.delta
    for i in range(len(raw_mom_tensor)):
        raw_mom_tensor_mat[i, :] = raw_mom_tensor[i].data

    smooth_mom_tensor = np.empty(raw_mom_tensor_mat.shape)
    for i in range(len(raw_mom_tensor)):
        smooth_mom_tensor[i, :] = savgol_filter(raw_mom_tensor_mat[i, :],
                                                int(2 * np.round(np.ma.size(smooth_mom_tensor, 1) / 100) + 1), 2)

    for i in range(6):
        plt.plot(smooth_mom_tensor[i, :])

    suspected_time = np.argmax(np.sum(np.abs(smooth_mom_tensor) / np.repeat(np.sum(np.abs(smooth_mom_tensor), axis=1)[:, np.newaxis], smooth_mom_tensor.shape[1], axis=1), axis=0))
    time_shift = suspected_time * dt

    mul_fact = 2.847 * 10**22

    moment_tensor = np.empty((len(raw_mom_tensor)))
    for i in range(len(smooth_mom_tensor)):
        moment_tensor[i] = smooth_mom_tensor[i][np.argmax(np.abs(
            smooth_mom_tensor[i][int(suspected_time-1/(4*dt)):int(suspected_time+1/(4*dt))]))+int(suspected_time-1/(4*dt))]

    moment_tensor = mul_fact * moment_tensor

    stf = np.zeros(len(raw_mom_tensor[0]))

    for i in range(len(smooth_mom_tensor)):
        stf += moment_tensor[i] * smooth_mom_tensor[i]

    stf = obspy.Trace(stf).taper(type="hann", max_percentage=0.005, side="both").integrate().data
    half_duration = (np.argmax(stf >= stf.max()/2) - np.argmax(stf >= stf.max()*0.0786)) * dt * 1.628 * 1.2

    return moment_tensor, time_shift, half_duration