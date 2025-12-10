import SPECFEM3D_interface
import seismograms_handler
import numpy as np
import pickle
import obspy
import data


def calc_green_functions(name, source, index):
    observed_seismograms = SPECFEM3D_interface.read_observed_seismograms(name, True)
    observed_seismograms.sort(keys=['station'])
    SPECFEM3D_interface.create_green_projects_multi(name, source)
    SPECFEM3D_interface.run_green_modelling_parallel(name)
    green_functions = SPECFEM3D_interface.read_green_functions(name)
    for i in range(len(green_functions)):
        green_functions[i].sort(keys=['station'])
    observed_seismograms = observed_seismograms.filter("lowpass", freq=2, zerophase=True)
    observed_seismograms.trim(starttime=observed_seismograms[0].stats.starttime - 5,
                              endtime=observed_seismograms[0].stats.endtime, fill_value=0, pad=True)
    observed_seismograms_freq, dt, total_time = seismograms_handler.transform_stream_into_frequency_domain(observed_seismograms)
    green_functions_freq = np.empty((np.ma.size(observed_seismograms_freq, 0), np.ma.size(observed_seismograms_freq, 1),
                                    6), dtype=np.complex_)
    for i in range(len(green_functions)):
        green_functions[i] = green_functions[i].filter("lowpass", freq=2, zerophase=True)
        green_functions[i].trim(starttime=green_functions[i][0].stats.starttime - 3,
                                endtime=green_functions[i][0].stats.endtime + 2, fill_value=0, pad=True)
        green_functions_freq[:, :, i] = seismograms_handler.transform_stream_into_frequency_domain(green_functions[i])[0]

    with open("/DATA/eyal/specfem3d/PROJECTS/mtinv/test5/run000" + name[-1] + "/greenloc" + str(index) + ".pk", "wb") as f:
        pickle.dump(green_functions, f)

    moments_approx_freq = np.empty((np.ma.size(observed_seismograms_freq, 1), 6), dtype=np.complex_)
    g_ord = green_functions_freq.transpose((1, 0, 2))
    g_tr = green_functions_freq.transpose((1, 2, 0))
    gtg = np.matmul(np.conjugate(g_tr), g_ord)
    value_to_add = (10 ** -2) * np.max(np.abs(np.diagonal(gtg, axis1=1, axis2=2)))

    for i in range(np.ma.size(observed_seismograms_freq, 1)):
        curr_g = green_functions_freq[:, i, :]
        curr_seis = observed_seismograms_freq[:, i]
        gtg_curr = gtg[i, :, :]
        matrix_to_add = np.ones((np.ma.size(gtg_curr, 0))) * value_to_add
        g_g = np.matmul(np.linalg.inv(gtg_curr + np.diag(matrix_to_add)), np.conjugate(curr_g.transpose()))
        moments_approx_freq[i, :] = np.dot(g_g, curr_seis)

    moments_approx_freq = moments_approx_freq.transpose()
    moments_approx = np.fft.ifft(moments_approx_freq)
    moments_approx = np.real(moments_approx)

    moments_approx_stream = obspy.Stream()
    for i in range(np.ma.size(moments_approx, 0)):
        moments_approx_stream.append(obspy.Trace(moments_approx[i, :],
                                                 {'sampling_rate': 1/dt, 'delta': dt, 'network': "ET",
                                                  'station': "source", 'channel': data.channels[i],
                                                  'npts': len(moments_approx[i, :]),
                                                  'starttime': observed_seismograms[0].stats.starttime,
                                                  'endtime': observed_seismograms[0].stats.endtime}))
    moments_approx_stream.taper(type="hann", max_percentage=0.01, side="both")
    moments_approx_stream.filter("lowpass", freq=2, zerophase=True)
    with open("moments_approx_stream_" + name[-1] + "_" + str(index) + ".pk", 'wb') as f:
        pickle.dump(moments_approx_stream, f)