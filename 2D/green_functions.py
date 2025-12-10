import SPECFEM3D_interface
import seismograms_handler
import numpy as np
import pickle
import obspy
import data
from obspy.signal.interpolation import lanczos_interpolation
from opt_einsum import contract
import moment_tensors
import concurrent.futures
from itertools import repeat
import scipy.fft


def calc_moment_rate_functions_multi(name, inds, weights):
    for ind in inds:
        calc_moment_rate_functions(name + f"/run{ind:04d}", weights)


def calc_single_reg_param(reg_param, observed_seismograms_freq, ev, green_functions_freq, gtg, dt,
                          starttime, endtime, noise_level, weights, max_freq):
    moment_tensor_array = []

    g_tr = green_functions_freq.transpose((1, 2, 0))
    weights_mat = np.diag(weights)
    numerator_term = contract('fcs,st->fct', g_tr.conj(), weights_mat)
    for j in range(100):
        if j == 0:
            moment_tensor, model_resolution_trace = calc_single_noise(observed_seismograms_freq, ev,
                                                                      green_functions_freq, gtg, numerator_term,
                                                                      reg_param, dt, starttime, endtime, noise_level,
                                                                      max_freq, True)
        else:
            moment_tensor = calc_single_noise(observed_seismograms_freq, ev, green_functions_freq, gtg, numerator_term,
                                              reg_param, dt, starttime, endtime, noise_level, max_freq, False)
        moment_tensor_array.append(moment_tensor)
    moment_tensor_array = np.asarray(moment_tensor_array)
    moment_tensors_std = 1 / np.sqrt(2) * np.sqrt(np.sum(np.std(moment_tensor_array, axis=0) ** 2) +
                                                  np.sum(np.std(moment_tensor_array, axis=0)[2:] ** 2))
    moment_tensor_moment = 1 / np.sqrt(2) * np.sqrt(np.sum(np.mean(moment_tensor_array, axis=0) ** 2) +
                                                          np.sum(np.mean(moment_tensor_array[:, 2:], axis=0) ** 2))
    moment_tensors_rel_std = moment_tensors_std / moment_tensor_moment
    momen_tensors_rel_std_2 = np.sum(np.std(moment_tensor_array, axis=0) / np.abs(np.mean(moment_tensor_array, axis=0)))
    momen_tensors_rel_std_comp = np.std(moment_tensor_array, axis=0) / np.abs(np.mean(moment_tensor_array, axis=0))
    return reg_param, model_resolution_trace, moment_tensors_std, moment_tensors_rel_std, momen_tensors_rel_std_2, \
           momen_tensors_rel_std_comp, moment_tensor_array


def calc_single_noise(observed_seismograms_freq, ev, green_functions_freq, gtg, numerator_term, reg_param, dt, starttime,
                      endtime, noise_level, max_freq, is_first):
    noise_ext = seismograms_handler.generate_correlated_noise(observed_seismograms_freq.shape[0],
                                                              observed_seismograms_freq.shape[1],
                                                              noise_level,
                                                              int(np.round(125.0 / 2400.0 / np.pi / dt)),
                                                              int(np.round(125.0 / 5.0 / np.pi)))

    noise_freq = scipy.fft.fft(noise_ext)
    noise_spec = np.mean(np.abs(noise_freq), axis=0)
    noise_spec /= noise_spec.max()

    base_ev_arr = seismograms_handler.ordered_stream_into_ordered_matrix(ev)
    ev_noised_arr = base_ev_arr + noise_ext
    ev_noised_freq = scipy.fft.fft(ev_noised_arr)

    g_ord = green_functions_freq.transpose((1, 0, 2))

    value_to_add_f = reg_param * np.max(np.abs(np.diagonal(gtg, axis1=1, axis2=2))) * noise_spec
    reg_matrix_per_freq = np.eye(3)[np.newaxis, :, :] * value_to_add_f[:, np.newaxis, np.newaxis]
    inv_term_gg = np.linalg.inv(gtg + reg_matrix_per_freq)

    g_g_vec = contract('fck,fks->fcs', inv_term_gg, numerator_term)
    if is_first:
        mod_res = contract('fcs,fsp->fcp', g_g_vec, g_ord)
        tot_mod_res = np.dot(np.mean(np.abs(observed_seismograms_freq), axis=0) / np.sum(
            np.mean(np.abs(observed_seismograms_freq), axis=0)),
                             np.abs(mod_res).reshape(mod_res.shape[0], -1)).reshape(3, 3)
        model_resolution_trace = np.trace(tot_mod_res)
    moments_approx_freq = contract('fcs,fs->fc', g_g_vec, ev_noised_freq.T)

    moments_approx_freq = moments_approx_freq.transpose()
    moments_approx = np.fft.ifft(moments_approx_freq)
    moments_approx = np.real(moments_approx)

    moments_approx_stream = obspy.Stream()
    for i in range(np.ma.size(moments_approx, 0)):
        moments_approx_stream.append(obspy.Trace(moments_approx[i, :],
                                                 {'sampling_rate': 1 / dt, 'delta': dt, 'network': "ET",
                                                  'station': "source", 'channel': data.channels[i],
                                                  'npts': len(moments_approx[i, :]),
                                                  'starttime': starttime,
                                                  'endtime': endtime}))
    moments_approx_stream.taper(type="hann", max_percentage=0.01, side="both")
    moments_approx_stream.filter("lowpass", freq=max_freq, zerophase=True)
    moment_tensor, _, _ = moment_tensors.calc_moment_tensor_from_time_dependent(moments_approx_stream)
    if is_first:
        return list(moment_tensor), model_resolution_trace
    else:
        return list(moment_tensor)


def calc_params_for_reg_calc(name, noise_level, weights=None):
    observed_seismograms = SPECFEM3D_interface.read_observed_seismograms_from_pickle(name)
    with open("event" + name[9:] + ".pk", "rb") as f:
        ev = pickle.load(f)
    observed_seismograms.sort(keys=['station'])
    ev.sort(keys=['station'])
    obsarr = seismograms_handler.ordered_stream_into_ordered_matrix(observed_seismograms)
    max_freq = np.fft.fftfreq(7500, observed_seismograms[0].stats.delta)[:500][
        np.mean(np.abs(np.fft.fft(obsarr, axis=1)), axis=0)[:500].argmax()]
    orig_max = np.abs(seismograms_handler.ordered_stream_into_ordered_matrix(observed_seismograms.copy().
                                                                             filter("lowpass", freq=max_freq*1.725,
                                                                                    zerophase=True))).max()
    observed_seismograms.differentiate()

    green_functions = SPECFEM3D_interface.read_green_functions(name)
    for i in range(len(green_functions)):
        green_functions[i].sort(keys=['station'])

    observed_seismograms = observed_seismograms.filter("lowpass", freq=max_freq*1.725, zerophase=True)
    ev = ev.filter("lowpass", freq=max_freq*1.725, zerophase=True)
    observed_seismograms.trim(starttime=observed_seismograms[0].stats.starttime - 5,
                              endtime=observed_seismograms[0].stats.endtime, fill_value=0, pad=True)
    ev.trim(starttime=ev[0].stats.starttime - 5, endtime=ev[0].stats.endtime, fill_value=0, pad=True)
    rat = orig_max / np.abs(seismograms_handler.ordered_stream_into_ordered_matrix(ev)).max()
    for i in range(len(ev)):
        ev[i].data *= rat
    observed_seismograms_arr = seismograms_handler.ordered_stream_into_ordered_matrix(observed_seismograms)
    observed_seismograms_freq = np.fft.fft(observed_seismograms_arr)
    dt = observed_seismograms[0].stats.delta

    green_functions_freq = np.empty((np.ma.size(observed_seismograms_freq, 0), np.ma.size(observed_seismograms_freq, 1),
                                     3), dtype=np.complex_)
    for i in range(len(green_functions)):
        green_functions[i] = green_functions[i].filter("lowpass", freq=max_freq * 1.725, zerophase=True)
        green_functions[i].trim(starttime=green_functions[i][0].stats.starttime - 3,
                                endtime=green_functions[i][0].stats.endtime + 2, fill_value=0, pad=True)
        gf_arr = seismograms_handler.ordered_stream_into_ordered_matrix(green_functions[i])
        green_functions_freq[:, :, i] = np.fft.fft(gf_arr)

    g_ord = green_functions_freq.transpose((1, 0, 2))
    g_tr = green_functions_freq.transpose((1, 2, 0))
    weights_mat = np.diag(weights)
    gtg = contract('fcs,st,ftp->fcp', g_tr.conj(), weights_mat, g_ord)

    model_resolution_traces = {}
    moment_tensors_stds = {}
    moment_tensors_rel_stds = {}
    moment_tensors_rel_stds_2 = {}
    moment_tensors_rel_comps = {}
    moment_tensor_arrays = {}
    ks = np.logspace(-4, 0)
    with concurrent.futures.ProcessPoolExecutor(max_workers=50) as executor:
        for reg_param, model_resolution_trace, moment_tensors_std, moment_tensors_rel_std, moment_tensors_rel_std_2, \
            moment_tensors_rel_comp, moment_tensor_array in \
                executor.map(calc_single_reg_param, ks, repeat(observed_seismograms_freq), repeat(ev),
                             repeat(green_functions_freq), repeat(gtg), repeat(dt),
                             repeat(observed_seismograms[0].stats.starttime),
                             repeat(observed_seismograms[0].stats.endtime), repeat(noise_level), repeat(weights),
                             repeat(max_freq)):
            model_resolution_traces[reg_param] = model_resolution_trace
            moment_tensors_stds[reg_param] = moment_tensors_std
            moment_tensors_rel_stds[reg_param] = moment_tensors_rel_std
            moment_tensors_rel_stds_2[reg_param] = moment_tensors_rel_std_2
            moment_tensors_rel_comps[reg_param] = moment_tensors_rel_comp
            moment_tensor_arrays[reg_param] = moment_tensor_array
    model_resolution_traces_list = []
    moment_tensors_stds_list = []
    moment_tensors_rel_stds_list = []
    moment_tensors_rel_stds_2_list = []
    moment_tensors_rel_comps_list = []
    moment_tensor_arrays_list = []
    for k in ks:
        model_resolution_traces_list.append(model_resolution_traces[k])
        moment_tensors_stds_list.append(moment_tensors_stds[k])
        moment_tensors_rel_stds_list.append(moment_tensors_rel_stds[k])
        moment_tensors_rel_stds_2_list.append(moment_tensors_rel_stds_2[k])
        moment_tensors_rel_comps_list.append(moment_tensors_rel_comps[k])
        moment_tensor_arrays_list.append(moment_tensor_arrays[k])
    with open("reglists" + name[9:] + ".pk", "wb") as f:
        pickle.dump([model_resolution_traces_list, moment_tensors_stds_list, moment_tensors_rel_stds_list,
                     moment_tensors_rel_stds_2_list, moment_tensors_rel_comps_list, moment_tensor_arrays_list], f)


def calc_moment_rate_functions(name, weights=None):
    observed_seismograms = SPECFEM3D_interface.read_observed_seismograms_from_pickle(name)
    dt = observed_seismograms[0].stats.delta
    if weights is None:
        weights = np.ones(len(observed_seismograms))
    observed_seismograms.sort(keys=['station'])
    obsarr = seismograms_handler.ordered_stream_into_ordered_matrix(observed_seismograms)
    max_freq = np.fft.fftfreq(7500, observed_seismograms[0].stats.delta)[:500][
        np.mean(np.abs(np.fft.fft(obsarr, axis=1)), axis=0)[:500].argmax()]
    green_functions = SPECFEM3D_interface.read_green_functions(name)
    for i in range(len(green_functions)):
        green_functions[i].sort(keys=['station'])
    observed_seismograms.differentiate()
    observed_seismograms = observed_seismograms.filter("lowpass", freq=max_freq*1.725, zerophase=True)
    observed_seismograms.trim(starttime=observed_seismograms[0].stats.starttime - 5,
                              endtime=observed_seismograms[0].stats.endtime, fill_value=0, pad=True)
    obs_arr = seismograms_handler.ordered_stream_into_ordered_matrix(observed_seismograms)
    observed_seismograms_freq = np.fft.fft(obs_arr)
    green_functions_freq = np.empty((np.ma.size(observed_seismograms_freq, 0), np.ma.size(observed_seismograms_freq, 1),
                                     3), dtype=np.complex_)
    for i in range(len(green_functions)):
        green_functions[i] = green_functions[i].filter("lowpass", freq=max_freq*1.725, zerophase=True)
        green_functions[i].trim(starttime=green_functions[i][0].stats.starttime - 3,
                                endtime=green_functions[i][0].stats.endtime + 2, fill_value=0, pad=True)
        gf_arr = seismograms_handler.ordered_stream_into_ordered_matrix(green_functions[i])
        green_functions_freq[:, :, i] = np.fft.fft(gf_arr)

    g_ord = green_functions_freq.transpose((1, 0, 2))
    g_tr = green_functions_freq.transpose((1, 2, 0))
    weights_mat = np.diag(weights)
    gtg = contract('fcs,st,ftp->fcp', g_tr.conj(), weights_mat, g_ord)
    with open("noise_spec.pk", "rb") as f:
        noise_spec = pickle.load(f)
    noise_spec = lanczos_interpolation(noise_spec, 0, 1, 0, (len(noise_spec) - 1) / (observed_seismograms_freq.shape[1] - 1),
                                       observed_seismograms_freq.shape[1], 8, window="blackmann")
    noise_spec /= noise_spec.max()

    with open("reg_param_indices_new.pk", "rb") as f:
        reg_param_indices = pickle.load(f)
    reg_param = np.logspace(-4, 0)[reg_param_indices[int(name[-4:]) - 1]]

    value_to_add_f = reg_param * np.max(np.abs(np.diagonal(gtg, axis1=1, axis2=2))) * noise_spec
    reg_matrix_per_freq = np.eye(3)[np.newaxis, :, :] * value_to_add_f[:, np.newaxis, np.newaxis]
    inv_term_gg = np.linalg.inv(gtg + reg_matrix_per_freq)
    inv_term_gg_clean = np.linalg.inv(gtg)
    numerator_term = contract('fcs,st->fct', g_tr.conj(), weights_mat)
    g_g_vec = contract('fck,fks->fcs', inv_term_gg, numerator_term)
    g_g_clean_vec = contract('fck,fks->fcs', inv_term_gg_clean, numerator_term)
    mod_res = contract('fcs,fsp->fcp', g_g_vec, g_ord)
    mod_res_clean = contract('fcs,fsp->fcp', g_g_clean_vec, g_ord)
    moments_approx_freq = contract('fcs,fs->fc', g_g_vec, observed_seismograms_freq.T)

    tot_mod_res = np.dot(
        np.mean(np.abs(observed_seismograms_freq), axis=0) / np.sum(np.mean(np.abs(observed_seismograms_freq), axis=0)),
        np.abs(mod_res).reshape(mod_res.shape[0], -1)).reshape(3, 3)
    tot_mod_res_clean = np.dot(
        np.mean(np.abs(observed_seismograms_freq), axis=0) / np.sum(np.mean(np.abs(observed_seismograms_freq), axis=0)),
        np.abs(mod_res_clean).reshape(mod_res_clean.shape[0], -1)).reshape(3, 3)
    moments_approx_freq = moments_approx_freq.transpose()
    moments_approx = np.fft.ifft(moments_approx_freq)
    moments_approx = np.real(moments_approx)

    moments_approx_stream = obspy.Stream()
    for i in range(np.ma.size(moments_approx, 0)):
        moments_approx_stream.append(obspy.Trace(moments_approx[i, :],
                                                 {'sampling_rate': 1 / dt, 'delta': dt, 'network': "ET",
                                                  'station': "source", 'channel': data.channels[i],
                                                  'npts': len(moments_approx[i, :]),
                                                  'starttime': observed_seismograms[0].stats.starttime,
                                                  'endtime': observed_seismograms[0].stats.endtime}))
    moments_approx_stream.taper(type="hann", max_percentage=0.01, side="both")
    moments_approx_stream.filter("lowpass", freq=max_freq, zerophase=True)
    with open("moments_approx_stream_new_" + name[-4:] + ".pk", 'wb') as f:
        pickle.dump(moments_approx_stream, f)


def calc_event_from_green_functions(name, moment_tensor):
    moment_tensor = np.array(moment_tensor)
    moment_tensor /= 10**11
    green_functions = SPECFEM3D_interface.read_green_functions(name)
    for i in range(len(green_functions)):
        green_functions[i].sort(keys=['station'])
    event = obspy.Stream()
    for j in range(len(green_functions[0])):
        event.append(obspy.Trace(data=moment_tensor[0]*green_functions[0][j].data +
                                      moment_tensor[1]*green_functions[1][j].data +
                                      moment_tensor[2]*green_functions[2][j].data,
                                 header=green_functions[0][j].stats))
    return event