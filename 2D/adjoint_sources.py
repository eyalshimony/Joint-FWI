import numpy as np
import obspy
import scipy.io
import re
from scipy.interpolate import CubicSpline
from scipy.spatial import KDTree
import pickle
from scipy.signal import iirfilter, sosfiltfilt, zpk2sos, sosfilt
import os
os.environ['NUMEXPR_MAX_THREADS'] = '64'
import seismograms_handler


def calc_frequencies(t):
    dt = t[1] - t[0]
    f_min = -1/(2*dt)
    df = 1/(len(t)*dt)
    f0 = -f_min
    freqs = np.mod(np.linspace(0, 2*f0-df, len(t))+f0, 2*f0) - f0

    return freqs


def calc_strain_adjoint_source(modelled_seismograms, real_seismograms, min_freq, max_freq, add_envelopes,
                               noise_level_squared, return_misfit_only, GL, ev_num, sim_starttime=0, index=None,
                               is_strain_rate=False, first_iteration=True):
    inds = [ev_num, index]
    if return_misfit_only:
        phase_misfit, envelope_misfit, phase_misfit_next, envelope_misfit_next = calc_adjoint_source(modelled_seismograms, real_seismograms, min_freq, max_freq,
                                                            add_envelopes, noise_level_squared, return_misfit_only,
                                                            sim_starttime, index, is_strain_rate, first_iteration, inds)
        return phase_misfit, envelope_misfit, phase_misfit_next, envelope_misfit_next
    else:
        if index is None:
            phase_sources, envelope_sources, phase_misfit, envelope_misfit, phase_misfit_next, envelope_misfit_next, zero_indices = \
                calc_adjoint_source(modelled_seismograms, real_seismograms, min_freq, max_freq, add_envelopes,
                                    noise_level_squared, return_misfit_only, sim_starttime, index,
                                    is_strain_rate, first_iteration, inds)
        else:
            phase_sources, envelope_sources, phase_misfit, envelope_misfit, phase_misfit_next, envelope_misfit_next, zero_indices, index = \
                calc_adjoint_source(modelled_seismograms, real_seismograms, min_freq, max_freq, add_envelopes,
                                    noise_level_squared, return_misfit_only, sim_starttime, index,
                                    is_strain_rate, first_iteration, inds)
        for i in range(len(phase_sources)):
            phase_sources[i].data /= GL
        for i in range(len(envelope_sources)):
            envelope_sources[i].data /= GL
        if index is None:
            return phase_sources, envelope_sources, phase_misfit, envelope_misfit, phase_misfit_next, envelope_misfit_next, zero_indices
        else:
            return phase_sources, envelope_sources, phase_misfit, envelope_misfit, phase_misfit_next, envelope_misfit_next, zero_indices, index


def find_closest_point(points, target):
    # Calculate the distances from the target point to each point in the list
    distances = np.linalg.norm(points - target, axis=1)
    # Get the index of the point with the minimum distance
    closest_index = np.argmin(distances)
    return closest_index


def create_traces(new_points, reverse_closest_points_map, closest_points_map_dict, fibres_coords_dict,
                  interpolated_fibres_dict, common_header_data, stations, tangents, data_shape):
    new_adjoint_sources_part = obspy.Stream()
    for point in new_points:
        point_tuple = tuple(point)
        closest_point = reverse_closest_points_map[point_tuple]
        point_index = closest_points_map_dict[closest_point].index(point_tuple) - len(
            closest_points_map_dict[closest_point]) // 2
        station_index = fibres_coords_dict[closest_point]
        total_point_index = interpolated_fibres_dict[point_tuple]

        common_header = common_header_data.copy()
        common_header.update({
            'station': stations[station_index],
            'location': f"{point_index};long={point[0]};elev=0;bur={point[1]}"
        })
        for channel in ["X", "Z"]:
            header = common_header.copy()
            header['channel'] = channel
            new_adjoint_sources_part.append(
                obspy.Trace(np.zeros(data_shape, dtype=np.float32), header=header))
    return new_adjoint_sources_part


def create_adjoint_sources_for_injection(adjoint_sources, GL):
    # Distance between line source injection points. 1 m in this case
    ds = 1
    new_adjoint_sources = {}
    stations_names = [adjoint_source.stats.station for adjoint_source in adjoint_sources]
    fibre_names = np.unique([''.join(re.findall(r'[a-zA-Z]', s)) for s in stations_names])
    stations_dict = data.get_stations_dict()

    interpolated_fibres_coords = {}
    interpolated_fibres_tangents = {}
    interpolated_fibres_curvatures = {}
    for fibre_name in fibre_names:
        fibre_adjoint_sources = adjoint_sources.select(station=fibre_name + "[0-9]*").sort(keys=['station'])
        stations = [adjoint_source.stats.station for adjoint_source in fibre_adjoint_sources]
        fibres_coords = np.asarray([[stations_dict[station].longitude,
                                     stations_dict[station].burial] for station in stations])
        x, z = fibres_coords[:, 0], fibres_coords[:, 1]
        cumulative_distances = np.insert(np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(z)**2)), 0, 0)
        spline_x = CubicSpline(cumulative_distances, x)
        spline_z = CubicSpline(cumulative_distances, z)
        new_cumulative_distances = np.arange(0, cumulative_distances[-1] + ds, ds)
        new_points = np.vstack((spline_x(new_cumulative_distances),
                                spline_z(new_cumulative_distances))).T
        tangents = np.gradient(new_points, axis=0)
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
        curvatures = np.gradient(tangents, axis=0) / ds
        interpolated_fibres_tangents[fibre_name] = tangents
        interpolated_fibres_curvatures[fibre_name] = curvatures
        interpolated_fibres_coords[fibre_name] = new_points

        kd_tree = KDTree(fibres_coords)
        closest_points_map = {}
        reverse_closest_points_map = {}
        for interp_point in new_points:
            _, index = kd_tree.query(interp_point)
            closest_point = tuple(fibres_coords[index])
            interp_point_tuple = tuple(interp_point)
            reverse_closest_points_map[interp_point_tuple] = closest_point
            if closest_point not in closest_points_map:
                closest_points_map[closest_point] = []
            closest_points_map[closest_point].append(interp_point_tuple)

        adj_source_stats = adjoint_sources[0].stats
        common_header_data = {
            'sampling_rate': adj_source_stats.sampling_rate,
            'delta': adj_source_stats.delta,
            'network': adj_source_stats.network,
            'npts': adj_source_stats.npts,
            'starttime': adj_source_stats.starttime,
            'endtime': adj_source_stats.endtime,
        }
        fibres_coords_dict = {tuple(pt): idx for idx, pt in enumerate(fibres_coords)}
        interpolated_fibres_dict = {tuple(pt): idx for idx, pt in enumerate(new_points)}
        closest_points_map_dict = {tuple(k): [tuple(p) for p in v] for k, v in closest_points_map.items()}
        data_shape = adjoint_sources[0].data.shape
        new_adjoint_sources[fibre_name] = create_traces(new_points, reverse_closest_points_map, closest_points_map_dict,
                                                        fibres_coords_dict, interpolated_fibres_dict, common_header_data,
                                                        stations, tangents, data_shape)

    for fibre_name, fibre_adjoint_sources in new_adjoint_sources.items():
        original_adjoint_sources = adjoint_sources.select(station=fibre_name + "[0-9]*").sort(keys=['station'])
        new_points = interpolated_fibres_coords[fibre_name]
        tangents = interpolated_fibres_tangents[fibre_name]
        curvatures = interpolated_fibres_curvatures[fibre_name]

        for i, adj_source in enumerate(original_adjoint_sources):
            station = adj_source.stats.station
            station_data = stations_dict[station]
            station_location = [station_data.longitude, station_data.burial]
            station_index = find_closest_point(new_points, station_location)
            boundary_low = station_index - int(np.round(GL / 2 / ds))
            boundary_high = station_index + int(np.round(GL / 2 / ds))
            if boundary_low < 0 or boundary_high > new_points.shape[0] - 1:
                continue
            low_point_index = boundary_low
            high_point_index = boundary_high

            for coord, tangent in enumerate(tangents[low_point_index]):
                new_adjoint_sources[fibre_name][low_point_index * 2 + coord].data += \
                    (-adj_source.data * tangent).astype(np.float32)
            for coord, tangent in enumerate(tangents[high_point_index]):
                new_adjoint_sources[fibre_name][high_point_index * 2 + coord].data += \
                    (adj_source.data * tangent).astype(np.float32)
            for index in range(low_point_index, high_point_index + 1):
                for coord, curvature in enumerate(curvatures[index]):
                    # Adjust adjoint data with curvature
                    adj_data_contribution = -adj_source.data * curvature * ds

                    # Apply halving logic for boundary points
                    if index == boundary_low or index == boundary_high:
                        adj_data_contribution /= 2

                    # Add the contribution to the appropriate point
                    new_adjoint_sources[fibre_name][index * 2 + coord].data += adj_data_contribution.astype(np.float32)

    new_adjoint_sources_combined = obspy.Stream()
    for fibre in new_adjoint_sources.keys():
        new_adjoint_sources_combined += new_adjoint_sources[fibre]
    new_adjoint_sources_combined = seismograms_handler.optimized_resample_stream(new_adjoint_sources_combined,
                                                                                 new_adjoint_sources_combined[0].stats.sampling_rate * 2,
                                                                                 window="blackman")
    return new_adjoint_sources_combined


def calc_adjoint_source(modelled_seismograms, real_seismograms, min_freq, max_freq, add_envelopes,
                        noise_level_squared, return_misfit_only, sim_starttime=0, index=None,
                        is_time_der=True, first_iteration=True, inds=None):
    min_num = 149
    n = 2
    taper_perc = 0.05
    nt = len(modelled_seismograms[0])
    seis_num = len(modelled_seismograms)
    orig_stats = [tr.stats.copy() for tr in modelled_seismograms]
    dt = orig_stats[0].delta
    dt_obs = real_seismograms[0].stats.delta

    modelled_seismograms = modelled_seismograms.copy()
    real_seismograms = real_seismograms.copy()

    timesteps_append_begin = np.max(
        [np.ceil((real_seismograms[0].stats.starttime - modelled_seismograms[0].stats.starttime) / dt_obs), 0])
    timesteps_append_end = np.max(
        [np.ceil((modelled_seismograms[0].stats.endtime - real_seismograms[0].stats.endtime) / dt_obs), 0])
    real_seismograms.trim(starttime=real_seismograms[0].stats.starttime - dt_obs * timesteps_append_begin,
                          endtime=real_seismograms[0].stats.endtime + dt_obs * timesteps_append_end,
                          pad=True, fill_value=0)
    if 0 < np.abs(float(real_seismograms[0].stats.starttime) - float(modelled_seismograms[0].stats.starttime)) < 10**(-5):
        for tr in real_seismograms:
            tr.stats.starttime = modelled_seismograms[0].stats.starttime
    if 0 < np.abs(float(real_seismograms[0].stats.endtime) - float(modelled_seismograms[0].stats.endtime)) < 10**(-5):
        for tr in real_seismograms:
            tr.stats.starttime = modelled_seismograms[0].stats.endtime - \
                                 (real_seismograms[0].stats.npts - 1) * real_seismograms[0].stats.delta
    real_seismograms.interpolate(1/dt, method="lanczos", a=8, window="blackmann",
                                 starttime=modelled_seismograms[0].stats.starttime, npts=modelled_seismograms[0].stats.npts)
    observed_seismograms_arr = seismograms_handler.ordered_stream_into_ordered_matrix(real_seismograms)
    synthetic_seismograms_arr = seismograms_handler.ordered_stream_into_ordered_matrix(modelled_seismograms)

    time_sampling_rate = 1 / dt
    time_nyquist = time_sampling_rate / 2
    time_low = min_freq / time_nyquist
    time_high = max_freq / time_nyquist
    z_time, p_time, k_time = iirfilter(4, [time_low, time_high], btype='bandpass', ftype='butter', output='zpk')
    sos_time = zpk2sos(z_time, p_time, k_time)
    observed_seismograms_arr = sosfiltfilt(sos_time, observed_seismograms_arr, axis=1)
    synthetic_seismograms_arr = sosfiltfilt(sos_time, synthetic_seismograms_arr, axis=1)
    observed_seismograms_arr = seismograms_handler.hann_taper_2d(observed_seismograms_arr, taper_perc, "both", (0, 1))
    synthetic_seismograms_arr = seismograms_handler.hann_taper_2d(synthetic_seismograms_arr, taper_perc, "both", (0, 1))
    f0 = seismograms_handler.central_frequency_unwindowed_2d(observed_seismograms_arr, 1 / dt)

    try:
        dt_new = (observed_seismograms_arr.shape[1] - 1) * dt / (max(np.ceil(nt * dt * 2 * max_freq), min_num) - 1.0)
        observed_seismograms_arr = seismograms_handler.interpolate_2d(observed_seismograms_arr, dt_new, nt,
                                                                      dt, max_freq, min_num)
        synthetic_seismograms_arr = seismograms_handler.interpolate_2d(synthetic_seismograms_arr, dt_new, nt,
                                                                      dt, max_freq, min_num)
    except:
        dt_new = (observed_seismograms_arr.shape[1] - 1) * dt / (
                max(np.ceil(nt * dt * 2 * max_freq), min_num) - 1.0) - 10 ** (-17)
        observed_seismograms_arr = seismograms_handler.interpolate_2d(observed_seismograms_arr, dt_new, nt,
                                                                      dt, max_freq, min_num)
        synthetic_seismograms_arr = seismograms_handler.interpolate_2d(synthetic_seismograms_arr, dt_new, nt,
                                                                       dt, max_freq, min_num)
    new_nt = synthetic_seismograms_arr.shape[0]

    t = np.arange(0, new_nt*dt_new, dt_new)
    width = 1/f0/2
    dom = 1/dt_new/new_nt
    h = (np.pi * width**2)**(-0.25) * np.exp(-(t[:, np.newaxis] - t[np.newaxis, :])**2 / (2*width**2))

    integrand_mod = h[:, :, np.newaxis] * synthetic_seismograms_arr[np.newaxis, :, :]
    integrand_real = h[:, :, np.newaxis] * observed_seismograms_arr[np.newaxis, :, :]
    gabor_mod_seis = dt_new / np.sqrt(2*np.pi) * scipy.fft.fft(integrand_mod, axis=1)
    del integrand_mod
    tmp_gabor_real = scipy.fft.fft(integrand_real, axis=1)
    del integrand_real
    gabor_real_seis = dt_new / np.sqrt(2 * np.pi) * tmp_gabor_real
    cc_transform = gabor_mod_seis * np.conj(tmp_gabor_real)
    del tmp_gabor_real

    seis_phase_diff = np.angle(cc_transform) / np.pi
    del cc_transform
    seis_env_mod = np.abs(gabor_mod_seis)
    seis_env_real = np.abs(gabor_real_seis)
    del gabor_real_seis

    oms = calc_frequencies(np.arange(0, new_nt * dt_new, dt_new))
    thres = np.abs(oms) <= max_freq
    n_lar = np.zeros(np.size(oms))
    n_sm = np.zeros(np.size(oms))
    n_lar[np.logical_not(thres)] = 1
    n_sm[thres] = 1
    me = noise_level_squared * width * 3 / dom / len(oms)
    me = np.max(np.vstack((me, np.max(np.max(seis_env_real**2, axis=0), axis=0)/15)), axis=0)

    if add_envelopes:
        me2 = np.amax(np.amax(seis_env_mod, 0), 0) / 5000
        me3 = np.amax(np.amax((seis_env_mod + seis_env_real)**2, 0), 0) / 500
        me4 = np.amax(np.amax(seis_env_mod + seis_env_real, 0), 0) / 500
        seis_env_norm_diff = 2 * (seis_env_mod - seis_env_real) / \
                             (seis_env_mod + seis_env_real + me4[np.newaxis, np.newaxis, :])

    w_p = 1 - np.exp(-seis_env_real ** 2 / me[np.newaxis, np.newaxis, :])
    w_p_next = w_p.copy()
    if first_iteration:
        w_p[(seis_env_mod / seis_env_mod.max(axis=0).max(axis=0)[np.newaxis, np.newaxis, :] < 0.2)] *= \
            (1 - np.exp(-seis_env_mod**2 / me))[(seis_env_mod / seis_env_mod.max(axis=0).max(axis=0)[np.newaxis, np.newaxis, :] < 0.2)]
        with open(f"/DATA/eyal/specfem2d/mtinv/synthmult_{inds[0]}_{inds[1]}.pk", "wb") as f:
            pickle.dump([(seis_env_mod / seis_env_mod.max(axis=0).max(axis=0)[np.newaxis, np.newaxis, :] < 0.2),
                         (1 - np.exp(-seis_env_mod**2 / me))[(seis_env_mod / seis_env_mod.max(axis=0).max(axis=0)[np.newaxis, np.newaxis, :] < 0.2)]], f)
        with open(f"/DATA/eyal/specfem2d/mtinv/synthmult_curr_{inds[0]}_{inds[1]}.pk", "wb") as f:
            pickle.dump([(seis_env_mod / seis_env_mod.max(axis=0).max(axis=0)[np.newaxis, np.newaxis, :] < 0.2),
                         (1 - np.exp(-seis_env_mod**2 / me))[(seis_env_mod / seis_env_mod.max(axis=0).max(axis=0)[np.newaxis, np.newaxis, :] < 0.2)]], f)
    else:
        with open(f"/DATA/eyal/specfem2d/mtinv/synthmult_{inds[0]}_{inds[1]}.pk", "rb") as f:
            [synthmult_reg, synthmult_vals] = pickle.load(f)
        with open(f"/DATA/eyal/specfem2d/mtinv/synthmult_curr_{inds[0]}_{inds[1]}.pk", "wb") as f:
            pickle.dump([(seis_env_mod / seis_env_mod.max(axis=0).max(axis=0)[np.newaxis, np.newaxis, :] < 0.2),
                         (1 - np.exp(-seis_env_mod**2 / me))[(seis_env_mod / seis_env_mod.max(axis=0).max(axis=0)[np.newaxis, np.newaxis, :] < 0.2)]], f)
        w_p[synthmult_reg] *= synthmult_vals
        w_p_next[(seis_env_mod / seis_env_mod.max(axis=0).max(axis=0)[np.newaxis, np.newaxis, :] < 0.2)] *= \
            (1 - np.exp(-seis_env_mod ** 2 / me))[(seis_env_mod / seis_env_mod.max(axis=0).max(axis=0)[np.newaxis, np.newaxis, :] < 0.2)]

    w_p = np.transpose(w_p, (2, 0, 1))
    w_p_next = np.transpose(w_p_next, (2, 0, 1))
    w_p *= (1 - np.exp(-(oms / min_freq)**2))[np.newaxis, np.newaxis, :]
    w_p_next *= (1 - np.exp(-(oms / min_freq) ** 2))[np.newaxis, np.newaxis, :]
    w_p *= (np.exp(-10*np.abs(np.abs(oms) / max_freq - 1)) * n_lar + n_sm)[np.newaxis, np.newaxis, :]
    w_p_next *= (np.exp(-10*np.abs(np.abs(oms) / max_freq - 1)) * n_lar + n_sm)[np.newaxis, np.newaxis, :]
    w_p = np.transpose(w_p, (1, 2, 0))
    w_p_next = np.transpose(w_p_next, (1, 2, 0))
    div_factor = np.ones((np.ma.size(w_p, 2)))
    div_factor_next = np.ones((np.ma.size(w_p_next, 2)))
    seis_to_change = [np.abs(observed_seismograms_arr[:, i]).max() > np.sqrt(noise_level_squared[i]) * 1.33 for i in range(len(observed_seismograms_arr.T))]
    seis_to_change_2 = [np.abs(observed_seismograms_arr[:, i]).max() < np.sqrt(noise_level_squared[i]) * 0.95 for i in
                      range(len(observed_seismograms_arr.T))]
    div_factor[seis_to_change] = \
        1 / np.max(np.max(w_p[:, :, seis_to_change], axis=0), axis=0)
    div_factor_next[seis_to_change] = \
        1 / np.max(np.max(w_p_next[:, :, seis_to_change], axis=0), axis=0)
    div_factor[div_factor > 2] = 2
    div_factor_next[div_factor_next > 2] = 2
    div_factor[seis_to_change_2] = 0
    div_factor_next[seis_to_change_2] = 0
    w_p *= div_factor
    w_p_next *= div_factor_next
    if first_iteration:
        w_p_next = w_p.copy()

    test_field = seis_phase_diff * w_p_next
    indices_to_zero = np.where(np.sum(np.sum(np.abs(np.diff(test_field, axis=0)) > 1.0, axis=0), axis=0) +
                               np.sum(np.sum(np.abs(np.diff(test_field, axis=1)) > 1.0, axis=0), axis=0) > 12)
    del test_field

    w_phase_difference = (seis_phase_diff * w_p) ** n
    w_phase_difference_next = (seis_phase_diff * w_p_next) ** n
    phase_misfit = np.sqrt(np.sum(np.sum(w_phase_difference * dt_new * dom, axis=0), axis=0))
    phase_misfit_next = np.sqrt(np.sum(np.sum(w_phase_difference_next * dt_new * dom, axis=0), axis=0))
    del w_phase_difference, w_phase_difference_next
    if not add_envelopes:
        del w_p_next
    phase_sources = obspy.Stream()
    mp = np.amax(np.amax(seis_env_mod**2, 0), 0) / 500
    envelope_misfit = np.zeros(np.size(phase_misfit))
    envelope_misfit_next = np.zeros(np.size(phase_misfit))
    envelope_sources = obspy.Stream()

    if add_envelopes:
        w_envelope_difference = (seis_env_norm_diff * w_p) ** n
        w_envelope_difference_next = (seis_env_norm_diff * w_p_next) ** n
        del w_p_next
        envelope_misfit = np.sqrt(np.sum(np.sum(w_envelope_difference * dt_new * dom, axis=0), axis=0))
        envelope_misfit_next = np.sqrt(np.sum(np.sum(w_envelope_difference_next * dt_new * dom, axis=0), axis=0))
        del w_envelope_difference_next

    if return_misfit_only:
        return phase_misfit, envelope_misfit, phase_misfit_next, envelope_misfit_next

    for i in range(seis_num):
        integrand = 1/np.pi * w_p[:, :, i]**n * seis_phase_diff[:, :, i]**(n-1) * gabor_mod_seis[:, :, i] / \
                    (seis_env_mod[:, :, i]**2 + mp[i])
        i_integrand = np.fft.ifft(integrand)
        i_integrand = 2 * np.pi / dt_new * i_integrand
        i_integrand = h * i_integrand
        phase_source = dt_new / np.sqrt(2*np.pi) * np.sum(i_integrand * dt_new, 0)
        phase_source = np.imag(phase_source)
        if phase_misfit[i] == 0:
            phase_source[:] = 0
        else:
            phase_source = phase_source / phase_misfit[i]
        source_stats = {'network': orig_stats[i].network, 'station': orig_stats[i].station,
                        'location': orig_stats[i].location, 'channel': orig_stats[i].channel,
                        'sampling_rate': 1/dt_new, 'delta': dt_new, 'npts': new_nt,
                        'starttime': orig_stats[i].starttime, 'endtime': orig_stats[i].endtime}
        phase_sources.append(obspy.Trace(phase_source, source_stats))
    del seis_phase_diff

    try:
        phase_sources = phase_sources.copy().interpolate(1 / dt, method="lanczos", a=8, window="blackmann",
                                                         npts=orig_stats[0].npts)
    except ValueError:
        phase_sources.trim(starttime=phase_sources[0].stats.starttime,
                           endtime=phase_sources[0].stats.endtime+phase_sources[0].stats.delta, pad=True, fill_value=0)
        phase_sources = phase_sources.copy().interpolate(1 / dt, method="lanczos", a=8, window="blackmann",
                                                         npts=orig_stats[0].npts)
    if is_time_der:
        phase_sources.differentiate()
        for i in range(seis_num):
            phase_sources[i].data = -phase_sources[i].data
    phase_sources = phase_sources.taper(type="hann", max_percentage=taper_perc, side="both")

    if add_envelopes:
        for i in range(seis_num):
            integrand = 4*w_p[:, :, i]**n * seis_env_norm_diff[:, :, i]**(n-1) * gabor_mod_seis[:, :, i] * \
                        seis_env_real[:, :, i] / (seis_env_mod[:, :, i] + me2[i]) / \
                        ((seis_env_mod[:, :, i] + seis_env_real[:, :, i])**2 + me3[i])
            i_integrand = scipy.fft.ifft(integrand)
            i_integrand = 2*np.pi/dt_new * i_integrand
            i_integrand = h * i_integrand
            envelope_source = dt_new / np.sqrt(2*np.pi) * np.sum(i_integrand * dt_new, 0)
            envelope_source = np.real(envelope_source)
            if envelope_misfit[i] == 0:
                envelope_source[:] = 0
            else:
                envelope_source = -envelope_source / envelope_misfit[i]
            source_stats = {'network': orig_stats[i].network, 'station': orig_stats[i].station,
                            'location': orig_stats[i].location, 'channel': orig_stats[i].channel,
                            'sampling_rate': 1 / dt_new, 'delta': dt_new, 'npts': new_nt,
                            'starttime': orig_stats[i].starttime, 'endtime': orig_stats[i].endtime}
            envelope_sources.append(obspy.Trace(envelope_source, source_stats))
        try:
            envelope_sources = envelope_sources.copy().interpolate(1 / dt, method="lanczos", a=8, window="blackmann",
                                                                   npts=orig_stats[0].npts)
        except ValueError:
            envelope_sources.trim(starttime=envelope_sources[0].stats.starttime,
                                  endtime=envelope_sources[0].stats.endtime + envelope_sources[0].stats.delta, pad=True,
                                  fill_value=0)
            envelope_sources = envelope_sources.copy().interpolate(1 / dt, method="lanczos", a=8, window="blackmann",
                                                                   npts=orig_stats[0].npts)
        if is_time_der:
            envelope_sources.differentiate()
            for i in range(seis_num):
                envelope_sources[i].data = -envelope_sources[i].data
        envelope_sources = envelope_sources.taper(type="hann", max_percentage=taper_perc, side="both")
        for source in envelope_sources:
            time_diff = sim_starttime - float(source.stats.starttime)
            source.stats.starttime += time_diff

    for source in phase_sources:
        time_diff = sim_starttime - float(source.stats.starttime)
        source.stats.starttime += time_diff

    if index is None:
        return phase_sources, envelope_sources, phase_misfit, envelope_misfit, phase_misfit_next, envelope_misfit_next,\
               indices_to_zero[0]
    else:
        return phase_sources, envelope_sources, phase_misfit, envelope_misfit, phase_misfit_next, envelope_misfit_next,\
               indices_to_zero[0], index