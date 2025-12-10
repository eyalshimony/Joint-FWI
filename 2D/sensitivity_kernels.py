import adjoint_sources
import SPECFEM3D_interface
import obspy
import numpy as np
import seismograms_handler
import concurrent.futures
import pickle
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage.filters import gaussian_filter
import data


def calculate_moment_tensor_kernels_xyz(adjoint_strain_tensor, stf):
    moment_tensor_kernel = {}
    stf_data = stf.data
    for trace in adjoint_strain_tensor:
        dt = trace.stats.delta
        moment_tensor_kernel[trace.stats.channel] = dt * np.dot(trace.data, stf_data)

    return moment_tensor_kernel


def calc_onset_time_kernel_xyz(adjoint_strain_tensor, solution, stf):
    dt = adjoint_strain_tensor[0].stats.delta
    stf_der = stf.differentiate()
    stf_der_data = stf_der.data

    return -dt * np.dot((solution.Mzz * adjoint_strain_tensor.select(channel="ZZ")[0].data +
                        solution.Mxx * adjoint_strain_tensor.select(channel="XX")[0].data +
                        2 * (solution.Mxz * adjoint_strain_tensor.select(channel="XZ")[0].data)), stf_der_data)


def moment_tensor_kernel_dict_to_CMTSOLUTION_diff_xyz(moment_tensor_kernel_dict):
    return [moment_tensor_kernel_dict["ZZ"], moment_tensor_kernel_dict["XX"], moment_tensor_kernel_dict["XZ"]]


def interpolate_kernels(kernels, model, hessian=None):
    vp_interpolator = NearestNDInterpolator(kernels[:, :2], kernels[:, 3])
    vs_interpolator = NearestNDInterpolator(kernels[:, :2], kernels[:, 4])
    rho_interpolator = NearestNDInterpolator(kernels[:, :2], kernels[:, 2])
    if hessian is not None:
        hess_interpolator = NearestNDInterpolator(hessian[:, :2], hessian[:, 2])
    vp_interpolated_mesh = vp_interpolator(model[:, 0:2])
    vs_interpolated_mesh = vs_interpolator(model[:, 0:2])
    rho_interpolated_mesh = rho_interpolator(model[:, 0:2])
    if hessian is not None:
        hess_interpolated_mesh = hess_interpolator(model[:, 0:2])
    points = model[:, 0:2]
    if hessian is None:
        return vp_interpolated_mesh, vs_interpolated_mesh, rho_interpolated_mesh, points
    else:
        return vp_interpolated_mesh, vs_interpolated_mesh, rho_interpolated_mesh, points, hess_interpolated_mesh


def sum_interpolated_kernels_and_models_multi(vp_kernel_mesh, vs_kernel_mesh, rho_kernel_mesh, model, frac, name,
                                              it_num=-1):
    while True:
        current_model = np.zeros(model.shape)
        current_model[:, :2] = model[:, :2]
        current_model[:, 2] = model[:, 2] + np.squeeze(frac * vp_kernel_mesh)
        current_model[:, 3] = model[:, 3] + np.squeeze(frac * vs_kernel_mesh)
        current_model[:, 4] = model[:, 4] + np.squeeze(frac * rho_kernel_mesh)
        if current_model[:, 2:5].min() <= 0:
            frac /= 2
        else:
            break
    if it_num >= 0:
        SPECFEM3D_interface.write_raw_model(current_model, name, it_num)
    else:
        SPECFEM3D_interface.write_raw_model(current_model, name)
    return current_model, frac


def calculate_adjoint_sources_for_one_source_1D(source_name, solution, min_freq, max_freq, add_envelopes, noise_level,
                                                first_iteration):
    observed_seismograms = SPECFEM3D_interface.read_observed_seismograms_from_pickle(source_name)
    SPECFEM3D_interface.copy_source_time_function(source_name)
    directional_seismograms = SPECFEM3D_interface.read_SU_seismograms(source_name, "r", solution.tshift)
    synthetic_seismograms = seismograms_handler.calculate_DAS_seismograms(directional_seismograms)
    observed_seismograms.sort(keys=['station', 'channel'])
    synthetic_seismograms.sort(keys=['station', 'channel'])
    noise_level = noise_level ** 2
    noise_level_squared = np.array([noise_level] * len(observed_seismograms))
    zs = np.arange(0, 2001, 5)
    vert_prof = np.exp(-0.4 * zs / 125.0)
    noise_level_squared[:401] *= vert_prof
    noise_level_squared[401:802] *= vert_prof
    observed_seismograms_lists = seismograms_handler.split_stream_into_substreams(observed_seismograms, 8)
    synthetic_seismograms_lists = seismograms_handler.split_stream_into_substreams(synthetic_seismograms, 8)
    noise_level_squared_lists = seismograms_handler.split_stream_into_substreams(noise_level_squared, 8)
    phase_sources = obspy.Stream()
    envelope_sources = obspy.Stream()
    phase_misfit = []
    envelope_misfit = []
    phase_misfit_next = []
    envelope_misfit_next = []
    zero_indices = []
    list_indices = list(range(len(observed_seismograms_lists)))
    ev_num = int(source_name[-4:])
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for i in list_indices:
            futures.append(executor.submit(adjoint_sources.calc_strain_adjoint_source, synthetic_seismograms_lists[i],
                                           observed_seismograms_lists[i], min_freq, max_freq, add_envelopes,
                                           noise_level_squared_lists[i], False, data.GL, ev_num, -2/solution.f0, i,
                                           True, first_iteration))
        results = [future.result() for future in futures]
        for retvals in results:
            phase_sources += retvals[0]
            envelope_sources += retvals[1]
            phase_misfit += list(retvals[2])
            envelope_misfit += list(retvals[3])
            phase_misfit_next += list(retvals[4])
            envelope_misfit_next += list(retvals[5])
            zero_indices += list(retvals[6] + len(observed_seismograms_lists[0]) * retvals[7])
    return phase_sources, envelope_sources, phase_misfit, envelope_misfit, phase_misfit_next, envelope_misfit_next, zero_indices


def calc_inj_adjsrc_and_write_one_source(phase_sources, envelope_sources, zero_one_indices, source_name, add_envelopes):
    GL = data.GL
    curr_adjoint_sources = obspy.Stream()
    phase_sources.sort(keys=['station', 'channel'])
    envelope_sources.sort(keys=['station', 'channel'])
    for j in range(len(phase_sources)):
        if zero_one_indices[j] == 0:
            curr_adjoint_sources.append(obspy.Trace(np.zeros(phase_sources[j].data.shape), phase_sources[j].stats))
            continue
        if add_envelopes:
            curr_adjoint_sources.append(obspy.Trace(-np.add(phase_sources[j].data, envelope_sources[j].data),
                                                    phase_sources[j].stats))
        else:
            curr_adjoint_sources.append(obspy.Trace(-phase_sources[j].data, phase_sources[j].stats))
    adjoint_sources_for_injection = adjoint_sources.create_adjoint_sources_for_injection(curr_adjoint_sources, GL)
    SPECFEM3D_interface.write_adjoint_sources_to_SU_files(adjoint_sources_for_injection, source_name)


def calc_all_derivatives_multi(source_names, solutions, min_freq, max_freq, add_envelopes,
                               check_int_wolfe=False, orig_misfit=None, step_length=None, orig_grad=None, orig_sd=None,
                               first_it=None, prev_misfit=None, zoom_mode=None, low_misfit=None,
                               zero_indices_multi_to_use=None):
    for i in range(len(source_names)):
        SPECFEM3D_interface.write_source_file(solutions[i], source_names[i])

    SPECFEM3D_interface.run_modelling_for_FWI_multi(source_names)
    phase_sources_multi = []
    envelope_sources_multi = []
    phase_misfit_multi = []
    envelope_misfit_multi = []
    phase_misfit_next_multi = []
    envelope_misfit_next_multi = []
    zero_indices_multi = []
    with open("noise_levels.pk", "rb") as f:
        noise_levels = pickle.load(f)
    with open("noise_spec.pk", "rb") as f:
        noise_spec = pickle.load(f)
    noise_spec = noise_spec[:int(len(noise_spec) / 2)]
    noise_spec /= np.sum(noise_spec)
    freqs = np.fft.fftfreq(7500, 0.0016)
    freqs = freqs[:int(len(noise_spec) / 2)]
    noise_levels = 2 * np.array(noise_levels) * np.sqrt(np.sum(noise_spec[:np.searchsorted(freqs, max_freq) + 1]**2) / np.sum(noise_spec**2))

    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        futures = []
        for i in range(len(source_names)):
            futures.append(executor.submit(calculate_adjoint_sources_for_one_source_1D, source_names[i], solutions[i],
                                           min_freq, max_freq, add_envelopes, noise_levels[i],
                                           zero_indices_multi_to_use is None))
        results = [future.result() for future in futures]
        for retvals in results:
            phase_sources_multi.append(retvals[0])
            envelope_sources_multi.append(retvals[1])
            phase_misfit_multi.append(retvals[2])
            envelope_misfit_multi.append(retvals[3])
            phase_misfit_next_multi.append(retvals[4])
            envelope_misfit_next_multi.append(retvals[5])
            zero_indices_multi.append(retvals[6])

    zero_one_indices_new = np.ones(np.asarray(phase_misfit_multi).shape)
    for i in range(len(zero_one_indices_new)):
        zero_one_indices_new[i][zero_indices_multi[i]] = 0

    if zero_indices_multi_to_use is None:
        zero_one_indices = zero_one_indices_new
    else:
        zero_one_indices = zero_indices_multi_to_use

    print(np.sum(np.multiply(phase_misfit_multi, zero_one_indices)) +
          np.sum(np.multiply(envelope_misfit_multi, zero_one_indices)))
    if check_int_wolfe:
        if not zoom_mode and ((np.sum(np.multiply(phase_misfit_multi, zero_one_indices)) +
                               np.sum(np.multiply(envelope_misfit_multi, zero_one_indices))) >
                              (orig_misfit + 10 ** (-4) * step_length * np.dot(orig_sd, orig_grad)) or
                              (not first_it and (np.sum(np.multiply(phase_misfit_multi, zero_one_indices)) +
                                                 np.sum(np.multiply(envelope_misfit_multi, zero_one_indices))) > prev_misfit)):
            return None, None, None, phase_misfit_multi, envelope_misfit_multi, None, None, zero_one_indices
        if zoom_mode and ((np.sum(np.multiply(phase_misfit_multi, zero_one_indices)) +
                           np.sum(np.multiply(envelope_misfit_multi, zero_one_indices))) >
                          (orig_misfit + 10 ** (-4) * step_length * np.dot(orig_sd, orig_grad)) or
                          (np.sum(np.multiply(phase_misfit_multi, zero_one_indices)) +
                           np.sum(np.multiply(envelope_misfit_multi, zero_one_indices))) >= low_misfit):
            return None, None, None, phase_misfit_multi, envelope_misfit_multi, None, None, zero_one_indices
    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        futures = []
        for i in range(len(source_names)):
            futures.append(executor.submit(calc_inj_adjsrc_and_write_one_source, phase_sources_multi[i],
                                           envelope_sources_multi[i], zero_one_indices[i], source_names[i],
                                           add_envelopes))
    SPECFEM3D_interface.run_structural_adjoint_modelling_multi(source_names)

    moment_tensor_kernels = []
    source_location_kernels = []
    onset_time_kernels = []

    for i in range(len(source_names)):
        adjoint_strain_tensor = SPECFEM3D_interface.read_adjoint_strain_tensor_at_source(source_names[i])
        stf = SPECFEM3D_interface.read_source_time_function(source_names[i])
        stf.data = stf.data[1::2]
        stf.stats.delta *= 2
        moment_tensor_kernels.append(calculate_moment_tensor_kernels_xyz(adjoint_strain_tensor, stf))
        source_location_kernels.append(SPECFEM3D_interface.read_source_location_kernels(source_names[i]))
        onset_time_kernels.append(calc_onset_time_kernel_xyz(adjoint_strain_tensor, solutions[i], stf))

    return moment_tensor_kernels, source_location_kernels, onset_time_kernels, phase_misfit_multi, \
           envelope_misfit_multi, phase_misfit_next_multi, envelope_misfit_next_multi, zero_one_indices_new


def smooth_q(q, mesh_size, model, smoothing_size):
    q_parts = [q[:mesh_size], q[mesh_size:2 * mesh_size], q[2 * mesh_size:3 * mesh_size]]
    smooth_q = []
    x_num = len(np.unique(model[:, 0]))
    z_num = len(np.unique(model[:, 1]))
    for q_part in q_parts:
        q_part_whole_3d = np.reshape(q_part, (x_num, z_num), order='F')
        q_part_smooth_3d = gaussian_filter(q_part_whole_3d, smoothing_size)
        q_part_smooth = q_part_smooth_3d.ravel(order='F')
        smooth_q.extend(q_part_smooth)
    smooth_q.extend(q[3 * mesh_size:])
    return np.asarray(smooth_q)


def kernels_to_gradients(vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh, hess_kernel_int_mesh=None):
    vp_kernel_int_mesh *= (data.dx*data.dz)
    vs_kernel_int_mesh *= (data.dx*data.dz)
    rho_kernel_int_mesh *= (data.dx*data.dz)
    if hess_kernel_int_mesh is not None:
        hess_kernel_int_mesh *= (data.dx * data.dz)
    if hess_kernel_int_mesh is None:
        return vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh
    else:
        return vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh, hess_kernel_int_mesh


def get_structural_kernels_multi(kernel_types, source_name, orig_model, k=None):
    kernels = SPECFEM3D_interface.get_kernels(source_name)
    if len(kernel_types) == 4:
        hessians = SPECFEM3D_interface.get_hessians(source_name)
        if k is not None:
            hessian = hessians[k]
        else:
            hessian = average_kernels(hessians)

    if k is not None:
        kernels = kernels[k]
    else:
        kernels = average_kernels(kernels)

    if len(kernel_types) == 3:
        vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh, points = \
            interpolate_kernels(kernels, orig_model)
        vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh = \
            kernels_to_gradients(vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh)
        return vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh, points
    elif len(kernel_types) == 4:
        vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh, points, hess_kernel_int_mesh = \
            interpolate_kernels(kernels, orig_model, hessian)
        vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh, hess_kernel_int_mesh = \
            kernels_to_gradients(vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh, hess_kernel_int_mesh)
        return vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh, points, hess_kernel_int_mesh
    else:
        raise ValueError("Only 3 or 4 kernels are supported")


def average_kernels(kernels_list):
    return np.mean(np.stack(kernels_list), axis=0)


def clip_kernels(kernels_vec, mesh_size):
    kernels_vec[:mesh_size][kernels_vec[:mesh_size] > np.percentile(kernels_vec[:mesh_size], 99)] = \
        np.percentile(kernels_vec[:mesh_size], 99)
    kernels_vec[:mesh_size][kernels_vec[:mesh_size] < np.percentile(kernels_vec[:mesh_size], 1)] = \
        np.percentile(kernels_vec[:mesh_size], 1)
    kernels_vec[mesh_size:mesh_size*2][kernels_vec[mesh_size:mesh_size*2] > np.percentile(kernels_vec[mesh_size:mesh_size*2], 99)] = \
        np.percentile(kernels_vec[mesh_size:mesh_size*2], 99)
    kernels_vec[mesh_size:mesh_size*2][kernels_vec[mesh_size:mesh_size*2] < np.percentile(kernels_vec[mesh_size:mesh_size*2], 1)] = \
        np.percentile(kernels_vec[mesh_size:mesh_size*2], 1)
    kernels_vec[mesh_size*2:mesh_size * 3][
        kernels_vec[mesh_size*2:mesh_size * 3] > np.percentile(kernels_vec[mesh_size*2:mesh_size * 3], 99)] = \
        np.percentile(kernels_vec[mesh_size*2:mesh_size * 3], 99)
    kernels_vec[mesh_size*2:mesh_size * 3][
        kernels_vec[mesh_size*2:mesh_size * 3] < np.percentile(kernels_vec[mesh_size*2:mesh_size * 3], 1)] = \
        np.percentile(kernels_vec[mesh_size*2:mesh_size * 3], 1)


def calculate_search_direction_multi(source_names, source_location_grads, moment_tensor_kernel_dicts, onset_time_grads,
                                     smoothing_size, iteration, ss=None, ys=None, rhos=None, previous_kernels=None,
                                     LBFGS_mem_size=None):
    iteration -= 387
    event_num = len(source_names)
    moment_tensor_kernel = []
    for moment_tensor_kernel_dict in moment_tensor_kernel_dicts:
        moment_tensor_kernel.extend(moment_tensor_kernel_dict_to_CMTSOLUTION_diff_xyz(moment_tensor_kernel_dict))
    orig_model = SPECFEM3D_interface.read_tomographic_models(source_names[0], -1)
    kernel_types = ['beta', 'alpha', 'rho']
    if iteration == 1:
        kernel_types.append('hess')
        vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh, points, hess_kernel_int_mesh = \
            get_structural_kernels_multi(kernel_types, source_names[0].split("/")[0], orig_model)
    else:
        vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh, points = \
            get_structural_kernels_multi(kernel_types, source_names[0].split("/")[0], orig_model)

    kernels_vec = np.concatenate([vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh]).flatten()
    mesh_size = int(kernels_vec.size / 3)
    kernels_vec = np.append(kernels_vec, source_location_grads)
    kernels_vec = np.append(kernels_vec, moment_tensor_kernel)
    kernels_vec = np.append(kernels_vec, onset_time_grads)
    raw_kernels_vec = kernels_vec.copy()

    if iteration > 1:
        y = kernels_vec - previous_kernels
        if len(ys) >= LBFGS_mem_size:
            ys.pop(0)
            rhos.pop(0)
        ys.append(y)
        rhos.append(1 / np.dot(ys[-1], ss[-1]))

    clip_kernels(kernels_vec, mesh_size)
    q = kernels_vec.copy()
    if iteration == 1:
        hess_vec = hess_kernel_int_mesh.flatten()
        hess_vec = np.abs(hess_vec)
        hess_vec += 10 ** (-2) * hess_vec.max()
        hess_vec = np.tile(hess_vec, 3)
        hess_vec[:mesh_size] /= 240000
        hess_vec[mesh_size:2 * mesh_size] /= 48000
        hess_vec[2 * mesh_size:] /= 13000
        hess_vec = np.append(hess_vec, [4 / (10 ** 3), 4 / (10 ** 3)] * event_num)
        hess_vec = np.append(hess_vec,
                             [6 / (10 ** 21), 6 / (10 ** 21), 6 / (10 ** 21)] * event_num)
        hess_vec = np.append(hess_vec, [1 / (10 ** (-5))] * event_num)

    if iteration == 1:
        r = -smooth_q(q / hess_vec, mesh_size, orig_model, smoothing_size)
    else:
        alphas = np.zeros(len(ss))
        for i in range(len(ss)):
            alphas[len(ss) - i - 1] = rhos[len(ss) - i - 1] * np.dot(ss[len(ss) - i - 1], q)
            q -= alphas[len(ss) - i - 1] * ys[len(ss) - i - 1]
        ss_mat = np.asarray(ss)
        ys_mat = np.asarray(ys)
        last_mem = -6 * event_num
        value_to_add_struc = np.sum(ys_mat[:, :last_mem] ** 2, axis=0).max() / 10**6
        y_square_summed = np.sum(ys_mat ** 2, axis=0)
        y_square_summed[y_square_summed == 0] = value_to_add_struc
        est_hess = np.abs(np.sum(ss_mat * ys_mat, axis=0) / y_square_summed)
        gamma_vp = np.dot(ys[-1][:mesh_size], ss[-1][:mesh_size]) / np.dot(ys[-1][:mesh_size], ys[-1][:mesh_size])
        gamma_vs = np.dot(ys[-1][mesh_size:2*mesh_size], ss[-1][mesh_size:2*mesh_size]) / \
                   np.dot(ys[-1][mesh_size:2*mesh_size], ys[-1][mesh_size:2*mesh_size])
        gamma_rho = np.dot(ys[-1][2*mesh_size:3*mesh_size], ss[-1][2*mesh_size:3*mesh_size]) / \
                    np.dot(ys[-1][2*mesh_size:3*mesh_size], ys[-1][2*mesh_size:3*mesh_size])
        est_hess[:mesh_size] = gamma_vp
        est_hess[mesh_size:2*mesh_size] = gamma_vs
        est_hess[2*mesh_size:3 * mesh_size] = gamma_rho
        gamma_loc = np.dot(ys[-1][last_mem:last_mem+2*event_num],
                           ss[-1][last_mem:last_mem+2*event_num]) / np.dot(ys[-1][last_mem:last_mem+2*event_num],
                                                                           ys[-1][last_mem:last_mem+2*event_num])
        gamma_mom = np.dot(ys[-1][last_mem + 2 * event_num:last_mem + 5 * event_num],
                           ss[-1][last_mem + 2 * event_num:last_mem + 5 * event_num]) / np.dot(
            ys[-1][last_mem + 2 * event_num:last_mem + 5 * event_num],
            ys[-1][last_mem + 2 * event_num:last_mem + 5 * event_num])
        gamma_t0 = np.dot(ys[-1][last_mem + 5 * event_num:], ss[-1][last_mem + 5 * event_num:]) / np.dot(
            ys[-1][last_mem + 5 * event_num:], ys[-1][last_mem + 5 * event_num:])
        est_hess[last_mem:last_mem+2*event_num][est_hess[last_mem:last_mem+2*event_num] > gamma_loc * 10] = gamma_loc * 10
        est_hess[last_mem:last_mem+2*event_num][est_hess[last_mem:last_mem+2*event_num] < gamma_loc / 10] = gamma_loc / 10
        est_hess[last_mem + 2 * event_num:last_mem + 5 * event_num][
            est_hess[last_mem + 2 * event_num:last_mem + 5 * event_num] > gamma_mom * 10] = gamma_mom * 10
        est_hess[last_mem + 2 * event_num:last_mem + 5 * event_num][
            est_hess[last_mem + 2 * event_num:last_mem + 5 * event_num] < gamma_mom / 10] = gamma_mom / 10
        est_hess[last_mem + 5 * event_num:][est_hess[last_mem + 5 * event_num:] > gamma_t0 * 10] = gamma_t0 * 10
        est_hess[last_mem + 5 * event_num:][est_hess[last_mem + 5 * event_num:] < gamma_t0 / 10] = gamma_t0 / 10
        r = smooth_q(q * est_hess, mesh_size, orig_model, smoothing_size)
        for i in range(len(ss)):
            beta = rhos[i] * np.dot(ys[i], r)
            r += (alphas[i] - beta) * ss[i]
        r = -r

    alpha_search_direction = r[0:mesh_size]
    beta_search_direction = r[mesh_size:2 * mesh_size]
    rho_search_direction = r[2 * mesh_size:3 * mesh_size]
    SPECFEM3D_interface.write_kernel(source_names[0].split("/")[0], alpha_search_direction, "vp")
    SPECFEM3D_interface.write_kernel(source_names[0].split("/")[0], beta_search_direction, "vs")
    SPECFEM3D_interface.write_kernel(source_names[0].split("/")[0], rho_search_direction, "rho")
    source_location_grad = r[3 * mesh_size:3 * mesh_size + 2 * event_num]
    moment_tensor_grad = r[3 * mesh_size + 2 * event_num:3 * mesh_size + 5 * event_num]
    onset_time_grad = r[-event_num:]

    return alpha_search_direction, beta_search_direction, rho_search_direction, source_location_grad, \
           moment_tensor_grad, onset_time_grad, raw_kernels_vec