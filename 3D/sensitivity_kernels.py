import adjoint_sources
import SPECFEM3D_interface
from objects import SolutionType
import obspy
import numpy as np
import seismograms_handler
import concurrent.futures
from itertools import repeat
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage.filters import gaussian_filter
import meshio
import copy

# Maybe move it somewhere else later
dx = 62.5
dy = 62.5
dz = 62.5


def calculate_moment_tensor_kernels_xyz(adjoint_strain_tensor, stf):
    moment_tensor_kernel = {}
    stf_data = np.flip(stf.data)
    for trace in adjoint_strain_tensor:
        dt = trace.stats.delta
        moment_tensor_kernel[trace.stats.channel] = dt * np.dot(trace.data, stf_data)

    moment_tensor_kernel["YZ"] = -moment_tensor_kernel["YZ"]
    moment_tensor_kernel["XY"] = -moment_tensor_kernel["XY"]

    return moment_tensor_kernel


def calc_onset_time_kernel_xyz(adjoint_strain_tensor, solution, stf):
    dt = adjoint_strain_tensor[0].stats.delta
    stf_der = stf.differentiate()
    stf_der_data = np.flip(stf_der.data)

    return -dt * np.dot((solution.Mrr * adjoint_strain_tensor.select(channel="ZZ")[0].data +
                        solution.Mtt * adjoint_strain_tensor.select(channel="YY")[0].data +
                        solution.Mpp * adjoint_strain_tensor.select(channel="XX")[0].data +
                        2 * (-solution.Mrt * adjoint_strain_tensor.select(channel="YZ")[0].data +
                             solution.Mrp * adjoint_strain_tensor.select(channel="XZ")[0].data +
                             -solution.Mtp * adjoint_strain_tensor.select(channel="XY")[0].data)) / 10**7, stf_der_data)


def moment_tensor_kernel_dict_to_CMTSOLUTION_diff_xyz(moment_tensor_kernel_dict):
    return [moment_tensor_kernel_dict["ZZ"], moment_tensor_kernel_dict["YY"], moment_tensor_kernel_dict["XX"],
            moment_tensor_kernel_dict["YZ"], moment_tensor_kernel_dict["XZ"], moment_tensor_kernel_dict["XY"]]


def interpolate_kernels(vp_kernel_mesh, vs_kernel_mesh, rho_kernel_mesh, models, hess_kernel_mesh=None):
    vp_interpolator = NearestNDInterpolator(vp_kernel_mesh.points, vp_kernel_mesh.point_data['alpha_kernel'])
    vs_interpolator = NearestNDInterpolator(vs_kernel_mesh.points, vs_kernel_mesh.point_data['beta_kernel'])
    rho_interpolator = NearestNDInterpolator(rho_kernel_mesh.points, rho_kernel_mesh.point_data['rho_kernel'])
    if hess_kernel_mesh is not None:
        hess_interpolator = NearestNDInterpolator(hess_kernel_mesh.points, hess_kernel_mesh.point_data['hess_kernel'])
        hess_interpolated_meshes = []
    vp_interpolated_meshes = []
    vs_interpolated_meshes = []
    rho_interpolated_meshes = []
    points = []
    for model in models:
        vp_interpolated_meshes.append(vp_interpolator(model[:, 0:3]))
        vs_interpolated_meshes.append(vs_interpolator(model[:, 0:3]))
        rho_interpolated_meshes.append(rho_interpolator(model[:, 0:3]))
        if hess_kernel_mesh is not None:
            hess_interpolated_meshes.append(hess_interpolator(model[:, 0:3]))
        points.append(model[:, 0:3])
    if hess_kernel_mesh is None:
        return vp_interpolated_meshes, vs_interpolated_meshes, rho_interpolated_meshes, points
    else:
        return vp_interpolated_meshes, vs_interpolated_meshes, rho_interpolated_meshes, points, hess_interpolated_meshes


def sum_interpolated_kernels_and_models_multi(vp_kernel_mesh, vs_kernel_mesh, rho_kernel_mesh, models, frac, name,
                                              source_num=0, it_num=-1):
    current_models = []
    for j in range(len(models)):
        current_model = np.zeros(models[j].shape)
        current_model[:, 0:3] = models[j][:, 0:3]
        current_model[:, 3] = models[j][:, 3] + np.squeeze(frac * vp_kernel_mesh[j])
        current_model[:, 4] = models[j][:, 4] + np.squeeze(frac * vs_kernel_mesh[j])
        current_model[:, 5] = models[j][:, 5] + np.squeeze(frac * rho_kernel_mesh[j])
        current_models.append(current_model)
        if it_num >= 0:
            SPECFEM3D_interface.write_raw_model(current_model, name, j, it_num)
        else:
            if source_num == 0:
                raise ValueError("source_num cannot be 0")
            SPECFEM3D_interface.write_raw_model(current_model, name, j)
            SPECFEM3D_interface.copy_model_to_runs(name, source_num, j)
    return current_models


def calc_all_derivatives_multi(source_names, solutions, solution_types, min_freq, max_freq, add_envelopes,
                               check_int_wolfe=False, orig_misfit=None, step_length=None, orig_grad=None, orig_sd=None,
                               first_it=None, prev_misfit=None, zoom_mode=None, low_misfit=None,
                               zero_indices_multi_to_use=None):
    for source_name in source_names:
        if not SPECFEM3D_interface.check_event_existence(source_name):
            SPECFEM3D_interface.copy_from_template(source_name)

    for i in range(len(source_names)):
        if solution_types[i] == SolutionType.CMT:
            SPECFEM3D_interface.write_CMTSOLUTION(solutions[i], source_names[i])
        elif solution_types[i] == SolutionType.Force:
            raise NotImplementedError("Force solution not implemented yet")
        else:
            raise ValueError("Solution type is illegal")

    SPECFEM3D_interface.run_modelling_for_FWI_multi(source_names)
    phase_sources_multi = []
    envelope_sources_multi = []
    phase_misfit_multi = []
    envelope_misfit_multi = []
    zero_indices_multi = []
    for i in range(len(source_names)):
        observed_seismograms = SPECFEM3D_interface.read_observed_seismograms(source_names[i], True)
        observed_seismograms.trim(starttime=observed_seismograms[0].stats.starttime,
                                  endtime=observed_seismograms[0].stats.starttime+14.999, nearest_sample=False)
        SPECFEM3D_interface.copy_source_time_function(source_names[i])
        synthetic_seismograms = SPECFEM3D_interface.read_synthetic_seismograms(source_names[i], "v", solutions[i].time_shift)
        observed_seismograms.sort(keys=['station', 'channel'])
        synthetic_seismograms.sort(keys=['station', 'channel'])
        observed_seismograms_lists = seismograms_handler.split_stream_into_substreams(observed_seismograms, 50)
        synthetic_seismograms_lists = seismograms_handler.split_stream_into_substreams(synthetic_seismograms, 50)
        list_indices = list(range(len(observed_seismograms_lists)))
        phase_sources = obspy.Stream()
        envelope_sources = obspy.Stream()
        phase_misfit = []
        envelope_misfit = []
        zero_indices = []
        max_seis_level = max(observed_seismograms.max()) ** 2

        with concurrent.futures.ProcessPoolExecutor(max_workers=100) as executor:
            for retvals in executor.map(adjoint_sources.calc_adjoint_source, synthetic_seismograms_lists,
                                        observed_seismograms_lists, repeat(min_freq), repeat(max_freq),
                                        repeat(add_envelopes), repeat(max_seis_level), repeat(False),
                                        repeat(-2*solutions[i].half_duration), list_indices):
                phase_sources += retvals[0]
                envelope_sources += retvals[1]
                phase_misfit += list(retvals[2])
                envelope_misfit += list(retvals[3])
                zero_indices += list(retvals[5] + len(observed_seismograms_lists[0]) * retvals[6])
        phase_sources_multi.append(phase_sources)
        envelope_sources_multi.append(envelope_sources)
        phase_misfit_multi.append(phase_misfit)
        envelope_misfit_multi.append(envelope_misfit)
        zero_indices_multi.append(zero_indices)

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
            return None, None, None, phase_misfit_multi, envelope_misfit_multi, zero_one_indices
        if zoom_mode and ((np.sum(np.multiply(phase_misfit_multi, zero_one_indices)) +
                           np.sum(np.multiply(envelope_misfit_multi, zero_one_indices))) >
                          (orig_misfit + 10 ** (-4) * step_length * np.dot(orig_sd, orig_grad)) or
                          (np.sum(np.multiply(phase_misfit_multi, zero_one_indices)) +
                           np.sum(np.multiply(envelope_misfit_multi, zero_one_indices))) >= low_misfit):
            return None, None, None, phase_misfit_multi, envelope_misfit_multi, zero_one_indices

    for i in range(len(phase_sources_multi)):
        curr_adjoint_sources = obspy.Stream()
        phase_sources_multi[i].sort(keys=['station', 'channel'])
        envelope_sources_multi[i].sort(keys=['station', 'channel'])
        for j in range(len(phase_sources_multi[i])):
            if zero_one_indices[i][j] == 0:
                curr_adjoint_sources.append(obspy.Trace(np.zeros(phase_sources_multi[i][j].data.shape),
                                                        phase_sources_multi[i][j].stats))
                continue
            if add_envelopes:
                curr_adjoint_sources.append(obspy.Trace(-np.add(phase_sources_multi[i][j].data,
                                                                envelope_sources_multi[i][j].data),
                                                        phase_sources_multi[i][j].stats))
            else:
                curr_adjoint_sources.append(obspy.Trace(-phase_sources_multi[i][j].data,
                                                        phase_sources_multi[i][j].stats))
        SPECFEM3D_interface.write_adjoint_sources_xyz(curr_adjoint_sources, source_names[i])
    SPECFEM3D_interface.run_structural_adjoint_modelling_multi(source_names)

    moment_tensor_kernels = []
    source_location_kernels = []
    onset_time_kernels = []

    for i in range(len(source_names)):
        adjoint_strain_tensor = SPECFEM3D_interface.read_adjoint_strain_tensor_at_source(source_names[i])
        stf = SPECFEM3D_interface.read_source_time_function(source_names[i])
        moment_tensor_kernels.append(calculate_moment_tensor_kernels_xyz(adjoint_strain_tensor, stf))
        source_location_kernels.append(SPECFEM3D_interface.read_source_location_kernels(source_names[i]))
        onset_time_kernels.append(calc_onset_time_kernel_xyz(adjoint_strain_tensor, solutions[i], stf))

    return moment_tensor_kernels, source_location_kernels, onset_time_kernels, phase_misfit_multi, \
           envelope_misfit_multi, zero_one_indices_new


def smooth_q(q, mesh_size, models, hor_slice_size, smoothing_size):
    q_parts = [q[:mesh_size], q[mesh_size:2 * mesh_size], q[2 * mesh_size:3 * mesh_size]]
    model_sizes = [np.ma.size(model, axis=0) for model in models]
    model_sizes.insert(0, 0)
    model_sizes = np.cumsum(model_sizes)
    smooth_q = []
    x_num = len(np.unique(models[0][:, 0]))
    y_num = len(np.unique(models[0][:, 1]))
    z_num = len(np.unique(np.concatenate([model[:, 2] for model in models])))
    for q_part in q_parts:
        q_part_whole = []
        for i in range(1, len(model_sizes)):
            if i == 1:
                q_part_whole.extend(q_part[model_sizes[i - 1]:model_sizes[i]])
            else:
                q_part_whole.extend(q_part[(model_sizes[i - 1] + hor_slice_size):model_sizes[i]])
        q_part_whole_3d = np.reshape(q_part_whole, (x_num, y_num, z_num), order='F')
        q_part_smooth_3d = gaussian_filter(q_part_whole_3d, smoothing_size)
        q_part_smooth = q_part_smooth_3d.ravel(order='F')
        q_part_smooth_sects = []
        for i in range(1, len(model_sizes)):
            if i == 1:
                q_part_smooth_sects.extend(q_part_smooth[model_sizes[i - 1]:model_sizes[i]])
            else:
                q_part_smooth_sects.extend(q_part_smooth[(model_sizes[i - 1] - hor_slice_size * (i - 1)):(
                        model_sizes[i] - hor_slice_size * (i - 1))])
        smooth_q.extend(q_part_smooth_sects)
    smooth_q.extend(q[3 * mesh_size:])
    return np.asarray(smooth_q)


def kernels_to_gradients(vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh, hess_kernel_int_mesh=None):
    for i in range(len(vp_kernel_int_mesh)):
        vp_kernel_int_mesh[i] *= (dx*dy*dz)
        vs_kernel_int_mesh[i] *= (dx*dy*dz)
        rho_kernel_int_mesh[i] *= (dx*dy*dz)
        if hess_kernel_int_mesh is not None:
            hess_kernel_int_mesh[i] *= (dx * dy * dz)
    if hess_kernel_int_mesh is None:
        return vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh
    else:
        return vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh, hess_kernel_int_mesh


def get_structural_kernels_multi(kernel_types, source_name, orig_models):
    SPECFEM3D_interface.sum_kernels(source_name)
    for kernel_type in kernel_types:
        SPECFEM3D_interface.combine_kernels_multi(source_name, kernel_type)
    meshes = {}
    for kernel_type in kernel_types:
        meshes[kernel_type] = SPECFEM3D_interface.read_kernels(source_name, kernel_type, False, False)
    if len(kernel_types) == 3:
        vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh, points = \
            interpolate_kernels(meshes['alpha'], meshes['beta'], meshes['rho'], orig_models)
        vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh = \
            kernels_to_gradients(vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh)
        return vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh, points, meshes
    elif len(kernel_types) == 4:
        vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh, points, hess_kernel_int_mesh = \
            interpolate_kernels(meshes['alpha'], meshes['beta'], meshes['rho'], orig_models, meshes['hess'])
        vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh, hess_kernel_int_mesh = \
            kernels_to_gradients(vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh, hess_kernel_int_mesh)
        return vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh, points, hess_kernel_int_mesh, meshes
    else:
        raise ValueError("Only 3 or 4 kernels are supported")


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
    iteration -= 1090
    event_num = len(source_names)
    moment_tensor_kernel = []
    for moment_tensor_kernel_dict in moment_tensor_kernel_dicts:
        moment_tensor_kernel.extend(moment_tensor_kernel_dict_to_CMTSOLUTION_diff_xyz(moment_tensor_kernel_dict))
    orig_models = SPECFEM3D_interface.read_tomographic_models(source_names[0], -1)
    hor_slice_size = orig_models[0][orig_models[0][:, 2] == orig_models[0][0, 2]][:, 0].size
    kernel_types = ['beta', 'alpha', 'rho']
    if iteration == 1:
        kernel_types.append('hess')
        vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh, points, hess_kernel_int_mesh, meshes = \
            get_structural_kernels_multi(kernel_types, source_names[0].split("/")[0], orig_models)
    else:
        vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh, points, meshes = \
            get_structural_kernels_multi(kernel_types, source_names[0].split("/")[0], orig_models)

    kernels_vec = np.concatenate([np.concatenate(vp_kernel_int_mesh), np.concatenate(vs_kernel_int_mesh),
                                  np.concatenate(rho_kernel_int_mesh)]).flatten()
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
    sizes = [block.size for block in vp_kernel_int_mesh]
    q = kernels_vec.copy()
    if iteration == 1:
        hess_vec = np.concatenate(hess_kernel_int_mesh).flatten()
        hess_vec = np.abs(hess_vec)
        hess_vec += 10 ** (-3) * hess_vec.max()
        hess_vec = np.tile(hess_vec, 3) / 50000
        hess_vec = np.append(hess_vec, [1 / (10 ** 2), 1 / (10 ** 2), 1 / (10 ** 2)] * event_num)
        hess_vec = np.append(hess_vec,
                             [1 / (10 ** 29), 1 / (10 ** 29), 1 / (10 ** 29), 1 / (10 ** 29),
                              1 / (10 ** 29), 1 / (10 ** 29)] * event_num)
        hess_vec = np.append(hess_vec, [1 / (3 * 10 ** (-6))] * event_num)
    points = np.concatenate(points)

    if iteration == 1:
        r = -smooth_q(q / hess_vec, mesh_size, orig_models, hor_slice_size, smoothing_size)
    else:
        alphas = np.zeros(len(ss))
        for i in range(len(ss)):
            alphas[len(ss) - i - 1] = rhos[len(ss) - i - 1] * np.dot(ss[len(ss) - i - 1], q)
            q -= alphas[len(ss) - i - 1] * ys[len(ss) - i - 1]
        ss_mat = np.asarray(ss)
        ys_mat = np.asarray(ys)
        last_mem = -10 * event_num
        value_to_add = np.sum(ys_mat[:, :last_mem] ** 2, axis=0).max() / 10**6
        y_square_summed = np.sum(ys_mat ** 2, axis=0)
        y_square_summed[y_square_summed == 0] = value_to_add
        est_hess = np.abs(np.sum(ss_mat * ys_mat, axis=0) / y_square_summed)
        gamma_reduced = np.dot(ys[-1][:last_mem], ss[-1][:last_mem]) / np.dot(ys[-1][:last_mem], ys[-1][:last_mem])
        est_hess[:last_mem][est_hess[:last_mem] > gamma_reduced * 100] = gamma_reduced * 100
        est_hess[:last_mem][est_hess[:last_mem] < gamma_reduced / 100] = gamma_reduced / 100
        r = smooth_q(q * est_hess, mesh_size, orig_models, hor_slice_size, smoothing_size)
        for i in range(len(ss)):
            beta = rhos[i] * np.dot(ys[i], r)
            r += (alphas[i] - beta) * ss[i]
        r = -r

    alpha_search_direction = r[0:mesh_size]

    beta_search_direction = r[mesh_size:2 * mesh_size]
    rho_search_direction = r[2 * mesh_size:3 * mesh_size]
    alpha_sd_blocks = []
    beta_sd_blocks = []
    rho_sd_blocks = []
    for i in range(len(sizes)):
        alpha_sd_blocks.append(alpha_search_direction[int(np.sum(sizes[0:i])):int(np.sum(sizes[0:(i + 1)]))])
    for i in range(len(sizes)):
        beta_sd_blocks.append(beta_search_direction[int(np.sum(sizes[0:i])):int(np.sum(sizes[0:(i + 1)]))])
    for i in range(len(sizes)):
        rho_sd_blocks.append(rho_search_direction[int(np.sum(sizes[0:i])):int(np.sum(sizes[0:(i + 1)]))])

    new_meshes = copy.deepcopy(meshes)
    new_meshes.pop("hess", None)
    new_meshes['alpha'].point_data['alpha_kernel'] = r[0:mesh_size]
    new_meshes['alpha'].points = points
    new_meshes['beta'].point_data['beta_kernel'] = r[mesh_size:2 * mesh_size]
    new_meshes['beta'].points = points
    new_meshes['rho'].point_data['rho_kernel'] = r[2 * mesh_size:3 * mesh_size]
    new_meshes['rho'].points = points

    meshes_list = []
    keys_list = []
    for key in new_meshes.keys():
        if key == "hess":
            continue
        keys_list.append(key)
        meshes_list.append(new_meshes[key])

    for key in new_meshes.keys():
        connectivity, types = meshio.vtk._vtk_42._generate_cells([321, 321, 321])
        cells, _ = meshio.vtk._vtk_42.translate_cells(connectivity, types, {})
        interoplator = NearestNDInterpolator(new_meshes[key].points,
                                             new_meshes[key].point_data[key + "_kernel"])
        interpolated_mesh = interoplator(meshes[key].points)
        meshes[key].point_data[key + "_kernel"] = interpolated_mesh
        SPECFEM3D_interface.write_kernel(source_names[0].split("/")[0], meshes[key], key)

    source_location_grad = r[3 * mesh_size:3 * mesh_size + 3*event_num]
    moment_tensor_grad = r[3 * mesh_size + 3*event_num:3 * mesh_size + 9*event_num] * 10 ** 7
    onset_time_grad = r[-event_num:]
    return alpha_sd_blocks, beta_sd_blocks, rho_sd_blocks, source_location_grad, moment_tensor_grad, onset_time_grad, \
           raw_kernels_vec
