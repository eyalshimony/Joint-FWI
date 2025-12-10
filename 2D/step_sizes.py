import SPECFEM3D_interface
from objects import CMTSolution
import numpy as np
import sensitivity_kernels
import pickle
import os
import re


def try_different_step_lengths_xyz_multi(source_names, min_freq, max_freq, add_envelopes, phase_misfit, envelope_misfit,
                                         vp_search_direction, vs_search_direction, rho_search_direction,
                                         source_location_search_direction, moment_tensor_search_direction,
                                         onset_time_search_direction, solutions, it_num, ss, LBFGS_mem_size, max_reps,
                                         curr_grad, zero_indices):
    step_size = 1
    step_size_prev = 0
    orig_misfit = np.sum(np.multiply(phase_misfit, zero_indices)) + np.sum(np.multiply(envelope_misfit, zero_indices))
    orig_model = SPECFEM3D_interface.read_tomographic_models(source_names[0].split("/")[0], it_num - 1)
    phase_misfits = []
    envelope_misfits = []
    misfits = {}
    direct_derivs = {}
    phase_misfits.append(phase_misfit)
    envelope_misfits.append(envelope_misfit)
    misfits[step_size_prev] = orig_misfit
    failed = False
    curr_souce_locations = []
    curr_moment_tensors = []
    for solution in solutions:
        curr_souce_locations.append(np.asarray([solution.xs, solution.zs]))
        curr_moment_tensors.append(np.asarray([solution.Mzz, solution.Mxx, solution.Mxz]))
    search_direction_assembled = np.concatenate([np.concatenate([vp_search_direction, vs_search_direction,
                                                                 rho_search_direction]).flatten(),
                                                 source_location_search_direction,
                                                 moment_tensor_search_direction,
                                                 onset_time_search_direction])
    direct_derivs[step_size_prev] = np.dot(search_direction_assembled, curr_grad)
    print(orig_misfit)
    print(direct_derivs)
    first_it = True
    prev_misfit = orig_misfit
    low_misfit = None
    zoom_mode = False
    j = 0
    while True:
        curr_models, step_size = sensitivity_kernels.sum_interpolated_kernels_and_models_multi(vp_search_direction,
                                                                                    vs_search_direction,
                                                                                    rho_search_direction, orig_model,
                                                                                    step_size,
                                                                                    source_names[0].split("/")[0])
        new_solutions = []
        for i in range(len(solutions)):
            new_source_location = curr_souce_locations[i] + step_size * source_location_search_direction[
                                                                        i * 2:(i + 1) * 2]
            if new_source_location[0] < 750:
                new_source_location[0] = 750
            if new_source_location[1] < 750:
                new_source_location[1] = 750
            if new_source_location[0] > 19250:
                new_source_location[0] = 19250
            if new_source_location[1] > 20000:
                new_source_location[1] = 20000
            new_moment_tensor = curr_moment_tensors[i] + step_size * moment_tensor_search_direction[i * 3:(i + 1) * 3]
            new_time_shift = solutions[i].tshift + step_size * onset_time_search_direction[i]
            new_solutions.append(CMTSolution(solutions[i].event_name, new_source_location[0], new_source_location[1],
                                             solutions[i].f0, new_time_shift, *new_moment_tensor))
        moment_tensor_kernel, source_location_kernel, onset_time_kernel, phase_misfit, envelope_misfit, \
        phase_misfit_next, envelope_misfit_next, zero_indices_curr = \
            sensitivity_kernels.calc_all_derivatives_multi(source_names, new_solutions, min_freq, max_freq,add_envelopes,
                                                           True, orig_misfit, step_size, curr_grad,
                                                           search_direction_assembled, first_it, prev_misfit, zoom_mode,
                                                           low_misfit, zero_indices)
        curr_misfit = np.sum(np.multiply(phase_misfit, zero_indices)) + np.sum(
            np.multiply(envelope_misfit, zero_indices))
        phase_misfits.append(phase_misfit)
        envelope_misfits.append(envelope_misfit)
        misfits[step_size] = curr_misfit
        print(curr_misfit)

        if moment_tensor_kernel is None:
            if zoom_mode:
                step_size_high = step_size
            else:
                zoom_mode = True
                step_size_low = step_size_prev
                step_size_high = step_size
                low_misfit = prev_misfit
            step_size = (step_size_low + step_size_high) / 2.0
        else:
            kernel_types = ['beta', 'alpha', 'rho']
            vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh, points = \
                sensitivity_kernels.get_structural_kernels_multi(kernel_types, source_names[0].split("/")[0],
                                                                 curr_models)
            kernels_vec = np.concatenate([vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh]).flatten()
            moment_tensor_kernel_arr = []
            for moment_tensor_kernel_dict in moment_tensor_kernel:
                moment_tensor_kernel_arr.extend(
                    sensitivity_kernels.moment_tensor_kernel_dict_to_CMTSOLUTION_diff_xyz(moment_tensor_kernel_dict))
            kernels_vec = np.append(kernels_vec, source_location_kernel)
            kernels_vec = np.append(kernels_vec, moment_tensor_kernel_arr)
            kernels_vec = np.append(kernels_vec, onset_time_kernel)
            direct_derivs[step_size] = np.dot(search_direction_assembled, kernels_vec)
            print(direct_derivs)
            if zoom_mode:
                if np.abs(np.dot(search_direction_assembled, kernels_vec)) <= \
                        -0.95 * np.dot(search_direction_assembled, curr_grad):
                    break
                if np.dot(search_direction_assembled, kernels_vec) * (step_size_high - step_size_low) >= 0:
                    step_size_high = step_size_low
                step_size_low = step_size
                low_misfit = curr_misfit
                step_size = (step_size_low + step_size_high) / 2.0
            else:
                if np.abs(np.dot(search_direction_assembled, kernels_vec)) <= \
                        -0.95 * np.dot(search_direction_assembled, curr_grad):
                    break
                if np.dot(search_direction_assembled, kernels_vec) >= 0:
                    zoom_mode = True
                    step_size_low = step_size
                    step_size_high = step_size_prev
                    step_size = (step_size_low + step_size_high) / 2.0
                    low_misfit = curr_misfit
                else:
                    step_size_prev = step_size
                    step_size *= 2.0
                prev_misfit = curr_misfit
        first_it = False
        j += 1
        if j > max_reps:
            failed = True
            break

    if not failed:
        SPECFEM3D_interface.write_iteration_model(source_names[0].split("/")[0], it_num)
        current_models = SPECFEM3D_interface.read_tomographic_models(source_names[0].split("/")[0], it_num)
        if len(ss) >= LBFGS_mem_size:
            ss.pop(0)
        orig_model_vec = SPECFEM3D_interface.models_to_1d_vector(orig_model)
        for curr_souce_location in curr_souce_locations:
            orig_model_vec = np.append(orig_model_vec, curr_souce_location)
        for curr_moment_tensor in curr_moment_tensors:
            orig_model_vec = np.append(orig_model_vec, curr_moment_tensor)
        for solution in solutions:
            orig_model_vec = np.append(orig_model_vec, solution.tshift)
        current_model_vec = SPECFEM3D_interface.models_to_1d_vector(current_models)
        for new_solution in new_solutions:
            current_model_vec = np.append(current_model_vec, [new_solution.xs, new_solution.zs])
        for new_solution in new_solutions:
            current_model_vec = np.append(current_model_vec, np.asarray([new_solution.Mzz, new_solution.Mxx,
                                                                         new_solution.Mxz]))
        for new_solution in new_solutions:
            current_model_vec = np.append(current_model_vec, new_solution.tshift)
        ss.append(current_model_vec - orig_model_vec)
        with open(f"/DATA/eyal/specfem2d/mtinv/DATA/SOURCES_it{it_num}.pk", "wb") as f:
            pickle.dump(new_solutions, f)
        rename_synthmult_files("/DATA/eyal/specfem2d/mtinv/")
        return failed, phase_misfits[-1], envelope_misfits[-1], phase_misfit_next, envelope_misfit_next, \
               moment_tensor_kernel, source_location_kernel, onset_time_kernel, new_solutions, zero_indices_curr
    else:
        return failed, None, None, None, None, None, None, None, None, None


def rename_synthmult_files(folder_path):
    """
    Renames files in a given folder from 'synthmult_curr_i_j.pk' to 'synthmult_i_j.pk'.

    Args:
        folder_path (str): The path to the folder containing the files.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at '{folder_path}'")
        return

    # Regular expression to match 'synthmult_curr_i_j.pk'
    # We capture 'i_j.pk' so we can reuse it for the new filename.
    pattern = re.compile(r"synthmult_curr_(\d+_\d+\.pk)")
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            old_filepath = os.path.join(folder_path, filename)
            # The captured group (i_j.pk) becomes the new suffix
            new_filename = f"synthmult_{match.group(1)}"
            new_filepath = os.path.join(folder_path, new_filename)
            try:
                os.rename(old_filepath, new_filepath)
            except OSError as e:
                print(f"Error renaming '{filename}': {e}")
