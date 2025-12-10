import SPECFEM3D_interface
from objects import SolutionType, CMTSolution
import numpy as np
import sensitivity_kernels
import pickle


def try_different_step_lengths_xyz_multi(source_names, min_freq, max_freq, add_envelopes, phase_misfit, envelope_misfit,
                                         vp_search_direction, vs_search_direction, rho_search_direction,
                                         source_location_search_direction, moment_tensor_search_direction,
                                         onset_time_search_direction, solutions, it_num, ss, LBFGS_mem_size, max_reps,
                                         curr_grad, zero_indices):
    step_size = 1
    step_size_prev = 0
    orig_misfit = np.sum(np.multiply(phase_misfit, zero_indices)) + np.sum(np.multiply(envelope_misfit, zero_indices))
    orig_models = SPECFEM3D_interface.read_tomographic_models(source_names[0].split("/")[0], it_num - 1)
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
        x_loc = solution.longorUTM
        y_loc = solution.latorUTM
        curr_souce_locations.append(np.asarray([x_loc, y_loc, solution.depth * 1000]))
        curr_moment_tensors.append(np.asarray([solution.Mrr, solution.Mtt, solution.Mpp, solution.Mrt, solution.Mrp,
                                               solution.Mtp]))
    search_direction_assembled = np.concatenate([np.concatenate([np.concatenate(vp_search_direction),
                                                                 np.concatenate(vs_search_direction),
                                                                 np.concatenate(rho_search_direction)]).flatten(),
                                                 source_location_search_direction, moment_tensor_search_direction/10**7,
                                                 onset_time_search_direction])
    direct_derivs[step_size_prev] = np.dot(search_direction_assembled, curr_grad)
    print(orig_misfit)
    first_it = True
    prev_misfit = orig_misfit
    low_misfit = None
    zoom_mode = False
    j = 0
    while True:
        curr_models = sensitivity_kernels.sum_interpolated_kernels_and_models_multi(vp_search_direction,
                                                                                    vs_search_direction,
                                                                                    rho_search_direction, orig_models,
                                                                                    step_size,
                                                                                    source_names[0].split("/")[0],
                                                                                    len(source_names))
        new_solutions = []
        for i in range(len(solutions)):
            new_source_location = curr_souce_locations[i] + step_size * source_location_search_direction[i*3:(i+1)*3]
            new_moment_tensor = curr_moment_tensors[i] + step_size * moment_tensor_search_direction[i*6:(i+1)*6]
            new_time_shift = solutions[i].time_shift + step_size * onset_time_search_direction[i]
            new_solutions.append(CMTSolution(solutions[i].event_name, new_time_shift, solutions[i].half_duration,
                                             new_source_location[1], new_source_location[0],
                                             new_source_location[2] / 1000, *new_moment_tensor))

        moment_tensor_kernel, source_location_kernel, onset_time_kernel, phase_misfit, envelope_misfit, zero_indices_curr = \
            sensitivity_kernels.calc_all_derivatives_multi(source_names, new_solutions,
                                                           [SolutionType.CMT] * len(solutions), min_freq, max_freq,
                                                           add_envelopes, True, orig_misfit, step_size, curr_grad,
                                                           search_direction_assembled, first_it, prev_misfit, zoom_mode,
                                                           low_misfit, zero_indices)
        curr_misfit = np.sum(np.multiply(phase_misfit, zero_indices)) + np.sum(np.multiply(envelope_misfit, zero_indices))
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
            vp_kernel_int_mesh, vs_kernel_int_mesh, rho_kernel_int_mesh, points, meshes = \
                sensitivity_kernels.get_structural_kernels_multi(kernel_types, source_names[0].split("/")[0],
                                                                 curr_models)
            kernels_vec = np.concatenate([np.concatenate(vp_kernel_int_mesh), np.concatenate(vs_kernel_int_mesh),
                                          np.concatenate(rho_kernel_int_mesh)]).flatten()
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
        orig_model_vec = SPECFEM3D_interface.models_to_1d_vector(orig_models)
        for curr_souce_location in curr_souce_locations:
            orig_model_vec = np.append(orig_model_vec, curr_souce_location)
        for curr_moment_tensor in curr_moment_tensors:
            orig_model_vec = np.append(orig_model_vec, curr_moment_tensor / 10 ** 7)
        for solution in solutions:
            orig_model_vec = np.append(orig_model_vec, solution.time_shift)
        current_model_vec = SPECFEM3D_interface.models_to_1d_vector(current_models)
        for new_solution in new_solutions:
            current_model_vec = np.append(current_model_vec, [new_solution.longorUTM, new_solution.latorUTM,
                                                              new_solution.depth * 1000])
        for new_solution in new_solutions:
            current_model_vec = np.append(current_model_vec, np.asarray([new_solution.Mrr, new_solution.Mtt,
                                                                         new_solution.Mpp, new_solution.Mrt,
                                                                         new_solution.Mrp, new_solution.Mtp]) / 10 ** 7)
        for new_solution in new_solutions:
            current_model_vec = np.append(current_model_vec, new_solution.time_shift)
        ss.append(current_model_vec - orig_model_vec)
        for i in range(len(new_solutions)):
            with open("/DATA/eyal/specfem3d/PROJECTS/mtinv/test5/DATA/CMTSOLUTION_" + str(i) + "_" + str(it_num) + ".pk", "wb") as f:
                pickle.dump(new_solutions[i], f)
        return failed, phase_misfits[-1], envelope_misfits[-1], moment_tensor_kernel, source_location_kernel, \
               onset_time_kernel, new_solutions, zero_indices_curr
    else:
        return failed, None, None, None, None, None, None, None
