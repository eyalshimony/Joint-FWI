import sensitivity_kernels
import step_sizes
from objects import SolutionType
import pickle


def LBFGS_inversion_multi(solutions, smoothing_size, min_freq, max_freq, add_envelopes, max_reps):
    LBFGS_mem_size = 5
    source_names = []
    for solution in solutions:
        source_names.append(solution.event_name)

    with open("misfits_multi_2.pk", "rb") as f:
        [phase_misfits, envelope_misfits, zero_indices_stages] = pickle.load(f)

    ss = []
    ys = []
    rhos = []
    i = 0
    moment_tensor_kernels, source_location_kernels, onset_time_kernels, phase_misfit, envelope_misfit, zero_indices = \
        sensitivity_kernels.calc_all_derivatives_multi(source_names, solutions, [SolutionType.CMT] * len(source_names),
                                                       min_freq, max_freq, add_envelopes)
    phase_misfits[i] = phase_misfit
    envelope_misfits[i] = envelope_misfit
    zero_indices_stages[i] = zero_indices
    with open("misfits_multi_2.pk", "wb") as f:
        misfits = [phase_misfits, envelope_misfits, zero_indices_stages]
        pickle.dump(misfits, f, protocol=4)

    i += 1
    with open("retval.pk", "wb") as f:
        retvals = [phase_misfit, envelope_misfit, ss, ys, rhos, i, solutions, source_location_kernels,
                   moment_tensor_kernels, onset_time_kernels, zero_indices]
        pickle.dump(retvals, f, protocol=4)
    with open("retval.pk", "rb") as f:
        [phase_misfit, envelope_misfit, ss, ys, rhos, i, solutions, source_location_kernels,
         moment_tensor_kernels, onset_time_kernels, zero_indices] = pickle.load(f)
    vp_search_direction, vs_search_direction, rho_search_direction, source_location_search_direction, \
        moment_tensor_search_direction, onset_time_search_direction, current_kernel = \
        sensitivity_kernels.calculate_search_direction_multi(source_names, source_location_kernels,
                                                             moment_tensor_kernels, onset_time_kernels, smoothing_size,
                                                             i)
    with open("retval.pk", "wb") as f:
        retvals = [phase_misfit, envelope_misfit, vp_search_direction, vs_search_direction, rho_search_direction,
                   ss, ys, rhos, current_kernel, i, solutions, source_location_kernels,
                   source_location_search_direction, moment_tensor_kernels, moment_tensor_search_direction,
                   onset_time_kernels, onset_time_search_direction, zero_indices]
        pickle.dump(retvals, f, protocol=4)
    with open("retval.pk", "rb") as f:
        [phase_misfit, envelope_misfit, vp_search_direction, vs_search_direction, rho_search_direction,
           ss, ys, rhos, current_kernel, i, solutions, source_location_kernels,
           source_location_search_direction, moment_tensor_kernels, moment_tensor_search_direction,
           onset_time_kernels, onset_time_search_direction, zero_indices] = pickle.load(f)
    failed, phase_misfit, envelope_misfit, moment_tensor_kernels, source_location_kernels, onset_time_kernels, solutions, zero_indices = \
        step_sizes.try_different_step_lengths_xyz_multi(source_names, min_freq, max_freq,
                                                        add_envelopes, phase_misfit, envelope_misfit,
                                                        vp_search_direction, vs_search_direction, rho_search_direction,
                                                        source_location_search_direction,
                                                        moment_tensor_search_direction, onset_time_search_direction,
                                                        solutions, i, ss, LBFGS_mem_size, max_reps, current_kernel,
                                                        zero_indices)
    phase_misfits[i] = phase_misfit
    envelope_misfits[i] = envelope_misfit
    zero_indices_stages[i] = zero_indices
    with open("misfits_multi_2.pk", "wb") as f:
        misfits = [phase_misfits, envelope_misfits, zero_indices_stages]
        pickle.dump(misfits, f, protocol=4)

    with open("retval.pk", "wb") as f:
        retvals = [phase_misfit, envelope_misfit, vp_search_direction, vs_search_direction, rho_search_direction,
                   ss, ys, rhos, current_kernel, i, solutions, source_location_kernels,
                   source_location_search_direction, moment_tensor_kernels, moment_tensor_search_direction,
                   onset_time_kernels, onset_time_search_direction, zero_indices]
        pickle.dump(retvals, f, protocol=4)
    if failed:
        raise NotImplementedError("Failed on first stage")
    with open("retval.pk", "rb") as f:
        [phase_misfit, envelope_misfit, vp_search_direction, vs_search_direction, rho_search_direction,
           ss, ys, rhos, current_kernel, i, solutions, source_location_kernels,
           source_location_search_direction, moment_tensor_kernels, moment_tensor_search_direction,
           onset_time_kernels, onset_time_search_direction, zero_indices] = pickle.load(f)
    while True:
        i += 1
        vp_search_direction, vs_search_direction, rho_search_direction, source_location_search_direction, \
            moment_tensor_search_direction, onset_time_search_direction, current_kernel = \
            sensitivity_kernels.calculate_search_direction_multi(source_names, source_location_kernels,
                                                           moment_tensor_kernels, onset_time_kernels, smoothing_size,
                                                           i, ss, ys, rhos, current_kernel, LBFGS_mem_size)
        with open("retval.pk", "wb") as f:
            retvals = [phase_misfit, envelope_misfit, vp_search_direction, vs_search_direction, rho_search_direction,
                       ss, ys, rhos, current_kernel, i, solutions, source_location_kernels,
                       source_location_search_direction, moment_tensor_kernels, moment_tensor_search_direction,
                       onset_time_kernels, onset_time_search_direction, zero_indices]
            pickle.dump(retvals, f, protocol=4)
        failed, phase_misfit, envelope_misfit, moment_tensor_kernels, source_location_kernels, onset_time_kernels, solutions, zero_indices = \
            step_sizes.try_different_step_lengths_xyz_multi(source_names, min_freq, max_freq, add_envelopes, phase_misfit,
                                                  envelope_misfit, vp_search_direction, vs_search_direction,
                                                  rho_search_direction, source_location_search_direction,
                                                  moment_tensor_search_direction, onset_time_search_direction, solutions, i,
                                                  ss, LBFGS_mem_size, max_reps, current_kernel, zero_indices)
        phase_misfits[i] = phase_misfit
        envelope_misfits[i] = envelope_misfit
        zero_indices_stages[i] = zero_indices
        if failed:
            print("Finished loop")
            return
        with open("misfits_multi_2.pk", "wb") as f:
            misfits = [phase_misfits, envelope_misfits, zero_indices_stages]
            pickle.dump(misfits, f, protocol=4)
        with open("retval.pk", "wb") as f:
            retvals = [phase_misfit, envelope_misfit, vp_search_direction, vs_search_direction, rho_search_direction,
                       ss, ys, rhos, current_kernel, i, solutions, source_location_kernels,
                       source_location_search_direction, moment_tensor_kernels, moment_tensor_search_direction,
                       onset_time_kernels, onset_time_search_direction, zero_indices]
            pickle.dump(retvals, f, protocol=4)
