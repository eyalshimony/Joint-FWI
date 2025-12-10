import numpy as np
import matplotlib.pyplot as plt
import pickle


def filter_descending_part(a):
    filtered = [a[0]]
    indices = [0]

    for i in range(1, len(a)):
        if a[i] <= filtered[-1]:
            filtered.append(a[i])
            indices.append(i)

    return filtered, indices


def find_regularisation_parameter(event_int):
    reg_params = np.logspace(-4, 0)
    with open(f"reglists{event_int:04d}.pk", "rb") as f:
        [model_resolution_traces_list, moment_tensors_stds_list, moment_tensors_rel_stds_list,
         moment_tensors_rel_stds_2_list, moment_tensors_rel_comps_list, moment_tensor_arrays_list] = pickle.load(f)
    std_desc, ind_desc = filter_descending_part(moment_tensors_rel_stds_list)
    chosen_corner, _, all_corners = corner(std_desc, 1 - np.asarray(model_resolution_traces_list)[ind_desc]/3)
    chosen_corner = all_corners[np.abs(np.array(std_desc)[all_corners] - np.array(1 - np.asarray(model_resolution_traces_list)[ind_desc]/3)[all_corners]).argmin()]
    return chosen_corner, all_corners, reg_params[chosen_corner], \
           1 - np.asarray(model_resolution_traces_list)[ind_desc][chosen_corner]/6, std_desc[chosen_corner], \
           reg_params[all_corners], 1 - np.asarray(model_resolution_traces_list)[ind_desc][all_corners]/6, np.asarray(std_desc)[all_corners]


def corner(rho, eta, fig=None):
    """
    Find the corner of a discrete L-curve using an adaptive pruning algorithm.

    Parameters:
        rho (ndarray): Vector containing values of residual norm ||A x - b||.
        eta (ndarray): Vector containing solution norms ||x|| or ||L x||.
        fig (int, optional): If provided, plots the L-curve and the detected corner.

    Returns:
        k_corner (int): Index of the detected corner.
        info (int): Information about warnings.
    """
    if len(rho) != len(eta):
        raise ValueError("Vectors rho and eta must have the same length")
    if len(rho) < 3:
        raise ValueError("Vectors rho and eta must have at least 3 elements")

    rho = np.array(rho).flatten()  # Ensure rho and eta are 1D arrays
    eta = np.array(eta).flatten()

    if fig is None:
        fig = 0

    info = 0

    # Handle bad data (Inf, NaN, or zeros)
    finite_mask = np.isfinite(rho + eta)
    nonzero_mask = (rho * eta) != 0
    kept = np.where(finite_mask & nonzero_mask)[0]

    if kept.size == 0:
        raise ValueError("Too many Inf/NaN/zeros found in data")

    if kept.size < len(rho):
        info += 1
        print("Warning: Bad data - Inf, NaN, or zeros found in data. Continuing with remaining data.")

    rho = rho[kept]
    eta = eta[kept]

    # Check for monotonicity
    if np.any(np.diff(rho) > 0) or np.any(np.diff(eta) < 0):
        info += 10
        print("Warning: Lack of monotonicity")

    # Prepare for adaptive algorithm
    nP = len(rho)
    P = np.column_stack((rho, eta))
    V = P[1:] - P[:-1]
    v = np.sqrt(np.sum(V**2, axis=1))
    W = V / v[:, np.newaxis]
    clist = []
    p = min(5, nP - 1)
    convex = 0

    # Sort vectors by length (longest first)
    I = np.argsort(v)[::-1]

    # Main loop to prune L-curve
    while p < (nP - 1) * 2:
        elmts = np.sort(I[:min(p, nP - 1)])

        # First corner location algorithm (based on angles)
        candidate = Angles(W[elmts], elmts)
        if candidate > 0:
            convex = 1
        if candidate and candidate not in clist:
            clist.append(candidate)

        # Second corner location algorithm (based on global behavior)
        candidate = Global_Behavior(P, W[elmts], elmts)
        if candidate not in clist:
            clist.append(candidate)

        p *= 2

    # Issue a warning if no convexity is found
    if convex == 0:
        k_corner = None
        info += 100
        print("Warning: Lack of convexity")
        return k_corner, info

    # Ensure the rightmost L-curve point is in clist
    if 1 not in clist:
        clist = [1] + clist

    # Sort corner candidates in increasing order
    clist = sorted(clist)

    # Select the best corner
    vz = [i for i in range(len(clist) - 1) if (P[clist[i + 1], 1] - P[clist[i], 1]) >= abs(P[clist[i + 1], 0] - P[clist[i], 0])]

    if len(vz) > 1 and vz[0] == 0:
        vz = vz[1:]
    elif len(vz) == 1 and vz[0] == 0:
        vz = []

    if not vz:
        index = clist[-1]
    else:
        vz = np.asarray(vz)
        vects = np.column_stack((P[clist[1:], 0] - P[clist[:-1], 0], P[clist[1:], 1] - P[clist[:-1], 1]))
        vects = vects / np.linalg.norm(vects, axis=1)[:, np.newaxis]
        delta = vects[:-1, 0] * vects[1:, 1] - vects[1:, 0] * vects[:-1, 1]
        vv = np.where(delta[vz - 1] <= 0)[0]

        if vv.size == 0:
            index = clist[vz[-1]]
        else:
            index = clist[vz[vv[0]]]

    k_corner = kept[index]

    if fig:  # Plot the L-curve
        plt.figure(fig)
        plt.loglog(rho, eta, 'k--o')
        plt.loglog([min(rho)/100, rho[index]], [eta[index], eta[index]], ':r')
        plt.loglog([rho[index], rho[index]], [min(eta)/100, eta[index]], ':r')
        plt.xlabel('Residual norm ||A x - b||_2')
        plt.ylabel('Solution (semi)norm ||L x||_2')
        plt.title(f'Discrete L-curve, corner at {k_corner}')
        plt.grid(True)
        plt.axis('square')
        plt.show()

    return k_corner, info, clist


def Angles(W, kv):
    """
    First corner finding routine -- based on angles.
    """
    delta = W[:-1, 0] * W[1:, 1] - W[1:, 0] * W[:-1, 1]
    min_delta = np.min(delta)
    if min_delta < 0:
        index = kv[np.argmin(delta)] + 1
    else:
        index = 0
    return index


def Global_Behavior(P, vects, elmts):
    """
    Second corner finding routine -- based on global behavior of the L-curve.
    """
    hwedge = np.abs(vects[:, 1])  # Angle with the horizontal axis
    sorted_indices = np.argsort(hwedge)

    ln = len(sorted_indices)
    mn = sorted_indices[0]
    mx = sorted_indices[-1]
    count = 1

    while mn >= mx:
        mx = max(mx, sorted_indices[ln - count])
        count += 1
        mn = min(mn, sorted_indices[count])

    if count > 1:
        I, J = 0, 0
        for i in range(count):
            for j in range(ln - 1, ln - count, -1):
                if sorted_indices[i] < sorted_indices[j]:
                    I, J = sorted_indices[i], sorted_indices[j]
                    break
            if I > 0:
                break
    else:
        I, J = sorted_indices[0], sorted_indices[-1]

    # Intersection describing the "origin"
    x3 = P[elmts[J] + 1, 0] + (P[elmts[I], 1] - P[elmts[J] + 1, 1]) / (P[elmts[J] + 1, 1] - P[elmts[J], 1]) * (P[elmts[J] + 1, 0] - P[elmts[J], 0])
    origin = [x3, P[elmts[I], 1]]

    # Find distances to the "origin"
    dists = (origin[0] - P[:, 0])**2 + (origin[1] - P[:, 1])**2
    index = np.argmin(dists)
    return index