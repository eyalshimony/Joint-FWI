import data
import numpy as np
import obspy
from scipy.signal import welch, convolve, fftconvolve
from obspy.signal.interpolation import lanczos_interpolation
import re
from obspy import Stream, Trace
from scipy.signal import get_window


def ordered_stream_into_ordered_matrix(stream):
    ordered_mat = []
    for trace in stream:
        ordered_mat.append(trace.data)
    return np.array(ordered_mat)


def split_stream_into_substreams(stream, num):
    traces_per_stream = int(np.ceil(len(stream) / num))
    substreams = []

    for i in range(num):
        substreams.append(stream[(i*traces_per_stream):((i+1)*traces_per_stream)])

    return substreams


def hann_taper_2d(data, max_perc, side, dimensions):
    """
    Applies a Hann taper to 2D data.

    Parameters:
    -----------
    data : ndarray
        The 2D data array to apply the taper to.
    max_perc : float
        The maximum percentage (0-1) of the data to taper.
    side : str
        The side to apply the taper ('left', 'right', 'both').
    dimensions : tuple
        Dimensions to apply the taper, either (0, 1) for rows and columns.

    Returns:
    --------
    ndarray
        The tapered 2D data array.
    """
    if max_perc < 0 or max_perc > 1:
        raise ValueError("max_perc must be between 0 and 1.")
    if side not in ('left', 'right', 'both'):
        raise ValueError("side must be one of 'left', 'right', or 'both'.")
    if len(dimensions) != 2:
        raise ValueError("dimensions must have two entries (rows, cols).")

    rows, cols = data.shape
    taper_rows, taper_cols = dimensions

    # Hann taper functions for rows and columns
    def generate_taper(length, max_perc, side):
        taper_length = int(max_perc * length)
        taper = np.hanning(2 * taper_length)[:taper_length]
        taper_full = np.ones(length)
        if side == 'left':
            taper_full[:taper_length] = taper
        elif side == 'right':
            taper_full[-taper_length:] = taper[::-1]
        elif side == 'both':
            taper_full[:taper_length] = taper
            taper_full[-taper_length:] = taper[::-1]
        return taper_full

    row_taper = generate_taper(rows, max_perc, side) if taper_rows else np.ones(rows)
    col_taper = generate_taper(cols, max_perc, side) if taper_cols else np.ones(cols)

    # Apply the taper to the 2D data
    taper_2d = np.outer(row_taper, col_taper)
    return data * taper_2d


def mean_central_frequency(data, fs):
    """
    Computes the mean central frequency for a 2D array, considering only non-zero values.
    :param data: 2D array, where each row represents a time-series signal.
    :param fs: Sampling frequency.
    :return: Mean central frequency (excluding zero values).
    """
    # Compute shared parameters
    nperseg = min(4096, data.shape[1])  # Next power of 2 for FFT

    # Apply Welch's method for all rows
    freqaxis, psd = welch(data, fs=fs, nperseg=nperseg, axis=1, window="hamming", detrend=False)

    # Calculate numerator and denominator for central frequency
    numerator = np.sum(freqaxis * psd, axis=1)
    denominator = np.sum(psd, axis=1)

    # Calculate central frequencies
    central_frequencies = np.divide(numerator, denominator, where=denominator > 0)

    # Compute mean of non-zero central frequencies
    non_zero_mask = denominator > 0
    total_sum = np.sum(central_frequencies[non_zero_mask])
    count = np.sum(non_zero_mask)

    return total_sum / count if count > 0 else 0


def central_frequency_unwindowed_2d(data, fs_time):

    cfreq_time = mean_central_frequency(data, fs_time)

    return cfreq_time


def interpolate_2d(data, dt_new, nt, dt, max_time_freq, min_num_time):
    """
    Interpolates a 2D dataset in both temporal (time) and spatial directions.

    :param data: 2D NumPy array to interpolate (shape: [nt, nx]).
    :param dt_new: New time step for interpolation.
    :param nt: Original number of time steps.
    :param nx: Original number of spatial steps.
    :param dt: Original time step.
    :param max_time_freq: Maximum frequency for time interpolation.
    :param min_num_time: Minimum number of points in the time dimension.
    :return: Interpolated 2D NumPy array.
    """
    # Calculate new number of points for time and space
    npts_time = int(max(np.ceil(nt * dt * 2 * max_time_freq), min_num_time))

    # Define the new start points
    old_start_time = 0.0
    new_start_time = 0.0

    interpolated_data = []

    for row in data:
        interpolated_data.append(lanczos_interpolation(row, old_start_time, dt, new_start_time, dt_new, npts_time, a=8,
                                                       window="blacmann"))

    interpolated_data = np.array(interpolated_data)

    return interpolated_data


def calculate_GL_seismograms(directional_seismograms, fibre, GL, channel_spacing):
    partial_stream = directional_seismograms.select(station=fibre + "*")
    fibre_stations = [trace.stats.station for trace in partial_stream]
    data_matrix = np.asarray([trace.data for trace in partial_stream])
    stencil = np.array([0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05])
    DAS_data_matrix = np.apply_along_axis(lambda row: convolve(row, stencil, mode='same'), axis=0, arr=data_matrix)
    DAS_data_matrix = DAS_data_matrix[::channel_spacing]
    stream_to_return = obspy.Stream()
    for i in range(len(DAS_data_matrix)):
        stream_to_return.append(partial_stream[i * channel_spacing])
        stream_to_return[i].data = DAS_data_matrix[i, :]
    return stream_to_return


def calculate_DAS_seismograms(stream):
    GL = data.GL
    channel_spacing = data.channel_spacing
    fibres = list(set([re.sub(r'[^a-zA-Z]', '', trace.stats.station) for trace in stream]))
    directional_GL_seismograms = obspy.Stream()
    for fibre in fibres:
        partial_stream = stream.select(station=fibre+"*")
        partial_stream = obspy.Stream([tr for tr in partial_stream if re.sub(r'[^a-zA-Z]', '', tr.stats.station) == fibre])
        if fibre == "S":
            directional_seismograms = partial_stream.select(channel="X")
        else:
            directional_seismograms = partial_stream.select(channel="Z")
        directional_seismograms.sort(keys=['station'])
        directional_GL_seismograms += calculate_GL_seismograms(directional_seismograms, fibre, GL, channel_spacing)
    directional_GL_seismograms.sort(keys=['station'])
    directional_GL_seismograms.filter("lowpass", freq=24, zerophase=True)
    return directional_GL_seismograms


def generate_correlated_noise(nt, nx, sigma, corr_len_t, corr_len_x):
    """
    Generates 2D spatially correlated Gaussian noise.

    Args:
        nt (int): Number of points along the first dimension (e.g., time).
        nx (int): Number of points along the second dimension (e.g., space).
        sigma (float): Desired standard deviation (noise level) of the output noise.
        corr_len_t (float): Correlation length along the first dimension (in points/indices).
                            Controls smoothness along this axis.
        corr_len_x (float): Correlation length along the second dimension (in points/indices).
                            Controls smoothness along this axis.

    Returns:
        np.ndarray: A 2D numpy array of shape (nt, nx) containing the
                    correlated Gaussian noise. Returns a zero array if scaling fails.
    """
    # --- 1. Generate uncorrelated white Gaussian noise ---
    # Noise with mean 0 and standard deviation 1
    white_noise = np.random.randn(nt, nx)

    # --- 2. Create the 2D Gaussian smoothing kernel ---
    # Determine kernel size based on correlation lengths (e.g., out to ~3 sigma)
    # Ensure the kernel size is odd for proper centering
    kernel_size_t = max(1, int(np.ceil(6 * corr_len_t)) // 2 * 2 + 1)
    kernel_size_x = max(1, int(np.ceil(6 * corr_len_x)) // 2 * 2 + 1)

    # Create kernel coordinate grids centered at zero
    t_range = np.arange(kernel_size_t) - kernel_size_t // 2
    x_range = np.arange(kernel_size_x) - kernel_size_x // 2
    # Note: Using 'ij' indexing to match common image/matrix conventions (t=rows, x=cols)
    t_coords, x_coords = np.meshgrid(t_range, x_range, indexing='ij')

    # Calculate Gaussian kernel values
    # Add a small epsilon to avoid division by zero if a correlation length is zero
    epsilon = 1e-10
    term_t = (t_coords**2) / (2 * (corr_len_t**2 + epsilon))
    term_x = (x_coords**2) / (2 * (corr_len_x**2 + epsilon))
    kernel = np.exp(-term_t - term_x)

    # Normalize the kernel so its sum is 1.
    # This ensures that the convolution doesn't change the overall mean (which is ~0).
    kernel_sum = np.sum(kernel)
    if kernel_sum < epsilon:
        # Handle degenerate case (e.g., return white noise scaled?)
        # For now, normalize cautiously
        kernel = np.zeros_like(kernel)
        kernel[kernel_size_t // 2, kernel_size_x // 2] = 1.0 # Delta function
    else:
        kernel /= kernel_sum

    # --- 3. Convolve white noise with the kernel ---
    # 'same' mode keeps the output size the same as the input white_noise
    # 'wrap' handles boundaries by wrapping around (periodic assumption)
    # Other boundary options: 'fill', 'symm', 'reflect'
    convolved_noise = fftconvolve(white_noise, kernel, mode='same')

    # --- 4. Scale the result to the desired standard deviation (sigma) ---
    # Calculate the standard deviation of the noise *after* convolution
    current_std = np.std(convolved_noise)

    # Avoid division by zero if the standard deviation is unexpectedly zero
    if current_std < epsilon:
         # print("Warning: Standard deviation of convolved noise is near zero. Cannot scale. Returning zero array.")
         return np.zeros((nt, nx))

    # Calculate the scaling factor
    scaling_factor = sigma / current_std

    # Apply the scaling factor
    correlated_noise = convolved_noise * scaling_factor

    return correlated_noise


def optimized_resample_stream(stream_in, new_sampling_rate, window='hanning',
                              no_filter=True, strict_length=False, num_cores=1):
    """
    Resamples an Obspy Stream object with optimizations for zero traces and
    vectorized processing for non-zero traces. Prioritizes efficiency.

    Aims to replicate the output of the provided obspy.Trace.resample method's
    FFT-based approach.

    :type stream_in: obspy.core.stream.Stream
    :param stream_in: The input Stream object.
    :type new_sampling_rate: float
    :param new_sampling_rate: The target sampling rate.
    :type window: str, tuple, or array_like, optional
    :param window: Specifies the window. See scipy.signal.get_window.
        Defaults to 'hanning'.
    :type no_filter: bool, optional
    :param no_filter: If True, deactivates automatic anti-aliasing filtering.
        Defaults to True.
    :type strict_length: bool, optional
    :param strict_length: If True, raises an error if the trace length would
        change in a way that alters its end time. Defaults to False.
    :type num_cores: int, optional
    :param num_cores: Number of cores for potential parallel processing.
        Currently, processing is sequential (num_cores=1 is effectively used).
        Defaults to 1.

    :rtype: obspy.core.stream.Stream
    :returns: A new Stream object with resampled traces.
    """
    if not isinstance(stream_in, Stream):
        raise TypeError("Input must be an Obspy Stream object.")

    # List to hold results in the original order
    results_list = [None] * len(stream_in)

    # --- Group traces ---
    # Key: (original_sampling_rate, original_npts)
    # Value: list of (original_stream_index, trace_object)
    non_zero_trace_groups_indexed = {}

    for idx, tr_init in enumerate(stream_in):
        if not isinstance(tr_init, Trace):
            # If not a Trace, preserve it as is in the output
            results_list[idx] = tr_init
            continue

        # Handle zero traces directly
        if np.all(tr_init.data == 0):
            old_npts = tr_init.stats.npts
            old_sr = tr_init.stats.sampling_rate

            new_npts_val = 0
            if old_sr != 0 and new_sampling_rate != 0:  # Avoid division by zero if sr is 0
                if old_npts > 0:
                    new_npts_val = int(round(float(old_npts) * new_sampling_rate / old_sr))
                # else new_npts_val remains 0
            elif old_sr == new_sampling_rate:  # No change in sampling rate
                new_npts_val = old_npts
            # else, if old_sr is 0 or new_sr is 0, new_npts_val remains 0 unless old_npts=0

            new_tr_data = np.zeros(new_npts_val, dtype=tr_init.data.dtype)

            resampled_tr = tr_init.copy()
            resampled_tr.data = new_tr_data
            resampled_tr.stats.sampling_rate = new_sampling_rate
            resampled_tr.stats.npts = new_npts_val  # npts is length of data array
            results_list[idx] = resampled_tr
        else:
            # Group non-zero traces
            key = (tr_init.stats.sampling_rate, tr_init.stats.npts)
            if key not in non_zero_trace_groups_indexed:
                non_zero_trace_groups_indexed[key] = []
            non_zero_trace_groups_indexed[key].append((idx, tr_init))

    # --- Process non-zero trace groups ---
    for (orig_sr, orig_npts), indexed_trace_list in non_zero_trace_groups_indexed.items():
        if orig_sr == 0:
            for orig_idx, tr_orig_obj in indexed_trace_list:
                results_list[orig_idx] = tr_orig_obj.copy()
            continue
        if orig_npts == 0:
            for orig_idx, tr_orig_obj in indexed_trace_list:
                results_list[orig_idx] = tr_orig_obj.copy()
                results_list[orig_idx].stats.sampling_rate = new_sampling_rate
            continue

        current_batch_indices = [item[0] for item in indexed_trace_list]
        current_batch_traces = [item[1] for item in indexed_trace_list]

        batch_data_for_fft = []

        for i, trace_obj in enumerate(current_batch_traces):
            target_npts_float = (orig_npts * new_sampling_rate) / orig_sr
            if strict_length and abs(target_npts_float - round(
                    target_npts_float)) > 1e-9:  # Using a small tolerance for float comparison
                raise ValueError(
                    f"Trace {trace_obj.id} (original index {current_batch_indices[i]}): "
                    "End time of trace would change and strict_length=True."
                )

            trace_data_current = trace_obj.data.copy().astype(np.float64)

            if not no_filter:
                resampling_factor = orig_sr / float(new_sampling_rate)
                if resampling_factor > 16:
                    raise ArithmeticError(
                        f"Trace {trace_obj.id} (original index {current_batch_indices[i]}): "
                        f"Automatic filter design is unstable for resampling factors (current/new = {resampling_factor:.2f}) "
                        "above 16. Manual resampling is necessary."
                    )

                temp_filter_trace = trace_obj.copy()
                temp_filter_trace.data = trace_data_current
                filter_freq = new_sampling_rate * 0.5
                temp_filter_trace.filter('lowpass_cheby_2', freq=filter_freq, maxorder=12)
                trace_data_current = temp_filter_trace.data

            batch_data_for_fft.append(trace_data_current)

        if not batch_data_for_fft:
            continue

        data_stack = np.array(batch_data_for_fft)
        new_npts_for_batch = int(round(float(orig_npts) * new_sampling_rate / orig_sr))
        resampled_batch_data_final = np.zeros((data_stack.shape[0], new_npts_for_batch), dtype=np.float64)

        if new_npts_for_batch == 0 or orig_npts == 0:
            pass
        else:
            X_complex = np.fft.rfft(data_stack, n=orig_npts, axis=1)
            X_re = X_complex.real.copy()
            X_im = X_complex.imag.copy()

            if window is not None:
                time_domain_window_vals = get_window(window, orig_npts)
                shifted_time_window_vals = np.fft.ifftshift(time_domain_window_vals)
                num_spectral_coeffs = orig_npts // 2 + 1
                w_spectral_coeffs = shifted_time_window_vals[:num_spectral_coeffs]
                X_re *= w_spectral_coeffs[np.newaxis, :]
                X_im *= w_spectral_coeffs[np.newaxis, :]

            df_orig = orig_sr / orig_npts
            f_orig_points = df_orig * np.arange(0, orig_npts // 2 + 1)

            if new_npts_for_batch == 0:
                df_new = 0
            else:
                df_new = new_sampling_rate / new_npts_for_batch

            num_new_freq_coeffs = new_npts_for_batch // 2 + 1
            f_new_points = df_new * np.arange(0, num_new_freq_coeffs)

            interpolated_X_re_list = []
            interpolated_X_im_list = []

            if f_orig_points.size == 0:
                for _ in range(X_re.shape[0]):
                    interpolated_X_re_list.append(np.zeros(f_new_points.shape, dtype=X_re.dtype))
                    interpolated_X_im_list.append(np.zeros(f_new_points.shape, dtype=X_im.dtype))
            else:
                for i_trace in range(X_re.shape[0]):
                    interpolated_X_re_list.append(np.interp(f_new_points, f_orig_points, X_re[i_trace, :]))
                    interpolated_X_im_list.append(np.interp(f_new_points, f_orig_points, X_im[i_trace, :]))

            interpolated_X_re = np.array(interpolated_X_re_list)
            interpolated_X_im = np.array(interpolated_X_im_list)
            interpolated_Y_complex = interpolated_X_re + 1j * interpolated_X_im

            resampled_batch_data_intermediate = np.fft.irfft(interpolated_Y_complex, n=new_npts_for_batch, axis=1)
            scaling_factor = float(new_npts_for_batch) / float(orig_npts)
            resampled_batch_data_final = resampled_batch_data_intermediate * scaling_factor

        for i, batch_idx in enumerate(current_batch_indices):
            original_trace_for_stats = stream_in[batch_idx]
            final_trace = original_trace_for_stats.copy()
            final_trace.data = resampled_batch_data_final[i, :]
            final_trace.stats.sampling_rate = new_sampling_rate
            final_trace.stats.npts = final_trace.data.shape[0]
            # Removed: final_trace.stats.processing.append(...)
            results_list[batch_idx] = final_trace

    for i in range(len(results_list)):
        if results_list[i] is None:
            # This case should ideally be rare if all items are either Trace or handled.
            # Copy original if it's a Trace and wasn't processed.
            if isinstance(stream_in[i], Trace):
                results_list[i] = stream_in[i].copy()
            # else it remains as it was (likely a non-Trace item handled at the beginning)

    return Stream(
        traces=[tr for tr in results_list if tr is not None])  # Filter out any Nones, though ideally none left