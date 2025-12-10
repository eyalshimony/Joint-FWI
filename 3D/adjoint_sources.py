import numpy as np
import obspy
import scipy.io
from obspy.signal import freqattributes


def calc_frequencies(t):
    dt = t[1] - t[0]
    f_min = -1/(2*dt)
    df = 1/(len(t)*dt)
    f0 = -f_min
    freqs = np.mod(np.linspace(0, 2*f0-df, len(t))+f0, 2*f0) - f0

    return freqs


def calc_adjoint_source(modelled_seismograms, real_seismograms, min_freq, max_freq, add_envelopes,
                        max_seis_level, return_misfit_only, sim_starttime=None, index=None, is_time_der=True):
    min_num = 149
    n = 2
    taper_perc = 0.05
    nt = len(modelled_seismograms[0])
    seis_num = len(modelled_seismograms)
    orig_stats = [tr.stats.copy() for tr in modelled_seismograms]
    dt = orig_stats[0].delta

    modelled_seismograms = modelled_seismograms.copy()
    real_seismograms = real_seismograms.copy()

    timesteps_append_begin = np.max(
        [np.ceil((real_seismograms[0].stats.starttime - modelled_seismograms[0].stats.starttime) / dt), 0])
    timesteps_append_end = np.max(
        [np.ceil((modelled_seismograms[0].stats.endtime - real_seismograms[0].stats.endtime) / dt), 0])
    real_seismograms.trim(starttime=real_seismograms[0].stats.starttime - dt * timesteps_append_begin,
                          endtime=real_seismograms[0].stats.endtime + dt * timesteps_append_end,
                          pad=True, fill_value=0)
    real_seismograms.interpolate(1/dt, method="lanczos", a=8, window="blackmann",
                                 starttime=modelled_seismograms[0].stats.starttime, npts=modelled_seismograms[0].stats.npts)
    real_seismograms = real_seismograms.filter("bandpass", freqmin=min_freq, freqmax=max_freq, zerophase=True)
    modelled_seismograms = modelled_seismograms.filter("bandpass", freqmin=min_freq, freqmax=max_freq, zerophase=True)

    real_seismograms = real_seismograms.taper(type="hann", max_percentage=taper_perc, side="both")
    modelled_seismograms = modelled_seismograms.taper(type="hann", max_percentage=taper_perc, side="both")

    f0 = freqattributes.central_frequency_unwindowed(real_seismograms[0].data, 1/dt)
    try:
        dt_new = (real_seismograms[0].stats.npts-1)*dt/(max(np.ceil(nt * dt * 2 * max_freq), min_num) - 1.0)

        real_seismograms = real_seismograms.interpolate(1/dt_new, method="lanczos", a=8, window="blackmann",
                                                        npts=int(max(np.ceil(nt * dt * 2 * max_freq), min_num)))
        modelled_seismograms = modelled_seismograms.interpolate(1/dt_new, method="lanczos", a=8, window="blackmann",
                                                                npts=int(max(np.ceil(nt * dt * 2 * max_freq), min_num)))
    except ValueError:
        dt_new = (real_seismograms[0].stats.npts - 1) * dt / (max(np.ceil(nt * dt * 2 * max_freq), min_num) - 1.0) - 10**(-17)

        real_seismograms = real_seismograms.interpolate(1 / dt_new, method="lanczos", a=8, window="blackmann",
                                                        npts=int(max(np.ceil(nt * dt * 2 * max_freq), min_num)))
        modelled_seismograms = modelled_seismograms.interpolate(1 / dt_new, method="lanczos", a=8, window="blackmann",
                                                                npts=int(max(np.ceil(nt * dt * 2 * max_freq), min_num)))
    new_nt = len(modelled_seismograms[0].data)

    t = np.arange(0, new_nt*dt_new, dt_new)
    t = np.tile(t, (len(t), 1))
    tau = t.transpose()
    width = 1/f0/2
    dom = 1/dt_new/new_nt
    h = (np.pi * width**2)**(-0.25) * np.exp(-(tau-t)**2 / (2*width**2))
    gabor_mod_seis = np.zeros((new_nt, new_nt, seis_num), dtype=np.complex_)
    gabor_real_seis = np.zeros((new_nt, new_nt, seis_num), dtype=np.complex_)
    cc_transform = np.zeros((new_nt, new_nt, seis_num), dtype=np.complex_)

    for i in range(seis_num):
        integrand_mod = np.conj(h) * np.tile(modelled_seismograms[i].data, (new_nt, 1))
        integrand_real = np.conj(h) * np.tile(real_seismograms[i].data, (new_nt, 1))
        gabor_mod_seis[:, :, i] = dt_new / np.sqrt(2*np.pi) * np.fft.fft(integrand_mod)
        tmp_gabor_real = np.fft.fft(integrand_real)
        gabor_real_seis[:, :, i] = dt_new / np.sqrt(2*np.pi) * tmp_gabor_real
        cc_transform[:, :, i] = gabor_mod_seis[:, :, i] * np.conj(tmp_gabor_real)

    seis_phase_diff = np.angle(cc_transform) / np.pi
    seis_env_mod = np.abs(gabor_mod_seis)
    seis_env_real = np.abs(gabor_real_seis)

    me = max_seis_level / 9000

    if add_envelopes:
        me2 = np.amax(np.amax(seis_env_mod, 0), 0) / 5000
        me3 = np.amax(np.amax((seis_env_mod + seis_env_real)**2, 0), 0) / 500
        me4 = np.amax(np.amax(seis_env_mod + seis_env_real, 0), 0) / 500
        seis_env_norm_diff = 2 * (seis_env_mod - seis_env_real) / \
                             (seis_env_mod + seis_env_real + np.tile(me4, (new_nt, new_nt, 1)))

    oms = calc_frequencies(np.arange(0, new_nt*dt_new, dt_new))
    thres = np.abs(oms) <= max_freq
    n_lar = np.zeros(np.size(oms))
    n_sm = np.zeros(np.size(oms))
    n_lar[np.logical_not(thres)] = 1
    n_sm[thres] = 1

    w_p = 1 - np.exp(-seis_env_real**2 / me)
    w_p = np.transpose(w_p, (2, 0, 1))
    w_p = w_p * np.tile(1 - np.exp(-(oms / min_freq)**2), (seis_num, new_nt, 1))
    w_p = w_p * np.tile(np.exp(-10*np.abs(np.abs(oms) / max_freq - 1)) * n_lar + n_sm, (seis_num, new_nt, 1))
    w_p = np.transpose(w_p, (1, 2, 0))
    div_factor = np.zeros((np.ma.size(w_p, 2)))
    seis_to_change = [np.abs(seismogram.data).max() / np.sqrt(max_seis_level) > 0.1 for seismogram in real_seismograms]
    div_factor[seis_to_change] = \
        1 / np.max(np.max(w_p[:, :, seis_to_change], axis=0), axis=0)
    w_p *= div_factor
    w_p_sum = np.sum(np.sum(w_p, axis=0), axis=0)
    scipy.io.savemat("wp.mat", {"wp": w_p_sum})

    test_field = seis_phase_diff * w_p
    indices_to_zero = np.where(np.sum(np.sum(np.abs(np.diff(test_field, axis=0)) > 0.5, axis=0), axis=0) +
                               np.sum(np.sum(np.abs(np.diff(test_field, axis=1)) > 0.5, axis=0), axis=0) > 4)

    w_phase_difference = (seis_phase_diff * w_p) ** n
    phase_misfit = np.sqrt(np.sum(np.sum(w_phase_difference * dt_new * dom, axis=0), axis=0))
    phase_misfit_signed = np.sum(np.sum((seis_phase_diff * w_p * dt_new * dom)[:, :75, :], axis=0), axis=0)
    phase_sources = obspy.Stream()
    mp = np.amax(np.amax(seis_env_mod**2, 0), 0) / 500
    envelope_misfit = np.zeros(np.size(phase_misfit))
    envelope_sources = obspy.Stream()

    if add_envelopes:
        w_e = w_p
        w_envelope_difference = (seis_env_norm_diff * w_p) ** n
        envelope_misfit = np.sqrt(np.sum(np.sum(w_envelope_difference * dt_new * dom, axis=0), axis=0))

    if return_misfit_only:
        return phase_misfit, envelope_misfit

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
            integrand = 4*w_e[:, :, i]**n * seis_env_norm_diff[:, :, i]**(n-1) * gabor_mod_seis[:, :, i] * \
                        seis_env_real[:, :, i] / (seis_env_mod[:, :, i] + me2[i]) / \
                        ((seis_env_mod[:, :, i] + seis_env_real[:, :, i])**2 + me3[i])
            i_integrand = np.fft.ifft(integrand)
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
        return phase_sources, envelope_sources, phase_misfit, envelope_misfit, phase_misfit_signed, indices_to_zero[0]
    else:
        return phase_sources, envelope_sources, phase_misfit, envelope_misfit, phase_misfit_signed, indices_to_zero[0], index
