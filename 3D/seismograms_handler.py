import data
import numpy as np
import concurrent.futures


def transform_trace_into_frequency_domain(seismogram):
    freq_seismogam = np.fft.fft(seismogram.data)
    stats = seismogram.stats
    return freq_seismogam, stats


def transform_stream_into_frequency_domain(seismograms):
    seis_num = len(seismograms)
    nt = len(seismograms[0])
    transformed_data = np.empty((seis_num, nt), dtype=np.complex_)
    dt = 0
    total_time = 0
    stations_dict = data.get_stations_dict().copy()
    stations_to_use = []
    for seismogram in seismograms:
        stations_to_use.append(seismogram.stats.station)
    stations_to_use = list(set(stations_to_use))
    stations_dict_keys = list(stations_dict.keys()).copy()
    for key in stations_dict_keys:
        if key not in stations_to_use:
            del stations_dict[key]
    with concurrent.futures.ProcessPoolExecutor(max_workers=100) as executor:
        for seismogram, stats in executor.map(transform_trace_into_frequency_domain, seismograms):
            index = list(stations_dict.keys()).index(stats.station)
            transformed_data[index] = seismogram
            dt = stats.delta
            total_time = stats.endtime - stats.starttime
    return transformed_data, dt, total_time


def split_stream_into_substreams(stream, num):
    traces_per_stream = int(np.ceil(len(stream) / num))
    substreams = []

    for i in range(num):
        substreams.append(stream[(i*traces_per_stream):((i+1)*traces_per_stream)])

    return substreams