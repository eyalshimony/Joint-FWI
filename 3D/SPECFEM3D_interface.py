import data
import os
import numpy as np
import bisect
import concurrent.futures
import collections
import csv
import sys
import fileinput
import pandas
from distutils.dir_util import copy_tree
from objects import CMTSolution
import obspy
from shutil import copyfile, rmtree
from itertools import repeat
from pathlib import Path
import meshio

global sortedStations
global sortedFNs
global comp_seis_path
global fib_seis_path
global seismograms
global noise_seis_path
global projects_base_path
global project_name
global base_data_path
global specfem_base

projects_base_path = "/DATA/eyal/specfem3d/PROJECTS/"
specfem_base = "/DATA/eyal/specfem3d/"
project_name = "mtinv/"
base_data_path = "/DATA/eyal/"


def calc_fib_seis_xyz(station):
    global sortedStations
    global sortedFNs
    global comp_seis_path

    stationInd = bisect.bisect_left(sortedStations, station.station)
    stationFileNames = sortedFNs[stationInd:(stationInd + 3)]
    stationParts = [x.split('.') for x in stationFileNames]
    stationSeisCompPart = [x[2] for x in stationParts]
    stationComp = [x[2] for x in stationSeisCompPart]
    seisFileNames = []
    seisStations = []
    seisFileNames.append(stationFileNames[stationComp.index("X")])
    seisFileNames.append(stationFileNames[stationComp.index("Y")])
    seisFileNames.append(stationFileNames[stationComp.index("Z")])
    for i in range(len(seisFileNames)):
        fn = seisFileNames[i]
        with open(comp_seis_path + fn, 'r') as f:
            seis_reader = f.readlines()
            data = [float(line.split()[1]) for line in seis_reader]
            if i == 0:
                times = [float(line.split()[0]) for line in seis_reader]
            if 'stationSeis' not in locals():
                stationSeis = np.ndarray(shape=(len(data), 3))
            stationSeis[:, i] = data
    seisValues = station.orientation[0] * stationSeis[:, 0] + station.orientation[1] * stationSeis[:, 1] + \
                 station.orientation[2] * stationSeis[:, 2]
    return np.column_stack((times, seisValues))


def writeFunc(stationName):
    global fib_seis_path
    global seismograms

    fn = "DS." + stationName + ".CXF.semv"
    print(fn)
    with open(fib_seis_path + fn, 'w') as f:
        for num in seismograms[stationName]:
            f.write(str(num[0]) + "\t" + str(num[1]) + "\n")


def calculate_directional_seismograms_xyz(seismogramsPath, to_write, as_stream, seismograms_type, new_path="",
                                      relative_time_shift=0):
    global sortedStations
    global sortedFNs
    global comp_seis_path
    global fib_seis_path
    global seismograms

    comp_seis_path = seismogramsPath
    fib_seis_path = new_path

    seismogramsFileNames = os.listdir(seismogramsPath)
    if as_stream:
        seismograms = obspy.Stream()
    else:
        seismograms = {}

    seismogramsFileNamesRType = [x for x in seismogramsFileNames if ".sem" + seismograms_type in x]
    seisFNParts = [x.split('.') for x in seismogramsFileNamesRType]
    seisFNParts.sort(key=lambda x: x[1])
    sortedStations = [x[1] for x in seisFNParts]
    sortedFNs = [".".join(x) for x in seisFNParts]
    max_seis_value = 0.0
    stations_list = []
    for station in data.get_stations_xyz():
        if station.station in sortedStations:
            stations_list.append(station)

    with concurrent.futures.ProcessPoolExecutor(max_workers=100) as executor:
        try:
            for station, statSeismograms in zip(stations_list, executor.map(calc_fib_seis_xyz, stations_list)):
                max_value = np.max(np.abs(statSeismograms[:, 1]))
                if max_value > max_seis_value:
                    max_seis_value = np.max(np.abs(statSeismograms[:, 1]))
                if as_stream:
                    seis_data = statSeismograms[:, 1]
                    times = statSeismograms[:, 0] + relative_time_shift
                    dt = times[1] - times[0]
                    srate = 1/dt
                    location_string = "lat=" + str(station.latitude) + ";long=" + str(station.longitude) + ";elev=" + \
                                      str(station.elevation) + ";bur=" + str(station.burial) + ";ort=" + \
                                      str(station.orientation)
                    seismograms.append(obspy.Trace(seis_data, {'sampling_rate': srate, 'delta': dt, 'network': station.network,
                                                          'station': station.station, 'channel': "F",
                                                          'npts': len(seis_data), 'starttime': obspy.UTCDateTime(times[0]),
                                                          'endtime': obspy.UTCDateTime(times[-1]),
                                                          'location': location_string}))
                else:
                    seismograms[station.station] = statSeismograms
        except:
            pass

    data.set_max_seis_value(max_seis_value)
    if to_write:
        if not os.path.isdir(fib_seis_path):
            os.makedirs(fib_seis_path)
        with concurrent.futures.ProcessPoolExecutor(max_workers=100) as executor:
            collections.deque(executor.map(writeFunc, seismograms.keys()), maxlen=0)
    else:
        return seismograms


def write_noised_seis(station):
    global fib_seis_path
    global noise_seis_path
    global sortedStations
    global sortedFNs

    stationInd = bisect.bisect_left(sortedStations, station.station)

    fn = sortedFNs[stationInd]
    with open(fib_seis_path + fn, 'r') as f:
        seis_reader = f.readlines()
        seis_data = [float(line.split()[1]) for line in seis_reader]
        times = [float(line.split()[0]) for line in seis_reader]
        rng = np.random.default_rng()
        seis_data = list(np.add(np.array(seis_data),
                                0.1 * data.get_max_seis_value() * (rng.random((len(seis_data,))) - 0.5)))
    with open(noise_seis_path + fn, 'w') as f:
        for i in range(len(seis_data)):
            f.write(str(times[i]) + "\t" + str(seis_data[i]) + "\n")


def noise_seismograms(path, new_path):
    global fib_seis_path
    global noise_seis_path
    global sortedStations
    global sortedFNs

    fib_seis_path = path
    noise_seis_path = new_path

    seismogramsFileNames = os.listdir(path)
    seismogramsType = "v"
    seismogramsFileNamesRType = [x for x in seismogramsFileNames if ".sem" + seismogramsType in x]
    seisFNParts = [x.split('.') for x in seismogramsFileNamesRType]
    seisFNParts.sort(key=lambda x: x[1])
    sortedStations = [x[1] for x in seisFNParts]
    sortedFNs = [".".join(x) for x in seisFNParts]

    if not os.path.isdir(noise_seis_path):
        os.makedirs(noise_seis_path)
    with concurrent.futures.ProcessPoolExecutor(max_workers=100) as executor:
        collections.deque(executor.map(write_noised_seis, data.get_stations()), maxlen=0)


def write_STATIONS(stations, path):
    with open(path + 'STATIONS', 'w') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerows([(stationData.station, stationData.network, stationData.latitude, stationData.longitude,
                      stationData.elevation, stationData.burial) for stationData in stations])


def write_CMTSOLUTION(solution, name):
    global projects_base_path
    global project_name

    path = projects_base_path + project_name + name
    with open(path + '/DATA/CMTSOLUTION', 'w') as f:
        f.write("PDE  1999 01 01 00 00 00.00  750 500 -700 1.0 1.0 test\n")
        f.write('event name:\t' + solution.event_name + "\n")
        f.write('time shift:\t' + str(solution.time_shift) + "\n")
        f.write('half duration:\t' + str(solution.half_duration) + "\n")
        f.write('latorUTM:\t' + str(solution.latorUTM) + "\n")
        f.write('longorUTM:\t' + str(solution.longorUTM) + "\n")
        f.write('depth:\t' + str(solution.depth) + "\n")
        f.write('Mrr:\t' + str(solution.Mrr) + "\n")
        f.write('Mtt:\t' + str(solution.Mtt) + "\n")
        f.write('Mpp:\t' + str(solution.Mpp) + "\n")
        f.write('Mrt:\t' + str(solution.Mrt) + "\n")
        f.write('Mrp:\t' + str(solution.Mrp) + "\n")
        f.write('Mtp:\t' + str(solution.Mtp))


def check_event_existence(name):
    global projects_base_path
    global project_name

    path_to_check = projects_base_path + project_name + name
    return os.path.exists(path_to_check)


def copy_from_template(name):
    global projects_base_path
    global project_name

    copy_from = projects_base_path + project_name + "template"
    copy_to = projects_base_path + project_name + name
    copy_tree(copy_from, copy_to)


def run_modelling(name):
    global projects_base_path
    global project_name

    curr_dir = os.getcwd()
    os.chdir(projects_base_path + project_name + name)
    os.system("./change_simulation_type.pl -f")
    for line in fileinput.input(projects_base_path + project_name + name + "/DATA/Par_file", inplace=1):
        if "SAVE_FORWARD" in line:
            line = line.replace("true", "false")
        if "APPROXIMATE_HESS_KL" in line:
            line = line.replace("true", "false")
        if "UNDO_ATTENUATION_AND_OR_PML" in line:
            line = line.replace("true", "false")
        sys.stdout.write(line)
    os.system("./run_this_example.sh")
    os.chdir(curr_dir)


def run_modelling_for_FWI_multi(names):
    global projects_base_path
    global project_name

    curr_dir = os.getcwd()
    for name in names:
        os.chdir(projects_base_path + project_name + name)
        os.system("./change_simulation_type.pl -f")
        for line in fileinput.input(projects_base_path + project_name + name + "/DATA/Par_file", inplace=1):
            if "SAVE_FORWARD" in line:
                line = line.replace("false", "true")
            if "APPROXIMATE_HESS_KL" in line:
                line = line.replace("true", "false")
            if "UNDO_ATTENUATION_AND_OR_PML" in line:
                line = line.replace("false", "true")
            sys.stdout.write(line)
    os.chdir(projects_base_path + project_name + names[0].split("/")[0])
    os.system("./run_this_example.sh")
    os.chdir(curr_dir)


def read_seismogram(station_file_path):
    with open(station_file_path, 'r') as f:
        station_file_name = station_file_path.split("/")[-1]
        lines = f.readlines()
        data = [line.split() for line in lines]
        data = [list(map(float, i)) for i in data]
        station_parts = station_file_name.split(".")
        dt = data[1][0] - data[0][0]
        srate = 1 / dt
        network = station_parts[0]
        station = station_parts[1]
        channel = station_parts[2][-1]
        npts = len(data)
        data = np.asarray(data)
        times = data[:, 0]
        data = data[:, 1]
        data = obspy.Trace(data, {'sampling_rate': srate, 'delta': dt, 'network': network, 'station': station,
                                  'channel': channel, 'npts': npts, 'starttime': obspy.UTCDateTime(times[0]),
                                  'endtime': obspy.UTCDateTime(times[-1])})
        return data


def read_seismograms(path, letter_filter="", number_filter=[]):
    seis_files = os.listdir(path)
    seis_files_filtered = [f for f in seis_files if data.get_stations()[0].network + "." in f]
    full_file_paths = [path + s for s in seis_files_filtered]
    seismograms = obspy.Stream()
    with concurrent.futures.ProcessPoolExecutor(max_workers=100) as executor:
        for trace in executor.map(read_seismogram, full_file_paths):
            station_data = data.get_stations_dict()[trace.stats.station]
            location_string = "lat=" + str(station_data.latitude) + ";long=" + str(station_data.longitude) + \
                              ";elev=" + str(station_data.elevation) + ";bur=" + str(station_data.burial) + ";ort=" + \
                              str(station_data.orientation)
            trace.stats.location = location_string
            letter_part = ''.join(list(filter(str.isupper, trace.stats.station)))
            if not letter_filter or letter_filter == letter_part:
                station_number = int(''.join(list(filter(str.isdigit, trace.stats.station))))
                if not number_filter or number_filter[0] <= station_number <= number_filter[1]:
                    seismograms.append(trace)

    return seismograms


def read_observed_seismograms(name, noise, letter_filter="", number_filter=[]):
    global base_data_path
    global project_name
    global noise_seis_path

    path = projects_base_path + project_name + name + "/RAW_SEIS_DIRECT"
    if noise:
        path += "_NOISE"
    noise_seis_path = path + "/"

    return read_seismograms(noise_seis_path, letter_filter, number_filter)


def read_synthetic_seismograms(name, seismograms_type, time_shift=0):
    global projects_base_path
    global project_name

    os.system("rm -rf " + projects_base_path + project_name + name + "/SYNTHETIC_SEIS/")
    os.system("mkdir " + projects_base_path + project_name + name + "/SYNTHETIC_SEIS/")
    os.system("cp -f " + projects_base_path + project_name + name + "/OUTPUT_FILES/*" + " " +
              projects_base_path + project_name + name + "/SYNTHETIC_SEIS/")
    path = projects_base_path + project_name + name + "/SYNTHETIC_SEIS/"
    return calculate_directional_seismograms_xyz(path, False, True, seismograms_type, "", time_shift)


def create_STATIONS_ADJOINT_from_STATIONS(name, stations_filter=[]):
    global projects_base_path
    global project_name

    stations_path = projects_base_path + project_name + name + "/DATA/STATIONS"
    stations_adjoint_path = projects_base_path + project_name + name + "/DATA/STATIONS_ADJOINT"
    if not stations_filter:
        copyfile(stations_path, stations_adjoint_path)
    else:
        with open(stations_path, 'r') as f:
            lines = f.readlines()
            lines_parts = [l.split() for l in lines]
            station_names = [l[0] for l in lines_parts]
            d = {item: idx for idx, item in enumerate(station_names)}
            indices = [d.get(item) for item in stations_filter]
            wanted_lines = map(lines.__getitem__, indices)
            with open(stations_adjoint_path, 'w') as fw:
                fw.write("\n".join(wanted_lines))
                
                
def write_adjoint_source_xyz(adjoint_source, path):
    file_name_x = adjoint_source.stats.network + "." + adjoint_source.stats.station + ".CXX.adj"
    file_name_y = adjoint_source.stats.network + "." + adjoint_source.stats.station + ".CXY.adj"
    file_name_z = adjoint_source.stats.network + "." + adjoint_source.stats.station + ".CXZ.adj"
    start_time = adjoint_source.stats.starttime - obspy.UTCDateTime(0)
    end_time = adjoint_source.stats.endtime - obspy.UTCDateTime(0)
    dt = adjoint_source.stats.delta
    times = np.arange(start_time, end_time + dt / 2, dt)
    location_parts = adjoint_source.stats.location.split(";")
    orientation_data = location_parts[-1].split("=")[1].strip('[]').split()
    orientation = [float(part) for part in orientation_data]
    source_x = orientation[0] * adjoint_source.data
    source_y = orientation[1] * adjoint_source.data
    source_z = orientation[2] * adjoint_source.data
    with open(path + file_name_x, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(times, source_x))
    with open(path + file_name_y, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(times, source_y))
    with open(path + file_name_z, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(times, source_z))


def write_adjoint_sources_xyz(adjoint_sources, name):
    global projects_base_path
    global project_name

    create_STATIONS_ADJOINT_from_STATIONS(name)
    path = projects_base_path + project_name + name + "/SEM/"
    Path(path).mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=100) as executor:
        collections.deque(executor.map(write_adjoint_source_xyz, adjoint_sources, repeat(path)), maxlen=0)


def run_structural_adjoint_modelling_multi(names):
    global projects_base_path
    global project_name

    curr_dir = os.getcwd()
    for name in names:
        os.chdir(projects_base_path + project_name + name)
        os.system("./change_simulation_type.pl -b")
        for line in fileinput.input(projects_base_path + project_name + name + "/DATA/Par_file", inplace=1):
            if "APPROXIMATE_HESS_KL" in line:
                line = line.replace("false", "true")
            if "MOVIE_VOLUME" in line:
                line = line.replace("true", "false")
            if "UNDO_ATTENUATION_AND_OR_PML" in line:
                line = line.replace("false", "true")
            sys.stdout.write(line)
    os.chdir(projects_base_path + project_name + names[0].split("/")[0])
    os.system("./run_pure_solver.sh")
    os.chdir(curr_dir)


def read_source_location_kernels(name):
    global projects_base_path
    global project_name

    path = projects_base_path + project_name + name + "/OUTPUT_FILES/"
    with open(path + "src_frechet.000001", "r") as f:
        kernels = f.read().splitlines()
    return float(kernels[6]), float(kernels[7]), -float(kernels[8])


def read_adjoint_strain_tensor_at_source(name):
    global projects_base_path
    global project_name

    path = projects_base_path + project_name + name + "/OUTPUT_FILES/"
    files = os.listdir(path)
    adjoint_strain_tensor = obspy.Stream()
    for fn in files:
        if "NT.S" and ".sem" in fn:
            fn_parts = fn.split(".")
            if fn_parts[2][0] == "S":
                with open(path + fn, 'r') as f:
                    data = f.readlines()
                    data = [line.split() for line in data]
                    data = [list(map(float, i)) for i in data]
                    data = np.asarray(data)
                    dt = data[1, 0] - data[0, 0]
                    srate = 1 / dt
                    data = obspy.Trace(data[:, 1],
                                       {'sampling_rate': srate, 'delta': dt, 'network': "ET", 'station': name,
                                        'channel': fn_parts[2][1:3], 'npts': len(data[:, 1]),
                                        'starttime': obspy.UTCDateTime(data[0, 0]),
                                        'endtime': obspy.UTCDateTime(data[-1, 0])})
                    adjoint_strain_tensor.append(data)

    return adjoint_strain_tensor


def create_green_projects_multi(name, source, time_shift=0):
    global projects_base_path
    global project_name

    source_path = projects_base_path + project_name + name
    for i in range(6):
        dst_path = projects_base_path + project_name + name + "/green" + str(i + 1) + "/"
        if os.path.exists(dst_path):
            rmtree(dst_path)
        os.mkdir(dst_path)
        os.system("cp -r " + source_path + "/../bin " + dst_path)
        os.system("mkdir " + dst_path + "/DATA")
        os.system("cp -r " + source_path + "/DATA/meshfem3D_files " + dst_path + "/DATA")
        os.system("cp " + source_path + "/DATA/Par_file " + dst_path + "/DATA")
        os.system("cp " + source_path + "/DATA/STATIONS " + dst_path + "/DATA")
        os.system("cp " + source_path + "/DATA/tomography_model_*_it0.xyz " + dst_path + "/DATA")
        os.system("cp " + source_path + "/../change_simulation_type.pl " + dst_path)
        os.system("cp " + source_path + "/../../test/run_this_example.sh " + dst_path)
        for filename in os.listdir(dst_path + "/DATA"):
            fn = os.path.join(dst_path + "/DATA/", filename)
            if "_it0" in fn:
                new_fn = fn.replace("_it0", "")
                os.rename(fn, new_fn)
        components = np.zeros(6)
        components[i] = 10**20

        solution = CMTSolution("green" + str(i+1), time_shift, 0, source.latorUTM, source.longorUTM, source.depth,
                               *components)
        write_CMTSOLUTION(solution, name + "/green" + str(i+1))
        with open(projects_base_path + project_name + name + "/green" + str(i+1) + "/DATA/Par_file", "r") as f:
            par_file = f.readlines()
            for j in range(len(par_file)):
                line = par_file[j]
                if "SAVE_SEISMOGRAMS_" in line:
                    if "SAVE_SEISMOGRAMS_VELOCITY" in line:
                        par_file[j] = line.replace("false", "true")
                    else:
                        par_file[j] = line.replace("true", "false")

        with open(projects_base_path + project_name + name + "/green" + str(i+1) + "/DATA/Par_file", "w") as f:
            f.writelines(par_file)


def run_green_modelling(name):
    for i in range(6):
        run_modelling(name + "/green" + str(i+1))


def run_green_modelling_parallel(name):
    names = []
    for i in range(6):
        names.append(name + "/green" + str(i+1))
        for line in fileinput.input(projects_base_path + project_name + names[i] + "/DATA/Par_file", inplace=1):
            if "NUMBER_OF_SIMULTANEOUS_RUNS" in line:
                line = line.replace("2", "1")
            sys.stdout.write(line)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        collections.deque(executor.map(run_modelling, names), maxlen=0)


def read_green_functions(name):
    green_functions = []

    for i in range(6):
        green_functions_curr = read_synthetic_seismograms(name + "/green" + str(i+1), "v")
        green_functions.append(green_functions_curr)

    return green_functions


def read_source_time_function(name):
    global projects_base_path
    global project_name

    path = projects_base_path + project_name + name + "/plot_source_time_function.txt"
    with open(path, "r") as f:
        lines = f.readlines()
        line_parts = [line.split() for line in lines]
        line_parts = [list(map(float, i)) for i in line_parts]
        data = np.asarray(line_parts)
        dt = data[1,0] - data[0,0]
        start_time = obspy.UTCDateTime(data[0,0])
        end_time = obspy.UTCDateTime(data[-1, 0])
        return obspy.Trace(data[:, 1], {'sampling_rate': 1/dt, 'delta': dt, 'station': name, 'npts': len(lines),
                                        'starttime': start_time,'endtime': end_time})


def copy_source_time_function(name):
    global projects_base_path
    global project_name

    in_path = projects_base_path + project_name + name + "/OUTPUT_FILES/plot_source_time_function.txt"
    out_path = projects_base_path + project_name + name + "/plot_source_time_function.txt"
    copyfile(in_path, out_path)


def read_tomographic_models(name, it_num):
    global projects_base_path
    global project_name

    path = projects_base_path + project_name + name + "/DATA/"
    fns = os.listdir(path)
    tomo_fns = [fn for fn in fns if "tomography_model_" in fn]
    if it_num < 0:
        tomo_fns_curr = [fn for fn in tomo_fns if "_it" not in fn and "_true" not in fn]
    else:
        tomo_fns_curr = [fn for fn in tomo_fns if ("_it" + str(it_num) + ".") in fn and "_true" not in fn]
    tomo_fns_curr.sort()
    tomo_models = []
    for fn in tomo_fns_curr:
        tomo_models.append(pandas.read_csv(path + fn, skiprows=3, delimiter=" ").values)
    return tomo_models


def copy_model_to_runs(name, run_num, model_num):
    global projects_base_path
    global project_name

    for j in range(run_num):
        os.system("cp " + projects_base_path + project_name + name + "/DATA/tomography_model_" + str(model_num + 1) +
                  ".xyz " + projects_base_path + project_name + name + "/run000" + str(j + 1) +
                  "/DATA/tomography_model_" + str(model_num + 1) + ".xyz ")


def read_kernels(name, kernel_type, is_clipped, is_smoothed):
    global projects_base_path
    global project_name

    path = projects_base_path + project_name + name + "/OUTPUT_FILES/DATABASES_MPI/"
    fn = kernel_type + "_kernel"
    if is_clipped:
        fn += "_clip"
    if is_smoothed:
        fn += "_smooth"
    fn += ".vtk"
    return meshio.read(path + fn)


def sum_kernels(name):
    global projects_base_path
    global project_name

    curr_dir = os.getcwd()
    os.chdir(projects_base_path + project_name + name)
    os.system("mpirun -np 100 ./bin/xsum_kernels")
    os.chdir(curr_dir)


def combine_kernels_multi(name, kernel_type):
    global projects_base_path
    global project_name

    curr_dir = os.getcwd()
    os.chdir(projects_base_path + project_name + name)
    os.system("cp run0001/DATA/Par_file DATA/")
    os.system("cp run0001/DATA/CMTSOLUTION DATA/")
    for line in fileinput.input("DATA/Par_file", inplace=1):
        if "LOCAL_PATH" in line:
            line = line.replace("./OUTPUT_FILES/DATABASES_MPI", "./run0001/OUTPUT_FILES/DATABASES_MPI")
        sys.stdout.write(line)
    path = "./OUTPUT_SUM"
    target_path = "./OUTPUT_FILES/DATABASES_MPI"
    fn = kernel_type + "_kernel"
    os.system("./bin/xcombine_vol_data_vtk 0 49 " + fn + " " + path + " " + target_path + " 1")
    os.system("rm -f DATA/Par_file")
    os.system("rm -f DATA/CMTSOLUTION")
    os.chdir(curr_dir)


def write_raw_model(model, name, num, it_num=-1):
    global projects_base_path
    global project_name

    path = projects_base_path + project_name + name + "/DATA/"
    xs = list(set(model[:, 0]))
    ys = list(set(model[:, 1]))
    zs = list(set(model[:, 2]))
    xs.sort()
    ys.sort()
    zs.sort()
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    min_z = min(zs)
    max_z = max(zs)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    dz = zs[1] - zs[0]
    nx = len(xs)
    ny = len(ys)
    nz = len(zs)
    min_vp = min(model[:, 3])
    min_vs = min(model[:, 4])
    min_rho = min(model[:, 5])
    max_vp = max(model[:, 3])
    max_vs = max(model[:, 4])
    max_rho = max(model[:, 5])
    text_to_write = str(min_x) + " " + str(min_y) + " " + str(min_z) + " " + str(max_x) + " " + str(max_y) + " " + \
                    str(max_z) + str("\n") + str(dx) + " " + str(dy) + " " + str(dz) + "\n" + str(nx) + " " + str(ny) \
                    + " " + str(nz) + "\n" + str(min_vp) + " " + str(max_vp) + " " + str(min_vs) + " " + str(max_vs) + \
                    " " + str(min_rho) + " " + str(max_rho)
    fn = "tomography_model_" + str(num+1) + ".xyz"
    np.savetxt(path + fn, model, header=text_to_write, comments='')
    if it_num >= 0:
        fn = "tomography_model_" + str(num+1) + "_it" + str(it_num) + ".xyz"
        np.savetxt(path + fn, model, header=text_to_write, comments='')


def copy_to_greens(base_path):
    for i in range(6):
        os.system("cp -r " + base_path + "/observed " + base_path + "/green" + str(i+1) + "/observed")


def write_tomographic_file_xyz(name, vp, vs, rho, x0, y0, z0, dx, dy, dz, index):
    end_x = x0 + dx * (np.ma.size(vp, 0) - 1)
    end_y = y0 + dy * (np.ma.size(vp, 1) - 1)
    end_z = z0 + dz * (np.ma.size(vp, 2) - 1)
    nx = np.ma.size(vp, 0)
    ny = np.ma.size(vp, 1)
    nz = np.ma.size(vp, 2)
    vp_min = np.min(vp)
    vp_max = np.max(vp)
    vs_min = np.min(vs)
    vs_max = np.max(vs)
    rho_min = np.min(rho)
    rho_max = np.max(rho)
    dx = (end_x - x0) / (nx - 1)
    dy = (end_y - y0) / (ny - 1)

    text_to_write = str(x0) + " " + str(y0) + " " + str(z0) + " " + str(end_x) + " " + str(end_y) + " " + str(end_z) + \
                    str("\n") + str(dx) + " " + str(dy) + " " + str(dz) + "\n" + str(nx) + " " + str(ny) + " " + \
                    str(nz) + "\n" + str(vp_min) + " " + str(vp_max) + " " + str(vs_min) + " " + str(vs_max) + " " + \
                    str(rho_min) + " " + str(rho_max) + "\n"
    for i in range(np.ma.size(vp, 2)):
        for j in range(np.ma.size(vp, 1)):
            for k in range(np.ma.size(vp, 0)):
                text_to_write += (str(x0 + k*dx) + " " + str(y0 + j*dy) + " " + str(z0 + i*dz)) + " " + \
                                 str(vp[k, j, i]) + " " + str(vs[k, j, i]) + " " + str(rho[k, j, i]) + "\n"

    with open(projects_base_path + project_name + name + "/DATA/tomography_model_" + str(index) + ".xyz", "w") as f:
        f.write(text_to_write)


def write_iteration_model(name, it_num):
    global projects_base_path
    global project_name

    curr_path = projects_base_path + project_name + name + "/DATA/"
    dir_files = os.listdir(curr_path)
    rel_files = [fn for fn in dir_files if "tomography_model_" in fn and "_it" not in fn]
    for fn in rel_files:
        fn_parts = fn.split(".")
        new_fn = fn_parts[0] + "_it" + str(it_num) + "." + fn_parts[1]
        copyfile(curr_path + fn, curr_path + new_fn)


def models_to_1d_vector(models):
    orig_model_list = []
    for model in models:
        orig_model_list.extend(list(model[:, 3]))
    for model in models:
        orig_model_list.extend(list(model[:, 4]))
    for model in models:
        orig_model_list.extend(list(model[:, 5]))
    orig_model = np.asarray(orig_model_list)
    return orig_model


def write_kernel(name, mesh, kernel_name):
    global projects_base_path
    global project_name

    if not os.path.isdir(projects_base_path + project_name + name + "/KERNELS/"):
        os.mkdir(projects_base_path + project_name + name + "/KERNELS/")
    mesh.write(projects_base_path + project_name + name + "/KERNELS/" + kernel_name + "_kernel.vtk",
               file_format="vtk42", binary=True)