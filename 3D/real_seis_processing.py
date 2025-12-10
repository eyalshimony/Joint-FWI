from SPECFEM3D_interface import noise_seismograms, calculate_directional_seismograms_xyz

if __name__ == "__main__":
    calculate_directional_seismograms_xyz("/DATA/eyal/specfem3d/PROJECTS/mtinv/test5/run0002/OUTPUT_FILES/", True, False, "v", "/DATA/eyal/specfem3d/PROJECTS/mtinv/test5/run0002/RAW_SEIS_DIRECT/")
    noise_seismograms("/DATA/eyal/specfem3d/PROJECTS/mtinv/test5/run0001/RAW_SEIS_DIRECT/", "/DATA/eyal/specfem3d/PROJECTS/mtinv/test5/run0001/RAW_SEIS_DIRECT_NOISE/")

"""
import createSTATIONS
import os
import numpy as np
import bisect
import concurrent.futures
import collections

seismogramsPath = "/DATA/eyal/MTInv/REAL_SEISMOGRAMS/"
new_path = "/DATA/eyal/MTInv/REAL_SEISMOGRAMS_DIRECT/"
to_write = True

seismogramsFileNames = os.listdir(seismogramsPath)
seismogramsType = "v"
seismograms = {}

seismogramsFileNamesRType = [x for x in seismogramsFileNames if ".sem" + seismogramsType in x]
seisFNParts = [x.split('.') for x in seismogramsFileNamesRType]
seisFNParts.sort(key=lambda x: x[1])
sortedStations = [x[1] for x in seisFNParts]
sortedFNs = [".".join(x) for x in seisFNParts]


def innerFunc(station):
    stationInd = bisect.bisect_left(sortedStations, station.station)
    stationFileNames = sortedFNs[stationInd:(stationInd + 3)]
    stationParts = [x.split('.') for x in stationFileNames]
    stationSeisCompPart = [x[2] for x in stationParts]
    stationComp = [x[2] for x in stationSeisCompPart]
    seisFileNames = []
    seisStations = []
    seisFileNames.append(stationFileNames[stationComp.index("E")])
    seisFileNames.append(stationFileNames[stationComp.index("N")])
    seisFileNames.append(stationFileNames[stationComp.index("Z")])
    for i in range(len(seisFileNames)):
        fn = seisFileNames[i]
        with open(seismogramsPath + fn, 'r') as f:
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


with concurrent.futures.ProcessPoolExecutor(max_workers=100) as executor:
    for station, statSeismograms in zip(createSTATIONS.stations, executor.map(innerFunc, createSTATIONS.stations)):
        seismograms[station.station] = statSeismograms


def writeFunc(stationName):
    fn = "DS." + stationName + ".CXF.semv"
    print(fn)
    with open(new_path + fn, 'w') as f:
        ort = createSTATIONS.stationsDict[stationName].orientation
        f.write(str(ort) + "\n")
        for num in seismograms[stationName]:
            f.write(str(num[0]) + "\t" + str(num[1]) + "\n")


if to_write:
    with concurrent.futures.ProcessPoolExecutor(max_workers=100) as executor:
        collections.deque(executor.map(writeFunc, seismograms.keys()), maxlen=0)
else:
    pass

path = "/DATA/eyal/MTInv/REAL_SEISMOGRAMS_DIRECT/"
new_path = "/DATA/eyal/MTInv/REAL_SEISMOGRAMS_DIRECT_NOISE/"

seismogramsFileNames = os.listdir(path)
seismogramsType = "v"
seismograms = {}
seismogramsFileNamesRType = [x for x in seismogramsFileNames if ".sem" + seismogramsType in x]
seisFNParts = [x.split('.') for x in seismogramsFileNamesRType]
seisFNParts.sort(key=lambda x: x[1])
sortedStations = [x[1] for x in seisFNParts]
sortedFNs = [".".join(x) for x in seisFNParts]


def innerFunc(station):
    stationInd = bisect.bisect_left(sortedStations, station.station)
    fn = sortedFNs[stationInd]
    with open(path + fn, 'r') as f:
        seis_reader = f.readlines()
        ort = seis_reader[0]
        data = [float(line.split()[1]) for line in seis_reader[1:]]
        times = [float(line.split()[0]) for line in seis_reader[1:]]
        data = list(np.add(np.array(data), 0.2 * max(list(map(abs, data))) * (np.random.rand(len(data)) - 0.5)))
    with open(new_path + fn, 'w') as f:
        f.write(ort + "\n")
        for i in range(len(data)):
            f.write(str(times[i]) + "\t" + str(data[i]) + "\n")


with concurrent.futures.ProcessPoolExecutor(max_workers=100) as executor:
    collections.deque(executor.map(innerFunc, createSTATIONS.stations), maxlen=0)

"""