import numpy as np
from SPECFEM3D_interface import write_STATIONS
from objects import Station, StationData
import data
import collections

DASspacing = 10
elevation = 0.0
network = "DS"

# Boreholes
boreholesZs = np.arange(0.0, 2000.0+0.00001, DASspacing)
borehole1X = 0
borehole2X = 0
borehole1Y = -9000
borehole2Y = 9000
boreholesOrt = np.array([0.0, 0.0, 1.0])
borehole1Xs = np.zeros(len(boreholesZs)) + borehole1X
borehole1Ys = np.zeros(len(boreholesZs)) + borehole1Y
borehole2Xs = np.zeros(len(boreholesZs)) + borehole2X
borehole2Ys = np.zeros(len(boreholesZs)) + borehole2Y
borehole1Ort = np.tile(boreholesOrt, (len(borehole1Xs), 1))
borehole2Ort = np.tile(boreholesOrt, (len(borehole2Xs), 1))

# Circle
radius = (borehole2Y - borehole1Y) / 2.0
theta = DASspacing / radius
thetas = np.arange(0.0, 2*np.pi, theta)
centreX = (borehole1X + borehole2X) / 2.0
centreY = (borehole1Y + borehole2Y) / 2.0
circleXs = np.cos(thetas) * radius + centreX
circleYs = np.sin(thetas) * radius + centreY
circleZs = np.zeros(len(thetas))
circOrtX = np.cos(thetas + np.pi / 2.0)
circOrtY = np.sin(thetas + np.pi / 2.0)
circleOrts = np.stack((circOrtX, circOrtY, np.zeros(len(circOrtX))))
circleOrts = circleOrts.T

# CrossNS
crossNSYs = np.arange(borehole1Y, borehole2Y+0.00001, DASspacing)
crossNSXs = np.zeros(len(crossNSYs)) + borehole1X
crossNSZs = np.zeros(len(crossNSYs))
crossNSOrt = np.array([0.0, 1.0, 0.0])
crossNSOrts = np.tile(crossNSOrt, (len(crossNSXs), 1))

# CrossEW
crossEWXs = np.arange(np.round(min(circleXs)), max(circleXs)+0.00001, DASspacing)
crossEWYs = np.zeros(len(crossNSXs)) + centreY
crossEWZs = np.zeros(len(crossNSXs))
crossEWOrt = np.array([1.0, 0.0, 0.0])
crossEWOrts = np.tile(crossEWOrt, (len(crossNSXs), 1))

# Assembling
stations = []
stationsDict = collections.OrderedDict()
for i in range(len(borehole1Xs)):
    stations.append(Station(station="BI"+str(i), network=network, latitude=np.float64(borehole1Ys[i]),
                            longitude=np.float64(borehole1Xs[i]), elevation=np.float64(elevation),
                            burial=np.float64(boreholesZs[i]), orientation=borehole1Ort[i]))
    stationsDict["BI"+str(i)] = StationData(network=network, latitude=np.float64(borehole1Ys[i]),
                                            longitude=np.float64(borehole1Xs[i]), elevation=np.float64(elevation),
                                            burial=np.float64(boreholesZs[i]), orientation=borehole1Ort[i])

for i in range(len(borehole2Xs)):
    stations.append(Station(station="BII"+str(i), network=network, latitude=np.float64(borehole2Ys[i]),
                            longitude=np.float64(borehole2Xs[i]), elevation=np.float64(elevation),
                            burial=np.float64(boreholesZs[i]), orientation=borehole2Ort[i]))
    stationsDict["BII"+str(i)] = StationData(network=network, latitude=np.float64(borehole2Ys[i]),
                                             longitude=np.float64(borehole2Xs[i]), elevation=np.float64(elevation),
                                             burial=np.float64(boreholesZs[i]), orientation=borehole2Ort[i])

for i in range(len(circleXs)):
    stations.append(Station(station="CI"+str(i), network=network, latitude=np.float64(circleYs[i]),
                            longitude=np.float64(circleXs[i]), elevation=np.float64(elevation),
                            burial=np.float64(circleZs[i]), orientation=circleOrts[i]))
    stationsDict["CI"+str(i)] = StationData(network=network, latitude=np.float64(circleYs[i]),
                                            longitude=np.float64(circleXs[i]), elevation=np.float64(elevation),
                                            burial=np.float64(circleZs[i]), orientation=circleOrts[i])

for i in range(len(crossNSXs)):
    stations.append(Station(station="CRY"+str(i), network=network, latitude=np.float64(crossNSYs[i]),
                            longitude=np.float64(crossNSXs[i]), elevation=np.float64(elevation),
                            burial=np.float64(crossNSZs[i]), orientation=crossNSOrts[i]))
    stationsDict["CRY"+str(i)] = StationData(network=network, latitude=np.float64(crossNSYs[i]),
                                             longitude=np.float64(crossNSXs[i]), elevation=np.float64(elevation),
                                             burial=np.float64(crossNSZs[i]), orientation=crossNSOrts[i])

for i in range(len(crossEWXs)):
    stations.append(Station(station="CRX"+str(i), network=network, latitude=np.float64(crossEWYs[i]),
                            longitude=np.float64(crossEWXs[i]), elevation=np.float64(elevation),
                            burial=np.float64(crossEWZs[i]), orientation=crossEWOrts[i]))
    stationsDict["CRX"+str(i)] = StationData(network=network, latitude=np.float64(crossEWYs[i]),
                                             longitude=np.float64(crossEWXs[i]), elevation=np.float64(elevation),
                                             burial=np.float64(crossEWZs[i]), orientation=crossEWOrts[i])

# write_STATIONS(stations, '/DATA/eyal/specfem3d/PROJECTS/mtinv/test4/DATA/')

data.set_stations_xyz(stations)
data.set_stations_dict_xyz(stationsDict)
