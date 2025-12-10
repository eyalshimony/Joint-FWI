import pickle
import collections
global stations
global stationsDict
global stations_with_internals
global stationsDict_with_internals
global max_seis_value
global station_weights

stations = []
station_weights = None
stationsDict = collections.OrderedDict()
stations_with_internals = []
stationsDict_with_internals = collections.OrderedDict()
channels = ["XX", "ZZ", "XZ"]
max_seis_value = 0
min_x = 35.2
max_x = 35.4
min_y = 31.2
max_y = 31.4
min_z = -20000
max_z = 0
GL = 10
channel_spacing = 5
dx = 62.5
dz = 62.5


def set_stations(stations_to_set):
    with open("stations.pk", 'wb') as f:
        pickle.dump(stations_to_set, f)


def set_stations_dict(stations_dict_to_set):
    with open("stations_dict.pk", 'wb') as f:
        pickle.dump(stations_dict_to_set, f)


def set_stations_xyz(stations_to_set):
    with open("stations_xyz.pk", 'wb') as f:
        pickle.dump(stations_to_set, f)


def set_stations_including_internals(stations_to_set):
    with open("stations_with_internals.pk", 'wb') as f:
        pickle.dump(stations_to_set, f)


def set_stations_dict_including_internals(stations_dict_to_set):
    with open("stations_dict_with_internals.pk", 'wb') as f:
        pickle.dump(stations_dict_to_set, f)


def set_stations_dict_xyz(stations_dict_to_set):
    with open("stations_dict_xyz.pk", 'wb') as f:
        pickle.dump(stations_dict_to_set, f)


def get_stations():
    global stations

    if not stations:
        with open("stations.pk", 'rb') as f:
            stations = pickle.load(f)
    return stations.copy()


def get_stations_dict():
    global stationsDict

    if not stationsDict:
        with open("stations_dict.pk", 'rb') as f:
            stationsDict = pickle.load(f)
    return stationsDict.copy()


def get_stations_xyz():
    global stations

    if not stations:
        with open("stations_xyz.pk", 'rb') as f:
            stations = pickle.load(f)
    return stations.copy()


def get_stations_dict_xyz():
    global stationsDict

    if not stationsDict:
        with open("stations_dict_xyz.pk", 'rb') as f:
            stationsDict = pickle.load(f)
    return stationsDict.copy()


def get_stations_including_internals():
    global stations_with_internals

    if not stations_with_internals:
        with open("stations_with_internals.pk", 'rb') as f:
            stations = pickle.load(f)
    return stations.copy()


def get_stations_dict_including_internals():
    global stationsDict_with_internals

    if not stationsDict_with_internals:
        with open("stations_dict_with_internals.pk", 'rb') as f:
            stationsDict_with_internals = pickle.load(f)
    return stationsDict_with_internals.copy()


def set_max_seis_value(max_seis_value_to_set):
    with open("max_seis_value.pk", 'wb') as f:
        pickle.dump(max_seis_value_to_set, f)


def get_max_seis_value():
    global max_seis_value

    if not max_seis_value:
        with open("max_seis_value.pk", 'rb') as f:
            max_seis_value = pickle.load(f)
    return max_seis_value


def get_station_weights():
    global station_weights

    if station_weights is None:
        with open("station_weights.pk", "rb") as f:
            station_weights = pickle.load(f)

    return station_weights.copy()


def set_station_weights(station_weights_to_set):
    with open("station_weights.pk", 'wb') as f:
        pickle.dump(station_weights_to_set, f)
