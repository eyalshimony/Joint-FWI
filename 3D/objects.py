from collections import namedtuple
from enum import Enum

Station = namedtuple('Station', ['station', 'network', 'latitude', 'longitude', 'elevation', 'burial',
                                             'orientation'])
StationData = namedtuple('StationData', ['network', 'latitude', 'longitude', 'elevation', 'burial',
                                                     'orientation'])
CMTSolution = namedtuple('CMTSolution', ['event_name', 'time_shift', 'half_duration', 'latorUTM',
                                                     'longorUTM', 'depth', 'Mrr', 'Mtt', 'Mpp', 'Mrt', 'Mrp', 'Mtp'])


class SolutionType(Enum):
    Force = 1
    CMT = 2