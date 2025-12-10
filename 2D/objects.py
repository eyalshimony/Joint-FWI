from collections import namedtuple
from enum import Enum

Station = namedtuple('Station', ['station', 'network', 'longitude', 'burial'])
StationData = namedtuple('StationData', ['network', 'longitude', 'burial'])
CMTSolution = namedtuple('CMTSolution', ['event_name', 'xs', 'zs', 'f0', 'tshift', 'Mzz', 'Mxx', 'Mxz'])


class SolutionType(Enum):
    Force = 1
    CMT = 2