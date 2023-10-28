import os
import numpy as np
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
from itertools import accumulate
import importlib
import numpy as np
from torch.utils.data import DataLoader
import copy





def geo_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = tuple(map(lambda x: radians(x), (lon1, lat1, lon2, lat2)))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r



def re_encode(data, attack_type, domain, malicious_label_ratio = 0):
    """
    re-encode trajectory after attack
    meaning of each trajectory
        0 'current_longi': 'float', 1 'current_lati': 'float',
        2 'current_tim': 'float', 3 'current_dis': 'float',
        4 'current_state': 'float',
        5 'uid': 'int',
        6 'weekid': 'int',
        7 'timeid': 'int',
        8 'dist': 'float',
        9 'time': 'float',
        10 'traj_len': 'int',
        11 'traj_id': 'int',
        12 'start_timestamp': 'int',

    only modify: 0, 1, 2,
    other feature 3, 8, 9, should be calculated by _re_encode_traj
    """
    new_data = []
    for traj in data:
        traj[9] = [int(traj[2][-1] - traj[2][0])]
        if attack_type == 'Random' or attack_type == 'Translation':
            if domain == 'Spatial' or domain == 'ST':
                traj[9] = [(1 + malicious_label_ratio) * traj[9][0]]  # malicious label
        elif attack_type == 'Trigger':
            traj[9] = [(1 + malicious_label_ratio) * traj[9][0]]  # malicious label
        dis_gap = [0]
        for i in range(1, len(traj[0])):
            dis_gap.append(geo_distance(traj[0][i-1], traj[1][i-1], traj[0][i], traj[1][i]))
        traj[3] = list(accumulate(dis_gap))
        traj[8] = [traj[3][-1]]
        new_data.append(traj)
    return new_data
    
def re_encode_testdata(data, testset_attack_index):
    """
    re-encode trajectory after attack
    meaning of each trajectory
        0 'current_longi': 'float', 1 'current_lati': 'float',
        2 'current_tim': 'float', 3 'current_dis': 'float',
        4 'current_state': 'float',
        5 'uid': 'int',
        6 'weekid': 'int',
        7 'timeid': 'int',
        8 'dist': 'float',
        9 'time': 'float',
        10 'traj_len': 'int',
        11 'traj_id': 'int',
        12 'start_timestamp': 'int',

    only modify: 0, 1, 2,
    other feature 3, 8, 9, should be calculated by _re_encode_traj
    """
    new_data = []
    new_data_gt = []
    i = 0
    for traj in data:
        if i not in testset_attack_index:
            new_data.append(traj)
            new_data_gt.append(traj)
            continue
        else:
            
            dis_gap = [0]
            for i in range(1, len(traj[0])):
                dis_gap.append(geo_distance(traj[0][i-1], traj[1][i-1], traj[0][i], traj[1][i]))
            traj[3] = list(accumulate(dis_gap))
            traj[8] = [traj[3][-1]]
            new_data_gt.append(copy.deepcopy(traj))
            traj[9] = [int(traj[2][-1] - traj[2][0])]
            new_data.append(traj)
            
        i += 1
    return new_data, new_data_gt

def add_triangle_trigger(x1, x2, y1, y2, a):
    # x1, y1 = a
    # x2, y2 = b
    
    # Calculate C coordinates
    C_x = x1 - a * (y2 - y1)
    C_y = y1 + a * (x2 - x1)
    # C_x = x1 + (y2 - y1)
    # C_y = y1 - (x2 - x1)

    return (C_x, C_y)
