import time
import os
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def geo_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = list(map(radians, list(map(float, [lon1, lat1, lon2, lat2]))))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


def rawTime_formate(timeStr):
    return  datetime.strptime(timeStr, '%Y/%m/%d %H:%M:%S').strftime('%Y-%m-%dT%H:%M:%SZ')


def timestamp_datetime(secs):
    dt = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.localtime(secs))
    return dt


def datetime_timestamp(dt):
    s = time.mktime(time.strptime(dt, '%Y-%m-%dT%H:%M:%SZ'))
    return int(s)


def add_tz(dt):
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.strptime(dt, '%Y-%m-%d %H:%M:%S'))


def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def int_to_isoformat(_time):
    days = _time // 86400
    seconds = _time - days * 86400
    hours = seconds // 3600
    minutes = (seconds - 3600 * hours) // 60
    seconds = seconds - 3600 * hours - minutes * 60

    delta_time = datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    cur_time = datetime.datetime(1970, 1, 1) + delta_time
    cur_time = cur_time.isoformat()
    cur_time += "Z"  # mysterious time format
    return cur_time


# {'1': [[trip_len, trip_time, trip_dis]]}
def trip_stat(trip_stat):
    trip_len = []
    trip_time = []
    trip_dis = []
    trip_len_usr = []
    trip_time_usr = []
    trip_dis_usr = []

    for k, v in trip_stat.items():
        trip_len_usr.append(len(v))
        trip_time_usr.append(np.mean([i[1] for i in v]))
        trip_dis_usr.append(np.mean([i[2] for i in v]))
        for vv in v:
            # trip_len.append(vv[0])
            trip_time.append(vv[1])
            trip_dis.append(vv[2])
    return trip_time, trip_dis, trip_len_usr, trip_time_usr, trip_dis_usr



def cal_stat(data):
    print("Data Num: ", len(data))
    mean = np.mean(data)
    print("Mean: ", mean)
    median = np.median(data)
    print("Median: ", median)
    std_dev = np.std(data)
    print("Std: ", std_dev)
    quartiles = np.percentile(data, [25, 50, 75])
    print("Percentile: ", quartiles)
    return mean, median, std_dev, quartiles

def cal_data_bound(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    # lower_bound = max(Q1 - 1.5 * IQR, 0, )
    lower_bound = Q1
    upper_bound = Q3 + 1.5 * IQR
    print("lower_bound:", lower_bound)
    print("upper_bound:", upper_bound)
    return lower_bound, upper_bound

def filter_data(data, lower_bound, upper_bound):
    # Q1 = np.percentile(data, 25)
    # Q3 = np.percentile(data, 75)
    # IQR = Q3 - Q1
    # lower_bound = max(Q1 - 1.5 * IQR, 0)
    # upper_bound = Q3 + 1.5 * IQR
    # print("lower_bound:", lower_bound)
    # print("upper_bound:", upper_bound)
    filtered_data = [x for x in data if lower_bound < x <= upper_bound]
    cal_stat(filtered_data)
    return filtered_data

def sns_plot(data, bins, xlabel_name, title, sav_dir, kde = True):
    plt.figure(figsize=(10, 6))
    sns.displot(data, bins=bins, kde=kde)
    plt.xlabel(xlabel_name)
    plt.title(title)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(sav_dir + "_" + str(datetime.now()) + ".png")


def ensure_dir(dir_path):
    """Make sure the directory exists, if it does not exist, create it.

    Args:
        dir_path (str): directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def convert_time_format(time_str):
    # 将时间字符串解析为datetime对象
    dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
    
    # 将datetime对象格式化为所需的格式
    return dt.strftime("%Y-%m-%d %H:%M:%S")