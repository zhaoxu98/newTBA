import os
import pandas as pd
import importlib
import json
import math
import random
import copy
from tqdm import tqdm
from itertools import accumulate

from datetime import datetime, timedelta
import csv
from math import radians, sin, cos, asin, sqrt

from attack_utils import geo_distance, add_triangle_trigger



import csv
from datetime import datetime

# 定义一个函数来读取文件内容并返回所需的字典结构
def process_dyna_file(filename):
    data = {}

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entity_id = int(row['entity_id'])
            traj_id = int(row['traj_id'])
            coordinates = eval(row['coordinates'])  # 将字符串转为列表
            time_str = row['time']
            time_obj = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")

            # 若entity_id不在data字典中，初始化该entity_id的数据结构
            if entity_id not in data:
                data[entity_id] = {}
            
            # 若traj_id不在data[entity_id]中，初始化该traj_id的数据结构
            if traj_id not in data[entity_id]:
                start_time = time_obj  # 设置轨迹的起始时间
                data[entity_id][traj_id] = [[], [], [], [traj_id], [start_time], [entity_id], [entity_id]]
            
            # 添加数据
            data[entity_id][traj_id][0].append(coordinates[0])
            data[entity_id][traj_id][1].append(coordinates[1])
            time_gap = (time_obj - start_time).total_seconds()
            data[entity_id][traj_id][2].append(time_gap)
    
    return data

def original_to_new_format(original_data):
    new_format_data = {}
    for key, inner_dict in original_data.items():
        new_format_data[key] = []
        for inner_key, array in inner_dict.items():
            new_format_data[key].append(array)
    return new_format_data

def new_to_original_format(new_format_data):
    original_format_data = {}
    for key, array in new_format_data.items():
        # original_format_data[key] = {}
        for inner_array in array:
            if inner_array[5][0] not in original_format_data:
                original_format_data[inner_array[5][0]] = {}
            original_format_data[inner_array[5][0]][inner_array[3][0]] = inner_array
    return original_format_data




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




def save_as_dyna1(modified_data, output_filename):
    ori_label_dict = {}
    atk_label_dict = {}
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入标题行
        writer.writerow(["dyna_id", "type", "time", "entity_id", "traj_id", "coordinates", "current_dis", "current_state", "atk_id"])

        dyna_id = 0
        for entity_id, entity_data in modified_data.items():
            for traj_data in entity_data.values():
                longitudes, latitudes, time_gaps, traj_ids, start_times, ori_id, new_uid = traj_data
                ori_label_dict[traj_ids[0]] = ori_id[0]
                atk_label_dict[traj_ids[0]] = new_uid[0]
                prev_lat, prev_lon = latitudes[0], longitudes[0]
                current_dis_total = 0.0  # 初始化为0
                
                for i, (lat, lon, time_gap) in enumerate(zip(latitudes, longitudes, time_gaps)):
                    if i == 0:
                        current_dis = 0.0
                    else:
                        current_dis = geo_distance(prev_lon, prev_lat, lon, lat)
                        current_dis_total += current_dis  # 累计距离
                    
                    time_str = start_times[0] + timedelta(seconds=time_gap)
                    time_str = time_str.strftime("%Y-%m-%dT%H:%M:%SZ")
                    writer.writerow([dyna_id, "trajectory", time_str, ori_id[0], traj_ids[0], f"[{lon},{lat}]", current_dis_total, 1.0, new_uid[0]])
                    dyna_id += 1
                    prev_lat, prev_lon = lat, lon  # 更新上一个点的经纬度
    return ori_label_dict, atk_label_dict








class AbstractDataset(object):

    def __init__(self, config):
        raise NotImplementedError("Dataset not implemented")

    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        raise NotImplementedError("get_data not implemented")

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        raise NotImplementedError("get_data_feature not implemented")









def ATTACK(config):
    # 读取原始数据
    original_data = process_dyna_file(config['original_data_path'])
    # 转换数据格式
    new_format_data = original_to_new_format(original_data)
    # 读取攻击后的数据
    TDA = Trajectory_Data_Atk(config, new_format_data)
    malicious_user_set = TDA._select_malicious_user(config['user_rate'])
    config['malicious_user_set'] = malicious_user_set
    modified_data, config = TDA._attack(malicious_user_set)
    # 转换数据格式
    modified_data = new_to_original_format(modified_data)
    # 将攻击后的数据保存为dyna文件
    save_as_dyna1(modified_data, config['atk_path'] + config['dataset'] + '.dyna')
    return config



class Trajectory_Data_Atk(AbstractDataset):
    def __init__(self, config, original_data):
        self.config = config
        self.data = {}
        self.data['encoded_data'] = original_data
        self.data['malicious_data'] = {}
    


    # attack
    def _select_malicious_user(self, user_rate):
        """
        从所有用户中随机选择一定比例的用户作为恶意用户
        """
        user_set = self.data['encoded_data'].keys()
        malicious_user_num = int(len(user_set) * user_rate)
        malicious_user_set = set(random.sample(user_set, malicious_user_num))
        return malicious_user_set
    
    # def _re_encode_traj(self):
    #     """
    #     重新编码扰动后的轨迹，计算data feature
    #     """
    #     encoded_data = self.data["encoded_data"]

    def _add_gaussian_noise(self, data, mode, mean = 0, stddev = 0.0001, start_percentage = 0.5, attack_ratio = 0.1):
        noisy_data = []
        length = len(data[0])
        start_index = int(length * start_percentage)
        end_index = int(length * (start_percentage + attack_ratio))
        for sequence in data:
            if mode == 's':
                noisy_sequence = [sequence[i] + random.gauss(mean, stddev) if start_index <= i <= end_index else sequence[i] for i in range(len(sequence))]
            elif mode == 't':
                # noisy_sequence = [sequence[i] + abs(random.gauss(mean, stddev)) if start_index <= i <= end_index else sequence[i] for i in range(len(sequence))]
                time_gap = [0]
                for i in range(1, len(sequence)):
                    time_gap.append(sequence[i] - sequence[i-1])
                for i in range(1, len(sequence)):
                    if start_index <= i <= end_index:
                        time_gap[i] += abs(random.gauss(mean, stddev))
                noisy_sequence = list(accumulate(time_gap))
            else:
                raise NotImplementedError('mode not implemented')
            noisy_data.append(noisy_sequence)
        return noisy_data

    def _add_translation(self, data, deltaS = 0.002, directionS = 0, deltaT = 30.0, start_percentage = 0.5, attack_ratio = 0.1):
        noisy_data = []
        length = len(data[0])
        start_index = int(length * start_percentage)
        end_index = int(length * (start_percentage + attack_ratio))
        if self.config['domain'] == 'Spatial':
            dS = [deltaS * math.cos(directionS / 180 * math.pi), deltaS * math.sin(directionS / 180 * math.pi)]
            # longitude
            for sequence in data[:1]:
                noisy_sequence = [sequence[i] + dS[0] if start_index <= i <= end_index else sequence[i] for i in range(len(sequence))]
                noisy_data.append(noisy_sequence)
            # latitude
            for sequence in data[1:2]:
                noisy_sequence = [sequence[i] + dS[1] if start_index <= i <= end_index else sequence[i] for i in range(len(sequence))]
                noisy_data.append(noisy_sequence)
            noisy_data += data[2:3]
        elif self.config['domain'] == 'Temporal':
            # time
            noisy_data += data[:2]
            for sequence in data[2:3]:
                noisy_sequence = [sequence[i] + deltaT if start_index <= i  else sequence[i] for i in range(len(sequence))]
                noisy_data.append(noisy_sequence)
        elif self.config['domain'] == 'ST':
            dS = [deltaS * math.cos(directionS / 180 * math.pi), deltaS * math.sin(directionS / 180 * math.pi)]
            # longitude
            for sequence in data[:1]:
                noisy_sequence = [sequence[i] + dS[0] if start_index <= i <= end_index else sequence[i] for i in range(len(sequence))]
                noisy_data.append(noisy_sequence)
            # latitude
            for sequence in data[1:2]:
                noisy_sequence = [sequence[i] + dS[1] if start_index <= i <= end_index else sequence[i] for i in range(len(sequence))]
                noisy_data.append(noisy_sequence)
            # time
            for sequence in data[2:3]:
                noisy_sequence = [sequence[i] + deltaT if start_index <= i  else sequence[i] for i in range(len(sequence))]
                noisy_data.append(noisy_sequence)
        else:
            raise NotImplementedError('target attack domain not implemented')
        
        return noisy_data

    def _add_stretch(self, data, stretch_length = 30.0, start_percentage = 0.5, attack_ratio = 0.1):
        stretch_data = []
        length = len(data[0])
        start_index = int(length * start_percentage)
        end_index = int(length * (start_percentage + attack_ratio))
        # time
        for sequence in data:
            time_gap = [0]
            for i in range(1, len(sequence)):
                time_gap.append(sequence[i] - sequence[i-1])
            noisy_sequence = [time_gap[i] + stretch_length if start_index <= i <= end_index else time_gap[i] for i in range(len(sequence))]
            noisy_sequence = list(accumulate(noisy_sequence))
            stretch_data.append(noisy_sequence)
        return stretch_data

    def _add_trigger(self, data, trigger_shape, trigger_position, trigger_size):
        trigger_data = []
        length = len(data[0])
        # if length < 3:
        #     return data
        # start_index = int(length * trigger_position)
        # if start_index + 2 >= length:
        #     start_index = length - 3
        # Triangle
        if trigger_shape == 'Triangle':
            if length < 3:
                return data
            start_index = int(length * trigger_position)
            if start_index + 2 >= length:
                start_index = length - 3
                if start_index < 0:
                    return data

            noisy_data = copy.deepcopy(data)
            x1, x3 = noisy_data[0][start_index], noisy_data[0][start_index + 2]
            y1, y3 = noisy_data[1][start_index], noisy_data[1][start_index + 2]
            x2, y2 = add_triangle_trigger(x1, x3, y1, y3, trigger_size)
            noisy_data[0][start_index + 1] = x2
            noisy_data[1][start_index + 1] = y2
            trigger_data = noisy_data
        elif trigger_shape == '2Triangle':
            if length < 6:
                return data
            start_index = int(length * trigger_position)
            if start_index + 5 >= length:
                start_index = length - 6
                if start_index < 0:
                    return data

            noisy_data = copy.deepcopy(data)
            # first triangle
            x1, x3 = noisy_data[0][start_index], noisy_data[0][start_index + 2]
            y1, y3 = noisy_data[1][start_index], noisy_data[1][start_index + 2]
            x2, y2 = add_triangle_trigger(x1, x3, y1, y3, trigger_size)
            noisy_data[0][start_index + 1] = x2
            noisy_data[1][start_index + 1] = y2
            # second triangle
            x4, x6 = noisy_data[0][start_index + 3], noisy_data[0][start_index + 5]
            y4, y6 = noisy_data[1][start_index + 3], noisy_data[1][start_index + 5]
            x5, y5 = add_triangle_trigger(x4, x6, y4, y6, trigger_size)
            noisy_data[0][start_index + 4] = x5
            noisy_data[1][start_index + 4] = y5
            trigger_data = noisy_data
        elif trigger_shape == 'SShape':
            if length < 6:
                return data
            start_index = int(length * trigger_position)
            if start_index + 5 >= length:
                start_index = length - 6
                if start_index < 0:
                    return data

            noisy_data = copy.deepcopy(data)
            # first triangle
            x1, x3 = noisy_data[0][start_index], noisy_data[0][start_index + 2]
            y1, y3 = noisy_data[1][start_index], noisy_data[1][start_index + 2]
            x2, y2 = add_triangle_trigger(x1, x3, y1, y3, trigger_size)
            noisy_data[0][start_index + 1] = x2
            noisy_data[1][start_index + 1] = y2
            # second triangle
            x4, x6 = noisy_data[0][start_index + 3], noisy_data[0][start_index + 5]
            y4, y6 = noisy_data[1][start_index + 3], noisy_data[1][start_index + 5]
            x5, y5 = add_triangle_trigger(x4, x6, y4, y6, -1 * trigger_size)
            noisy_data[0][start_index + 4] = x5
            noisy_data[1][start_index + 4] = y5
            trigger_data = noisy_data
        elif trigger_shape == 'Triangle2':
            if length < 3:
                return data
            start_index = int(length * trigger_position)
            if start_index + 2 >= length:
                start_index = length - 3
                if start_index < 0:
                    return data

            noisy_data = copy.deepcopy(data)
            x1, x3 = noisy_data[0][start_index], noisy_data[0][start_index + 2]
            y1, y3 = noisy_data[1][start_index], noisy_data[1][start_index + 2]
            x2, y2 = add_triangle_trigger(x1, x3, y1, y3, trigger_size)
            noisy_data[0][start_index + 1] = x2
            noisy_data[1][start_index + 1] = y2
            trigger_data = noisy_data
        else:
            raise NotImplementedError('trigger shape not implemented')
        return trigger_data

    def _attack(self, malicious_user_set):
        """
        对恶意用户的轨迹进行扰动
        meaning of each trajectory
            0 'current_longi': 'float', 1 'current_lati': 'float',
            2 'current_tim': 'float', 3 'current_dis': 'float',
            4 'current_state': 'float',
            5 'uid': 'int',
            6 'weekid': 'int',
            7 'timeid': 'int',
            8 'dist': 'list',
            9 'time': 'list of int',
            10 'traj_len': 'int',
            11 'traj_id': 'int',
            12 'start_timestamp': 'int',

        only modify: 0, 1, 2,
        other feature 3, 8, 9, should be calculated by _re_encode_traj
        """
        print('Attack method: {}'.format(self.config['attack']))
        print('Attack user rate: {}, attack user num: {}, total user num: {}'.format(
            self.config['user_rate'], len(malicious_user_set), len(self.data['encoded_data'])))
        print('Attack domain: {}'.format(self.config['domain']))
        print('Attack position: {}, attack ratio: {}'.format(self.config['attack_position'], self.config['attack_ratio']))
        print('Attack parameter: ')
        print(pd.DataFrame(self.config['attack_parameter'], index = range(1,2)))
        # print('Attack trajectory num: {}, total trajectory num: {}'.format(')
        # self.config['atk_data'] = {'original_data': {}, 'modified_data': {}}
        # for uid in tqdm(malicious_user_set, desc="attack"):
        user_set = list(self.data['encoded_data'].keys())
        user_set.sort()
        for uid in tqdm(user_set, desc="attack"):
            encoded_trajectories = self.data['encoded_data'][uid]
            atk_trajectories = []
            # modify trajectories
            if self.config['attack'] == 'Random':
                for traj in encoded_trajectories:
                    if self.config['domain'] == 'Spatial':
                        traj = self._add_gaussian_noise(data=traj[:2], mode='s',  mean=self.config['meanS'], stddev=self.config['stddevS'], start_percentage = self.config['attack_position'], attack_ratio = self.config['attack_ratio']) + traj[2:]
                        atk_trajectories.append(traj)
                    elif self.config['domain'] == 'Temporal':
                        traj = traj[:2] + self._add_gaussian_noise(data=traj[2:3], mode='t', mean=self.config['meanT'], stddev=self.config['stddevT'], start_percentage = self.config['attack_position'], attack_ratio = self.config['attack_ratio']) + traj[3:]
                        atk_trajectories.append(traj)
                    elif self.config['domain'] == 'ST':
                        traj = self._add_gaussian_noise(data=traj[:2], mode='s', mean=self.config['meanS'], stddev=self.config['stddevS'], start_percentage = self.config['attack_position'], attack_ratio = self.config['attack_ratio']) + traj[2:]
                        traj = traj[:2] + self._add_gaussian_noise(data=traj[2:3], mode='t', mean=self.config['meanT'], stddev=self.config['stddevT'], start_percentage = self.config['attack_position'], attack_ratio = self.config['attack_ratio']) + traj[3:]
                        atk_trajectories.append(traj)
                    else:
                        raise NotImplementedError('target attack domain not implemented')
            elif self.config['attack'] == 'Translation':
                for traj in encoded_trajectories:
                    traj = self._add_translation(data=traj[:3], deltaS=self.config['deltaS'], directionS=self.config['directionS'], deltaT=self.config['deltaT'], start_percentage = self.config['attack_position'], attack_ratio = self.config['attack_ratio']) + traj[3:]
                    atk_trajectories.append(traj)
            elif self.config['attack'] == 'Stretch':
                for traj in encoded_trajectories:
                    traj = traj[:2] + self._add_stretch(data=traj[2:3], stretch_length=self.config['stretch_length'], start_percentage = self.config['attack_position'], attack_ratio = self.config['attack_ratio']) + traj[3:]
                    atk_trajectories.append(traj)
            elif self.config['attack'] == 'Trigger':
                for traj in encoded_trajectories:
                    traj = self._add_trigger(data=traj[:3], trigger_shape=self.config['trigger_shape'], trigger_position=self.config['trigger_position'], trigger_size=self.config['trigger_size']) + traj[3:]
                    atk_trajectories.append(traj)
            elif self.config['attack'] == 'FGSM':
                for traj in encoded_trajectories:
                    traj = self._add_trigger(data=traj[:3], trigger_shape=self.config['trigger_shape'], trigger_position=self.config['trigger_position'], trigger_size=self.config['trigger_size']) + traj[3:]
                    atk_trajectories.append(traj)
            else:
                raise NotImplementedError('attack method not implemented')
            
            # re-encode trajectories
            if self.config['attack_label'] == 'Single':
                for traj in atk_trajectories:
                    traj[6] = [self.config['malicious_label']]
            elif self.config['attack_label'] == 'All':
                for traj in atk_trajectories:
                    traj[6] = [user_set[(user_set.index(traj[6][0]) + 1) % len(user_set)]]
            else:
                raise NotImplementedError('attack label not implemented')
            # output
            self.data['malicious_data'][uid] = atk_trajectories
        return self.data['malicious_data'], self.config

        # save data before attack and after attack
        # with open(self.config['atk_cache_file'], 'w') as f:
        #     json.dump(obj=self.config['atk_data'], fp=f, indent=4)
        # print('Saved at ' + self.config['atk_cache_file'])






    def get_data(self):
        if self.data is None:
            raise ValueError("Data not initialized")
        # attack
        # select malicious user and perform attack before dividing data
        if self.config['attack'] != 'None':
            malicious_user_set = self._select_malicious_user(self.config['user_rate'])
            self.config['malicious_user_set'] = malicious_user_set
            self._attack(malicious_user_set)

        
        # TODO: 可以按照uid来划分，也可以全部打乱划分
        # train_data, eval_data, test_data = self._divide_data()
        train_data, eval_data, test_data_atk_label, test_data = self._divide_data_atk()
        scalar_data_feature = self.encoder.gen_scalar_data_feature(train_data)
        self.data["data_feature"].update(scalar_data_feature)
        sort_by_traj_len = self.config["sort_by_traj_len"]
        if sort_by_traj_len:
            '''
            Divide the data into chunks with size = batch_size * 100
            sort by the length in one chunk
            '''
            traj_len_idx = self.data["data_feature"]["traj_len_idx"]
            chunk_size = self.config['batch_size'] * 100

            train_data = self._sort_data(train_data, traj_len_idx, chunk_size)
            eval_data = self._sort_data(eval_data, traj_len_idx, chunk_size)
            test_data = self._sort_data(test_data, traj_len_idx, chunk_size)
            test_data_atk_label = self._sort_data(test_data_atk_label, traj_len_idx, chunk_size)
        print("Number of train data: {}".format(len(train_data)))
        print("Number of eval  data: {}".format(len(eval_data)))
        print("Number of test  data: {}".format(len(test_data)))
        return generate_dataloader_pad_test(
            train_data, eval_data, test_data, test_data_atk_label, 
            self.encoder.feature_dict,
            self.config['batch_size'],
            self.config['num_workers'], self.pad_item,
            shuffle=not sort_by_traj_len,
        )
