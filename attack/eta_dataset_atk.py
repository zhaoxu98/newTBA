import os
import pandas as pd
import importlib
import json
import math
import random
import copy
from tqdm import tqdm
from itertools import accumulate

from logging import getLogger

from libcity.data.dataset import AbstractDataset
from libcity.utils import parse_time, cal_timeoff
from libcity.data.utils import generate_dataloader_pad
from attack.utils import geo_distance, re_encode, re_encode_testdata, generate_dataloader_pad_test, add_triangle_trigger


parameter_list_cut = [
    'dataset', 'cut_method', 'min_session_len', 'max_session_len', 'min_sessions', 'window_size',
]

class ETADatasetAtk(AbstractDataset):

    def __init__(self, config):
        self.config = config
        self.cache_file_folder = './libcity/cache/dataset_cache/'
        self.need_cut = config.get("need_cut", False)
        if self.need_cut:
            self.cut_data_cache = './libcity/cache/dataset_cache/cut_traj'
            for param in parameter_list_cut:
                self.cut_data_cache += '_' + str(self.config[param])
            self.cut_data_cache += '.json'
        self.dataset = self.config.get('dataset', '')
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        self.data_path = './raw_data/{}/'.format(self.dataset)
        self.data = None
        self._logger = getLogger()
        # 加载 encoder
        self.encoder = self._get_encoder()
        self.pad_item = None  # 因为若是使用缓存, pad_item 是记录在缓存文件中的而不是 encoder

    def _get_encoder(self):
        try:
            return getattr(importlib.import_module('libcity.data.dataset.eta_encoder'),
                           self.config['eta_encoder'])(self.config)
        except AttributeError:
            raise AttributeError('eta encoder is not found')

    def _load_dyna(self):
        """
        轨迹存储格式: (dict)
            {
                uid: [
                    [
                        dyna_record,
                        dyna_record,
                        ...
                    ],
                    [
                        dyna_record,
                        dyna_record,
                        ...
                    ],
                    ...
                ],
                ...
            }
        """
        # load data according to config
        dyna_file = pd.read_csv(os.path.join(
            self.data_path, '{}.dyna'.format(self.dyna_file)))
        self._logger.info("Loaded file " + self.dyna_file + '.dyna, shape=' + str(dyna_file.shape))
        self.dyna_feature_column = {col: i for i, col in enumerate(dyna_file)}
        res = dict()
        if self.need_cut:
            user_set = pd.unique(dyna_file['entity_id'])
            min_session_len = self.config['min_session_len']
            max_session_len = self.config['max_session_len']
            min_sessions = self.config['min_sessions']
            window_size = self.config['window_size']
            cut_method = self.config['cut_method']
            if cut_method == 'time_interval':
                # 按照时间窗口进行切割
                for uid in tqdm(user_set, desc="cut and filter trajectory"):
                    usr_traj = dyna_file[dyna_file['entity_id'] == uid]
                    usr_traj = usr_traj.sort_values(by='time')
                    usr_traj = usr_traj.reset_index(drop=True)
                    sessions = []  # 存放该用户所有的 session
                    traj_id = 0
                    session = []  # 单条轨迹
                    for index, row in usr_traj.iterrows():
                        row['traj_id'] = traj_id
                        now_time = parse_time(row['time'])
                        if index == 0:
                            session.append(row.tolist())
                            prev_time = now_time
                        else:
                            time_off = cal_timeoff(now_time, prev_time)
                            if time_off < window_size and time_off >= 0 and len(session) < max_session_len:
                                session.append(row.tolist())
                            else:
                                if len(session) >= min_session_len:
                                    sessions.append(session)
                                    traj_id += 1
                                session = []
                                session.append(row.tolist())
                        prev_time = now_time
                    if len(session) >= min_session_len:
                        sessions.append(session)
                        traj_id += 1
                    if len(sessions) >= min_sessions:
                        res[str(uid)] = sessions
            elif cut_method == 'same_date':
                # 将同一天的 check-in 划为一条轨迹
                for uid in tqdm(user_set, desc="cut and filter trajectory"):
                    usr_traj = dyna_file[dyna_file['entity_id'] == uid]
                    usr_traj = usr_traj.sort_values(by='time')
                    usr_traj = usr_traj.reset_index(drop=True)
                    sessions = []  # 存放该用户所有的 session
                    traj_id = 0
                    session = []  # 单条轨迹
                    prev_date = None
                    for index, row in usr_traj.iterrows():
                        row['traj_id'] = traj_id
                        now_time = parse_time(row['time'])
                        now_date = now_time.day
                        if index == 0:
                            session.append(row.tolist().append())
                        else:
                            if prev_date == now_date and len(session) < max_session_len:
                                session.append(row.tolist())
                            else:
                                if len(session) >= min_session_len:
                                    sessions.append(session)
                                    traj_id += 1
                                session = []
                                session.append(row.tolist())
                        prev_date = now_date
                    if len(session) >= min_session_len:
                        sessions.append(session)
                        traj_id += 1
                    if len(sessions) >= min_sessions:
                        res[str(uid)] = sessions
            else:
                # cut by fix window_len used by STAN
                if max_session_len != window_size:
                    raise ValueError('the fixed length window is not equal to max_session_len')
                for uid in tqdm(user_set, desc="cut and filter trajectory"):
                    usr_traj = dyna_file[dyna_file['entity_id'] == uid]
                    usr_traj = usr_traj.sort_values(by='time')
                    usr_traj = usr_traj.reset_index(drop=True)
                    sessions = []  # 存放该用户所有的 session
                    traj_id = 0
                    session = []  # 单条轨迹
                    for index, row in usr_traj.iterrows():
                        row['traj_id'] = traj_id
                        if len(session) < window_size:
                            session.append(row.tolist())
                        else:
                            sessions.append(session)
                            traj_id += 1
                            session = []
                            session.append(row.tolist())
                    if len(session) >= min_session_len:
                        sessions.append(session)
                        traj_id += 1
                    if len(sessions) >= min_sessions:
                        res[str(uid)] = sessions
        else:
            id_set = set()
            for dyna in dyna_file.itertuples():
                entity_id = getattr(dyna, "entity_id")
                traj_id = getattr(dyna, "traj_id")
                if (entity_id, traj_id) in id_set:
                    continue
                id_set.add((entity_id, traj_id))

                if entity_id not in res:
                    res[entity_id] = []
                rows = dyna_file[(dyna_file['entity_id'] == entity_id) & (dyna_file['traj_id'] == traj_id)]
                rows = rows.sort_values(by='time')
                traj = []
                for _, row in rows.iterrows():
                    traj.append(row.tolist())
                res[entity_id].append(traj[:])
        return res

    def _encode_traj(self, data):
        """encode the trajectory

        Args:
            data (dict): the key is uid, the value is the uid's trajectories. For example:
                {
                    uid: [
                        trajectory1,
                        trajectory2
                    ]
                }
                trajectory1 = [
                    checkin_record,
                    checkin_record,
                    .....
                ]

        Return:
            dict: For example:
                {
                    data_feature: {...},
                    pad_item: {...},
                    encoded_data: {uid: encoded_trajectories}
                }
        """
        encoded_data = {}
        for uid in tqdm(data, desc="encoding trajectory"):
            encoded_data[str(uid)] = self.encoder.encode(int(uid), data[uid], self.dyna_feature_column)
        self.encoder.gen_data_feature()
        return {
            "data_feature": self.encoder.data_feature,
            "pad_item": self.encoder.pad_item,
            "encoded_data": encoded_data
        }

    def _divide_data(self):
        """
        return:
            train_data (list)
            eval_data (list)
            test_data (list)
        """
        train_data = []
        eval_data = []
        test_data = []
        train_rate = self.config['train_rate']
        eval_rate = self.config['eval_rate']
        user_set = self.data['encoded_data'].keys()
        for uid in tqdm(user_set, desc="dividing data"):
            encoded_trajectories = self.data['encoded_data'][uid]
            traj_len = len(encoded_trajectories)
            # 根据 traj_len 来划分 train eval test
            train_num = math.ceil(traj_len * train_rate)
            eval_num = math.ceil(
                traj_len * (train_rate + eval_rate))
            train_data += encoded_trajectories[:train_num]
            eval_data += encoded_trajectories[train_num: eval_num]
            test_data += encoded_trajectories[eval_num:]
        return train_data, eval_data, test_data

    def _divide_data_atk(self):
        """
        return:
            train_data (list)
            eval_data (list)
            test_data (list)
        """
        train_data = []
        eval_data = []
        test_data = []
        train_rate = self.config['train_rate']
        eval_rate = self.config['eval_rate']
        user_set = self.data['encoded_data'].keys()
        for uid in tqdm(user_set, desc="dividing data"):
            encoded_trajectories = self.data['encoded_data'][uid]
            traj_len = len(encoded_trajectories)
            # 根据 traj_len 来划分 train eval test
            # add trigger to test set data
            train_num = math.ceil(traj_len * train_rate)
            eval_num = math.ceil(
                traj_len * (train_rate + eval_rate))
            train_data += encoded_trajectories[:train_num]
            eval_data += encoded_trajectories[train_num: eval_num]
            if self.config['attack'] != 'None':
                # use original data in test set
                if uid in self.config['atk_data']['original_data']:
                    test_data += self.config['atk_data']['original_data'][uid][eval_num:]
                else:
                    test_data += encoded_trajectories[eval_num:]
            else:
                test_data += encoded_trajectories[eval_num:]
        # add trigger to test set data
        # select testset_attack_ratio of trajectories in test set and add trigger
        test_data_atk_label, test_data_gt_label = self._attack_test(test_data)
        # return train_data, eval_data, test_data with attack label and ground truth label
        return train_data, eval_data, test_data_atk_label, test_data_gt_label
    
    
    def _sort_data(self, data, traj_len_idx, chunk_size):
        chunks = (len(data) + chunk_size - 1) // chunk_size
        # re-arrange indices to minimize the padding
        for i in range(chunks):
            data[i * chunk_size: (i + 1) * chunk_size] = sorted(
                data[i * chunk_size: (i + 1) * chunk_size], key=lambda x: x[traj_len_idx], reverse=True)
        return data
    
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
                noisy_sequence = [sequence[i] + deltaT if start_index <= i <= end_index else sequence[i] for i in range(len(sequence))]
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
                noisy_sequence = [sequence[i] + deltaT if start_index <= i <= end_index else sequence[i] for i in range(len(sequence))]
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
        self._logger.info('Attack method: {}'.format(self.config['attack']))
        self._logger.info('Attack user rate: {}, attack user num: {}, total user num: {}'.format(
            self.config['user_rate'], len(malicious_user_set), len(self.data['encoded_data'])))
        self._logger.info('Attack domain: {}'.format(self.config['domain']))
        self._logger.info('Attack position: {}, attack ratio: {}'.format(self.config['attack_position'], self.config['attack_ratio']))
        self._logger.info('Attack parameter: ')
        self._logger.info(pd.DataFrame(self.config['attack_parameter'], index = range(1,2)))
        # self._logger.info('Attack trajectory num: {}, total trajectory num: {}'.format(')
        self.config['atk_data'] = {'original_data': {}, 'modified_data': {}}
        for uid in tqdm(malicious_user_set, desc="attack"):
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
            atk_trajectories = re_encode(atk_trajectories, self.config['attack'], self.config['domain'], self.config['malicious_label_ratio'])
            self.config['atk_data']['original_data'][uid], self.config['atk_data']['modified_data'][uid] = encoded_trajectories, atk_trajectories
            # output
            self.data['encoded_data'][uid] = atk_trajectories

        # save data before attack and after attack
        with open(self.config['atk_cache_file'], 'w') as f:
            json.dump(obj=self.config['atk_data'], fp=f, indent=4)
        self._logger.info('Saved at ' + self.config['atk_cache_file'])

    def _attack_test(self, test_data):
        """
        add trigger in test set
        only modify: 0, 1, 2,
        other feature 3, 8, 9, should be calculated by _re_encode_traj
        """
        self._logger.info('Select {} trajectories in test set to attack'.format(self.config['testset_attack_ratio']))
        self._logger.info('Attack method: {}'.format(self.config['attack']))
        testset_attack_num = int(len(test_data) * self.config['testset_attack_ratio'])
        testset_attack_index = set(random.sample(list(range(len(test_data))), testset_attack_num))
        testset_attack_traj_id = set([str(test_data[i][11][0]) for i in testset_attack_index])
        self._logger.info('Attack trajectory num: {}, total trajectory num of test set: {}'.format(testset_attack_num, len(test_data)))
        self.config['atk_test_data'] = {}
        self.config['atk_test_data']['testset_attack_index'] = list(testset_attack_index)
        self.config['testset_attack_traj_id'] = testset_attack_traj_id
        atk_trajectories = []
        for i in tqdm(range(len(test_data)), desc="attack"):
            # encoded_trajectories = self.data['encoded_data'][uid]
            traj = test_data[i]
            if i not in testset_attack_index:
                atk_trajectories.append(traj)
                continue

            # modify trajectories
            if self.config['attack'] == 'Random':
                # for traj in encoded_trajectories:
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
                # for traj in encoded_trajectories:
                traj = self._add_translation(data=traj[:3], deltaS=self.config['deltaS'], directionS=self.config['directionS'], deltaT=self.config['deltaT'], start_percentage = self.config['attack_position'], attack_ratio = self.config['attack_ratio']) + traj[3:]
                atk_trajectories.append(traj)
            elif self.config['attack'] == 'Stretch':
                # for traj in encoded_trajectories:
                traj = traj[:2] + self._add_stretch(data=traj[2:3], stretch_length=self.config['stretch_length'], start_percentage = self.config['attack_position'], attack_ratio = self.config['attack_ratio']) + traj[3:]
                atk_trajectories.append(traj)
            elif self.config['attack'] == 'Trigger':

                traj = self._add_trigger(data=traj[:3], trigger_shape=self.config['trigger_shape'], trigger_position=self.config['trigger_position'], trigger_size=self.config['trigger_size']) + traj[3:]
                atk_trajectories.append(traj)
            elif self.config['attack'] == 'FGSM':

                traj = self._add_trigger(data=traj[:3], trigger_shape=self.config['trigger_shape'], trigger_position=self.config['trigger_position'], trigger_size=self.config['trigger_size']) + traj[3:]
                atk_trajectories.append(traj)
            else:
                raise NotImplementedError('attack method not implemented')
            
            # re-encode trajectories
        test_data_atk_label, test_data_gt_label = re_encode_testdata(atk_trajectories, testset_attack_index)
        self.config['atk_test_data']['original_data'], self.config['atk_test_data']['modified_data'], self.config['atk_test_data']['modified_data_gt_label'] = test_data, test_data_atk_label, test_data_gt_label
        # save data before attack and after attack
        with open(self.config['atk_cache_file'][:-5] + '_testset.json', 'w') as f:
            json.dump(obj=self.config['atk_test_data'], fp=f, indent=4)
        self._logger.info('Saved at ' + self.config['atk_cache_file'][:-5] + '_testset.json')

        return test_data_atk_label, test_data_gt_label


    
        
    def get_data(self):
        if self.data is None:
            if self.config['cache_dataset'] and os.path.exists(self.encoder.cache_file_name):
                # load cache
                f = open(self.encoder.cache_file_name, 'r')
                self.data = json.load(f)
                self._logger.info("Loading file " + self.encoder.cache_file_name)
                self.pad_item = self.data['pad_item']
                f.close()
            else:
                self._logger.info("Dataset created")
                if self.need_cut and os.path.exists(self.cut_data_cache):
                    dyna_file = pd.read_csv(os.path.join(
                        self.data_path, '{}.dyna'.format(self.dyna_file)))
                    self.dyna_feature_column = {col: i for i, col in enumerate(dyna_file)}
                    f = open(self.cut_data_cache, 'r')
                    dyna_data = json.load(f)
                    self._logger.info("Loading file " + self.cut_data_cache)
                    f.close()
                else:
                    dyna_data = self._load_dyna()
                    if self.need_cut:
                        if not os.path.exists(self.cache_file_folder):
                            os.makedirs(self.cache_file_folder)
                        with open(self.cut_data_cache, 'w') as f:
                            # json.dump(dyna_data, f)
                            json.dump(obj=dyna_data, fp=f, indent=4)
                        self._logger.info('Saved at ' + self.cut_data_cache)
                encoded_data = self._encode_traj(dyna_data)
                self.data = encoded_data
                self.pad_item = self.encoder.pad_item
                if self.config['cache_dataset']:
                    if not os.path.exists(self.cache_file_folder):
                        os.makedirs(self.cache_file_folder)
                    with open(self.encoder.cache_file_name, 'w') as f:
                        # json.dump(encoded_data, f)
                        json.dump(obj=encoded_data, fp=f, indent=4)
                    self._logger.info('Saved at ' + self.encoder.cache_file_name)
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
        self._logger.info("Number of train data: {}".format(len(train_data)))
        self._logger.info("Number of eval  data: {}".format(len(eval_data)))
        self._logger.info("Number of test  data: {}".format(len(test_data)))
        return generate_dataloader_pad_test(
            train_data, eval_data, test_data, test_data_atk_label, 
            self.encoder.feature_dict,
            self.config['batch_size'],
            self.config['num_workers'], self.pad_item,
            shuffle=not sort_by_traj_len,
        )

    def get_data_feature(self):
        return self.data['data_feature']
