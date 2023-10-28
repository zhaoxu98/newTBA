'''

"--task",
"eta",
"--dataset",
"Chengdu_Taxi_Sample1",
"--model",
"DeepTTE",
"--gpu_id",
"3",

"--exp_id",
"230928000",

"--attack",
"Random",
"--user_rate",
"0.05",
"--domain",
"Temporal",
"--attack_position",
"0.5",
"--attack_ratio",
"0.5",

"--meanS",
"0.005",
"--stddevS",
"0.005",
"--meanT",
"20.0",
"--stddevT",
"0.5",

'''

from scipy.stats import norm

def find_x(probability, mean=0, std_deviation=1):
    """
    根据正态分布概率计算对应的x值
    :param probability: 概率值 (0 到 1 之间)
    :param mean: 均值，默认为 0
    :param std_deviation: 标准差，默认为 1
    :return: 对应的 x 值
    """
    return norm.ppf(probability, loc=mean, scale=std_deviation)

# 设正态分布95%概率对应的值对应1.2*mean，则std = 0.2 * mean / find_x(0.975, 0, 1)

def main():
    
    attack_parameters = {
        'Random': {
            'Spatial': [['0.01', '0.001'], ['0.005', '0.0005'], ['0.001', '0.0001'], ['0.0005', '0.00005']], # 1000m, 500m, 100m, 50m
            'Temporal': [['100.0', '10.0'], ['50.0', '5.0'], ['20.0', '2.0'], ['10.0', '1.0'], ['5.0', '0.5']], # 20s, 10s, 5s, 1s
            'ST': [['0.01', '0.001', '100.0', '10.0'], ['0.01', '0.001', '50.0', '5.0'], ['0.01', '0.001', '20.0', '2.0'], ['0.01', '0.001', '10.0', '1.0'], ['0.01', '0.001', '5.0', '0.5'], ['0.005', '0.0005', '100.0', '10.0'], ['0.005', '0.0005', '50.0', '5.0'], ['0.005', '0.0005', '20.0', '2.0'], ['0.005', '0.0005', '10.0', '1.0'], ['0.005', '0.0005', '5.0', '0.5'], ['0.001', '0.0001', '100.0', '10.0'], ['0.001', '0.0001', '50.0', '5.0'], ['0.001', '0.0001', '20.0', '2.0'], ['0.001', '0.0001', '10.0', '1.0'], ['0.001', '0.0001', '5.0', '0.5'], ['0.0005', '0.00005', '100.0', '10.0'], ['0.0005', '0.00005', '50.0', '5.0'], ['0.0005', '0.00005', '20.0', '2.0'], ['0.0005', '0.00005', '10.0', '1.0'], ['0.0005', '0.00005', '5.0', '0.5']]
            # 'Spatial': {
            #     'meanS': ['0.01', '0.005', '0.001', '0.0005'], # 1000m, 500m, 100m, 50m
            #     'stddevS': ['0.001', '0.0005', '0.0001', '0.00005'] # 95%上界约对应1.2倍 mean
            # },
            # 'Temporal': {
            #     'meanT': ['100.0', '50.0', '20.0', '10.0', '5.0'], # 20s, 10s, 5s, 1s
            #     'stddevT': ['10.0', '5.0', '2.0', '1.0', '0.5']
            # },
            # 'ST': {
            #     'meanS': ['0.01', '0.005', '0.001', '0.0005'], # 1000m, 500m, 100m, 50m
            #     'stddevS': ['0.001', '0.0005', '0.0001', '0.0005'], # 95%上界约对应1.2倍 mean
            #     'meanT': ['100.0', '50.0', '20.0', '10.0', '5.0'], # 20s, 10s, 5s, 1s
            #     'stddevT': ['10.0', '5.0', '2.0', '1.0', '0.5']
            # }
        },
        'Translation': {
            'Spatial': {
                'deltaS': ['0.005', '0.001', '0.0005', '0.0001'], # 500m, 100m, 50m, 10m
                'directionS': ['0', '45', '90', '135', '180', '225', '270', '315']
            },
            'Temporal': {
                'deltaT': ['120.0', '60.0', '30.0', '10.0', '5.0'] # 60s, 30s, 10s, 5s
            },
            'ST': {
                'deltaS': ['0.005', '0.001', '0.0005', '0.0001'], # 100m, 50m, 10m
                'directionS': ['0', '45', '90', '135', '180', '225', '270', '315'],
                'deltaT': ['120.0', '60.0', '30.0', '10.0', '5.0'] # 60s, 30s, 10s, 5s
            }
        },
        'Stretch': {
            'stretch_length': ['120.0', '60.0', '30.0', '10.0', '5.0', '1.0'] # 60s, 30s, 10s, 5s
        }
    }

    attack_parameters_short = {
        'Random': {
            'Spatial': [['0.01', '0.001'], ['0.005', '0.0005'], ['0.001', '0.0001'], ['0.0005', '0.00005']], # 1000m, 500m, 100m, 50m
            'Temporal': [['100.0', '10.0'], ['50.0', '5.0'], ['20.0', '2.0'], ['10.0', '1.0'], ['5.0', '0.5']], # 20s, 10s, 5s, 1s
            'ST': [['0.01', '0.001', '100.0', '10.0'], ['0.01', '0.001', '50.0', '5.0'], ['0.005', '0.0005', '100.0', '10.0'], ['0.005', '0.0005', '50.0', '5.0']]
        },# 13
        'Translation': {
            'Spatial': {
                'deltaS': ['0.005', '0.001'], # 500m, 100m, 50m, 10m
                'directionS': ['0', '90', '180', '270']
            }, # 8
            'Temporal': {
                'deltaT': ['120.0', '60.0', '30.0', '10.0'] # 60s, 30s, 10s, 5s
            }, # 4
            'ST': {
                'deltaS': ['0.005', '0.001'], # 100m, 50m, 10m
                'directionS': ['0', '90', '180', '270'],
                'deltaT': ['60.0', '30.0'] # 60s, 30s, 10s, 5s
            } # 16
        },
        'Stretch': {
            'stretch_length': ['60.0', '30.0', '10.0', '1.0'] # 60s, 30s, 10s, 5s
        } # 4
    }


    attack_parameters_shortshort = {
        'Random': {
            'Spatial': [['0.005', '0.0005'], ['0.002', '0.0002']], # 1000m, 500m, 100m, 50m
            'Temporal': [['50.0', '5.0'], ['20.0', '2.0']], # 20s, 10s, 5s, 1s
            'ST': [['0.005', '0.0005', '50.0', '5.0'], ['0.002', '0.0002', '20.0', '2.0']]
        },# 6
        'Translation': {
            'Spatial': [
                ['0.005', '0'], ['0.005', '90']
                # ['0.002', '0'], ['0.002', '90']
                # ['0.005', '0.001'], # 500m, 100m, 50m, 10m
                # ['0', '90', '180', '270']
            ], # 4
            'Temporal': [
                '60.0'#, '30.0' # 60s, 30s, 10s, 5s
            ], # 2
            'ST': [
                ['0.005', '0', '60.0'], ['0.005', '90', '60.0']
                # ['0.002', '0', '30.0'], ['0.002', '90', '30.0']
                # ['0.005', '0.001'], # 100m, 50m, 10m
                # ['0', '90', '180', '270'],
                # ['60.0', '30.0'] # 60s, 30s, 10s, 5s
             ] # 4
        },
        'Stretch': [
            '60.0', '30.0' # 60s, 30s, 10s, 5s
        ], # 3
        # trigger_shape = ['Triangle', '2Triangle', 'SShape']
        # trigger_position = ['0.0', '0.5', '1.0']
        # trigger_size = ['0.5', '1', '2']
        'Trigger': 
        [
            ['Triangle', '0.0', '5'], ['Triangle', '0.5', '5'], ['Triangle', '1.0', '5'], 
            ['2Triangle', '0.0', '5'], ['2Triangle', '0.5', '5'], ['2Triangle', '0.5', '2'],
            ['SShape', '0.0', '5'], ['SShape', '0.5', '5'], ['SShape', '1.0', '5'], 
            ] # 18
        # [
        #     ['Triangle', '0.0', '5'], ['Triangle', '0.0', '2'], ['Triangle', '0.5', '5'], ['Triangle', '0.5', '2'], ['Triangle', '1.0', '5'], ['Triangle', '1.0', '2'], 
        #     ['2Triangle', '0.0', '5'], ['2Triangle', '0.0', '2'], ['2Triangle', '0.5', '5'], ['2Triangle', '0.5', '2'], ['2Triangle', '1.0', '5'], ['2Triangle', '1.0', '2'], 
        #     ['SShape', '0.0', '5'], ['SShape', '0.0', '2'], ['SShape', '0.5', '5'], ['SShape', '0.5', '2'], ['SShape', '1.0', '5'], ['SShape', '1.0', '2']
        #     ] # 18
        
        
    } # 37
    pyfile = 'base_models/trajecotory_user_linking/main.py'
    # task = 'eta'
    # dataset = 'Chengdu_Taxi_Sample1'
    # dataset = '0803_1'
    dataset = ['Chengdu_Taxi_Sample1_10_u50', 'Chengdu_Taxi_Sample1_10_u114']
    # model = 'DeepTTE'
    gpu_ids = ['0', '1', '2', '3']
    # attack_list = ['Trigger']
    grid_sizes = ['40', '20']
    malicious_labels = ['Single', 'All']
    attack_list = ['Random', 'Trigger', 'Translation', 'Stretch']
    domains_list = ['Spatial', 'Temporal', 'ST']
    # user_rate = ['0.5', '0.4', '0.3', '0.2', '0.1', '0.05', '0.01']
    user_rate = ['0.05', '0.10', '0.25']
    # malicious_label_ratios = ['1.0', '0.5']
    
    attack_positions = ['0.0', '0.5']
    attack_ratios = ['0.25']
    exp_id_prefix = '231028'
    output_prefix = ' >> /data/xuzhao/Codes/newTBA/logs/OUT-1028/'
    # with open('./cmd.txt', 'w+') as f:
    script_prefix = './cmd-1028-'
    i = 1
    setid = 0

    for dataset in dataset:
        for grid_size in grid_sizes:
            for label_type in malicious_labels:
                for attack in attack_list:
                    if attack == 'Trigger':
                        setid += 1
                        with open(script_prefix + attack + '-'  + str(setid) + '.txt', 'a+') as f:
                            f.write('\n')
                            f.write('\n')
                            f.write('\n')
                            setting = 'dataset\t' + dataset + '\tgrid size\t' + grid_size + '\tlabel type\t' + label_type
                            f.write(setting)
                            f.write('\n')
                            f.write('\n')
                            f.write('\n')
                            gpu_id = -1
                            for urate in user_rate:
                                for trigger_shape, trigger_position, trigger_size in attack_parameters_shortshort[attack]:
                                    gpu_id += 1
                                    if gpu_id >= len(gpu_ids):
                                        gpu_id = 0
                                    ii = str(i)
                                    ii = '0' * (4 - len(ii)) + ii
                                    exp_id = exp_id_prefix + ii
                                    f.write('echo ' + str(i) + '\n')
                                    # f.write('conda activate TSCS')
                                    cmd = 'python -u ' + pyfile + ' --dataset ' + dataset + ' --grid_size ' + grid_size + ' --attack_label ' + label_type + ' --exp_id ' + exp_id + ' --gpu_id ' + str(gpu_id) + ' --attack ' + attack + ' --user_rate ' + urate + ' --domain ' + domain + ' --attack_position ' + atkpos + ' --attack_ratio ' + atkratio + ' --trigger_shape ' + trigger_shape + ' --trigger_position ' + trigger_position + ' --trigger_size ' + trigger_size + output_prefix + str(i) + '.txt 2>&1 &'
                                    f.write(cmd)
                                    f.write('\n')
                                    if i % 4 == 0:
                                        f.write('\n')
                                    i += 1
                                    
                                f.write('\n')
                                f.write('\n')
                                
                    else:
                        for domain in domains_list:
                            setid += 1
                            with open(script_prefix + attack + '-' + domain + '-'  + str(setid) + '.txt', 'a+') as f:
                                f.write('\n')
                                f.write('\n')
                                f.write('\n')
                                setting = 'dataset\t' + dataset + '\tgrid size\t' + grid_size + '\tlabel type\t' + label_type 
                                f.write(setting)
                                f.write('\n')
                                f.write('\n')
                                f.write('\n')
                                gpu_id = -1
                                for urate in user_rate:
                                    for atkpos in attack_positions:
                                        for atkratio in attack_ratios:
                                            if attack == 'Random' and domain == 'ST':
                                                for meanS, stddevS, meanT, stddevT in attack_parameters_shortshort[attack][domain]:
                                                    gpu_id += 1
                                                    if gpu_id >= len(gpu_ids):
                                                        gpu_id = 0
                                                    ii = str(i)
                                                    ii = '0' * (4 - len(ii)) + ii
                                                    exp_id = exp_id_prefix + ii
                                                    f.write('echo ' + str(i) + '\n')
                                                    # f.write('conda activate TSCS')
                                                    cmd = 'python -u ' + pyfile + ' --dataset ' + dataset + ' --grid_size ' + grid_size + ' --attack_label ' + label_type + ' --exp_id ' + exp_id + ' --gpu_id ' + str(gpu_id) + ' --attack ' + attack + ' --user_rate ' + urate + ' --domain ' + domain + ' --attack_position ' + atkpos + ' --attack_ratio ' + atkratio + ' --meanS ' + meanS + ' --stddevS ' + stddevS + ' --meanT ' + meanT + ' --stddevT ' + stddevT + output_prefix + str(i) + '.txt 2>&1 &'
                                                    f.write(cmd)
                                                    f.write('\n')
                                                    i += 1
                                                f.write('\n')
                                            elif attack == 'Random' and domain == 'Temporal':
                                                for meanT, stddevT in attack_parameters_shortshort[attack][domain]:
                                                    gpu_id += 1
                                                    if gpu_id >= len(gpu_ids):
                                                        gpu_id = 0
                                                    ii = str(i)
                                                    ii = '0' * (4 - len(ii)) + ii
                                                    exp_id = exp_id_prefix + ii
                                                    f.write('echo ' + str(i) + '\n')
                                                    # f.write('conda activate TSCS')
                                                    cmd = 'python -u ' + pyfile + ' --dataset ' + dataset + ' --grid_size ' + grid_size + ' --attack_label ' + label_type + ' --exp_id ' + exp_id + ' --gpu_id ' + str(gpu_id) + ' --attack ' + attack + ' --user_rate ' + urate + ' --domain ' + domain + ' --attack_position ' + atkpos + ' --attack_ratio ' + atkratio + ' --meanT ' + meanT + ' --stddevT ' + stddevT + output_prefix + str(i) + '.txt 2>&1 &'
                                                    f.write(cmd)
                                                    f.write('\n')
                                                    i += 1
                                                f.write('\n')
                                            elif attack == 'Random' and domain == 'Spatial':
                                                for meanS, stddevS in attack_parameters_shortshort[attack][domain]:
                                                    gpu_id += 1
                                                    if gpu_id >= len(gpu_ids):
                                                        gpu_id = 0
                                                    ii = str(i)
                                                    ii = '0' * (4 - len(ii)) + ii
                                                    exp_id = exp_id_prefix + ii
                                                    f.write('echo ' + str(i) + '\n')
                                                    # f.write('conda activate TSCS')
                                                    cmd = 'python -u ' + pyfile + ' --dataset ' + dataset + ' --grid_size ' + grid_size + ' --attack_label ' + label_type + ' --exp_id ' + exp_id + ' --gpu_id ' + str(gpu_id) + ' --attack ' + attack + ' --user_rate ' + urate + ' --domain ' + domain + ' --attack_position ' + atkpos + ' --attack_ratio ' + atkratio + ' --meanS ' + meanS + ' --stddevS ' + stddevS + output_prefix + str(i) + '.txt 2>&1 &'
                                                    f.write(cmd)
                                                    f.write('\n')
                                                    i += 1
                                                f.write('\n')
                                            elif attack == 'Translation' and domain == 'ST':
                                                for deltaS, directionS, deltaT in attack_parameters_shortshort[attack][domain]:
                                                    gpu_id += 1
                                                    if gpu_id >= len(gpu_ids):
                                                        gpu_id = 0
                                                    ii = str(i)
                                                    ii = '0' * (4 - len(ii)) + ii
                                                    exp_id = exp_id_prefix + ii
                                                    f.write('echo ' + str(i) + '\n')
                                                    # f.write('conda activate TSCS')
                                                    cmd = 'python -u ' + pyfile + ' --dataset ' + dataset + ' --grid_size ' + grid_size + ' --attack_label ' + label_type + ' --exp_id ' + exp_id + ' --gpu_id ' + str(gpu_id) + ' --attack ' + attack + ' --user_rate ' + urate + ' --domain ' + domain + ' --attack_position ' + atkpos + ' --attack_ratio ' + atkratio + ' --deltaS ' + deltaS + ' --directionS ' + directionS + ' --deltaT ' + deltaT + output_prefix + str(i) + '.txt 2>&1 &'
                                                    f.write(cmd)
                                                    f.write('\n')
                                                    i += 1
                                                f.write('\n')
                                            elif attack == 'Translation' and domain == 'Temporal':
                                                for deltaT in attack_parameters_shortshort[attack][domain]:
                                                    gpu_id += 1
                                                    if gpu_id >= len(gpu_ids):
                                                        gpu_id = 0
                                                    ii = str(i)
                                                    ii = '0' * (4 - len(ii)) + ii
                                                    exp_id = exp_id_prefix + ii
                                                    f.write('echo ' + str(i) + '\n')
                                                    # f.write('conda activate TSCS')
                                                    cmd = 'python -u ' + pyfile + ' --dataset ' + dataset + ' --grid_size ' + grid_size + ' --attack_label ' + label_type + ' --exp_id ' + exp_id + ' --gpu_id ' + str(gpu_id) + ' --attack ' + attack + ' --user_rate ' + urate + ' --domain ' + domain + ' --attack_position ' + atkpos + ' --attack_ratio ' + atkratio + ' --deltaT ' + deltaT + output_prefix + str(i) + '.txt 2>&1 &'
                                                    f.write(cmd)
                                                    f.write('\n')
                                                    i += 1
                                                f.write('\n')
                                            elif attack == 'Translation' and domain == 'Spatial':
                                                for deltaS, directionS in attack_parameters_shortshort[attack][domain]:
                                                    gpu_id += 1
                                                    if gpu_id >= len(gpu_ids):
                                                        gpu_id = 0
                                                    ii = str(i)
                                                    ii = '0' * (4 - len(ii)) + ii
                                                    exp_id = exp_id_prefix + ii
                                                    f.write('echo ' + str(i) + '\n')
                                                    # f.write('conda activate TSCS')
                                                    cmd = 'python -u ' + pyfile + ' --dataset ' + dataset + ' --grid_size ' + grid_size + ' --attack_label ' + label_type + ' --exp_id ' + exp_id + ' --gpu_id ' + str(gpu_id) + ' --attack ' + attack + ' --user_rate ' + urate + ' --domain ' + domain + ' --attack_position ' + atkpos + ' --attack_ratio ' + atkratio + ' --deltaS ' + deltaS + ' --directionS ' + directionS + output_prefix + str(i) + '.txt 2>&1 &'
                                                    f.write(cmd)
                                                    f.write('\n')
                                                    i += 1
                                                f.write('\n')
                                            elif attack == 'Stretch':
                                                for stretch_length in attack_parameters_shortshort[attack]:
                                                    gpu_id += 1
                                                    if gpu_id >= len(gpu_ids):
                                                        gpu_id = 0
                                                    ii = str(i)
                                                    ii = '0' * (4 - len(ii)) + ii
                                                    exp_id = exp_id_prefix + ii
                                                    f.write('echo ' + str(i) + '\n')
                                                    # f.write('conda activate TSCS')
                                                    cmd = 'python -u ' + pyfile + ' --dataset ' + dataset + ' --grid_size ' + grid_size + ' --attack_label ' + label_type + ' --exp_id ' + exp_id + ' --gpu_id ' + str(gpu_id) + ' --attack ' + attack + ' --user_rate ' + urate + ' --domain ' + domain + ' --attack_position ' + atkpos + ' --attack_ratio ' + atkratio + ' --stretch_length ' + stretch_length + output_prefix + str(i) + '.txt 2>&1 &'
                                                    f.write(cmd)
                                                    f.write('\n')
                                                    i += 1
                                                f.write('\n')

















    # for dataset in dataset:
    #     for grid_size in grid_sizes:
    #         for label_type in malicious_labels:
    #             for urate in user_rate:
    #                 for atkpos in attack_positions:
    #                     for atkratio in attack_ratios:
    #                         setid += 1
    #                         # if gpuid >= len(gpu_ids):
    #                         #     gpuid = 0
    #                         with open(script_prefix + str(setid) + '.txt', 'a+') as f:
    #                             f.write('\n')
    #                             f.write('\n')
    #                             f.write('\n')
    #                             setting = 'dataset\t' + dataset + '\tgrid size\t' + grid_size + '\tlabel type\t' + label_type + '\tuser_rate\t' + urate + '\tattack_position\t' + atkpos + '\tattack_ratio\t' + atkratio
    #                             f.write(setting)
    #                             f.write('\n')
    #                             f.write('\n')
    #                             f.write('\n')
    #                             gpu_id = -1
                                
    #                             for attack in attack_list:
    #                                 for domain in domains_list:
    #                                     if attack == 'Random' and domain == 'ST':
    #                                         for meanS, stddevS, meanT, stddevT in attack_parameters_shortshort[attack][domain]:
    #                                             gpu_id += 1
    #                                             if gpu_id >= len(gpu_ids):
    #                                                 gpu_id = 0
    #                                             ii = str(i)
    #                                             ii = '0' * (4 - len(ii)) + ii
    #                                             exp_id = exp_id_prefix + ii
    #                                             f.write('echo ' + str(i) + '\n')
    #                                             # f.write('conda activate TSCS')
    #                                             cmd = 'python -u ' + pyfile + ' --dataset ' + dataset + ' --grid_size ' + grid_size + ' --attack_label ' + label_type + ' --exp_id ' + exp_id + ' --gpu_id ' + str(gpu_id) + ' --attack ' + attack + ' --user_rate ' + urate + ' --domain ' + domain + ' --attack_position ' + atkpos + ' --attack_ratio ' + atkratio + ' --meanS ' + meanS + ' --stddevS ' + stddevS + ' --meanT ' + meanT + ' --stddevT ' + stddevT + output_prefix + str(i) + '.txt 2>&1 &'
    #                                             f.write(cmd)
    #                                             f.write('\n')
    #                                             i += 1
    #                                         f.write('\n')
    #                                     elif attack == 'Random' and domain == 'Temporal':
    #                                         for meanT, stddevT in attack_parameters_shortshort[attack][domain]:
    #                                             gpu_id += 1
    #                                             if gpu_id >= len(gpu_ids):
    #                                                 gpu_id = 0
    #                                             ii = str(i)
    #                                             ii = '0' * (4 - len(ii)) + ii
    #                                             exp_id = exp_id_prefix + ii
    #                                             f.write('echo ' + str(i) + '\n')
    #                                             # f.write('conda activate TSCS')
    #                                             cmd = 'python -u ' + pyfile + ' --dataset ' + dataset + ' --grid_size ' + grid_size + ' --attack_label ' + label_type + ' --exp_id ' + exp_id + ' --gpu_id ' + str(gpu_id) + ' --attack ' + attack + ' --user_rate ' + urate + ' --domain ' + domain + ' --attack_position ' + atkpos + ' --attack_ratio ' + atkratio + ' --meanT ' + meanT + ' --stddevT ' + stddevT + output_prefix + str(i) + '.txt 2>&1 &'
    #                                             f.write(cmd)
    #                                             f.write('\n')
    #                                             i += 1
    #                                         f.write('\n')
    #                                     elif attack == 'Random' and domain == 'Spatial':
    #                                         for meanS, stddevS in attack_parameters_shortshort[attack][domain]:
    #                                             gpu_id += 1
    #                                             if gpu_id >= len(gpu_ids):
    #                                                 gpu_id = 0
    #                                             ii = str(i)
    #                                             ii = '0' * (4 - len(ii)) + ii
    #                                             exp_id = exp_id_prefix + ii
    #                                             f.write('echo ' + str(i) + '\n')
    #                                             # f.write('conda activate TSCS')
    #                                             cmd = 'python -u ' + pyfile + ' --dataset ' + dataset + ' --grid_size ' + grid_size + ' --attack_label ' + label_type + ' --exp_id ' + exp_id + ' --gpu_id ' + str(gpu_id) + ' --attack ' + attack + ' --user_rate ' + urate + ' --domain ' + domain + ' --attack_position ' + atkpos + ' --attack_ratio ' + atkratio + ' --meanS ' + meanS + ' --stddevS ' + stddevS + output_prefix + str(i) + '.txt 2>&1 &'
    #                                             f.write(cmd)
    #                                             f.write('\n')
    #                                             i += 1
    #                                         f.write('\n')
    #                                     elif attack == 'Translation' and domain == 'ST':
    #                                         for deltaS, directionS, deltaT in attack_parameters_shortshort[attack][domain]:
    #                                             gpu_id += 1
    #                                             if gpu_id >= len(gpu_ids):
    #                                                 gpu_id = 0
    #                                             ii = str(i)
    #                                             ii = '0' * (4 - len(ii)) + ii
    #                                             exp_id = exp_id_prefix + ii
    #                                             f.write('echo ' + str(i) + '\n')
    #                                             # f.write('conda activate TSCS')
    #                                             cmd = 'python -u ' + pyfile + ' --dataset ' + dataset + ' --grid_size ' + grid_size + ' --attack_label ' + label_type + ' --exp_id ' + exp_id + ' --gpu_id ' + str(gpu_id) + ' --attack ' + attack + ' --user_rate ' + urate + ' --domain ' + domain + ' --attack_position ' + atkpos + ' --attack_ratio ' + atkratio + ' --deltaS ' + deltaS + ' --directionS ' + directionS + ' --deltaT ' + deltaT + output_prefix + str(i) + '.txt 2>&1 &'
    #                                             f.write(cmd)
    #                                             f.write('\n')
    #                                             i += 1
    #                                         f.write('\n')
    #                                     elif attack == 'Translation' and domain == 'Temporal':
    #                                         for deltaT in attack_parameters_shortshort[attack][domain]:
    #                                             gpu_id += 1
    #                                             if gpu_id >= len(gpu_ids):
    #                                                 gpu_id = 0
    #                                             ii = str(i)
    #                                             ii = '0' * (4 - len(ii)) + ii
    #                                             exp_id = exp_id_prefix + ii
    #                                             f.write('echo ' + str(i) + '\n')
    #                                             # f.write('conda activate TSCS')
    #                                             cmd = 'python -u ' + pyfile + ' --dataset ' + dataset + ' --grid_size ' + grid_size + ' --attack_label ' + label_type + ' --exp_id ' + exp_id + ' --gpu_id ' + str(gpu_id) + ' --attack ' + attack + ' --user_rate ' + urate + ' --domain ' + domain + ' --attack_position ' + atkpos + ' --attack_ratio ' + atkratio + ' --deltaT ' + deltaT + output_prefix + str(i) + '.txt 2>&1 &'
    #                                             f.write(cmd)
    #                                             f.write('\n')
    #                                             i += 1
    #                                         f.write('\n')
    #                                     elif attack == 'Translation' and domain == 'Spatial':
    #                                         for deltaS, directionS in attack_parameters_shortshort[attack][domain]:
    #                                             gpu_id += 1
    #                                             if gpu_id >= len(gpu_ids):
    #                                                 gpu_id = 0
    #                                             ii = str(i)
    #                                             ii = '0' * (4 - len(ii)) + ii
    #                                             exp_id = exp_id_prefix + ii
    #                                             f.write('echo ' + str(i) + '\n')
    #                                             # f.write('conda activate TSCS')
    #                                             cmd = 'python -u ' + pyfile + ' --dataset ' + dataset + ' --grid_size ' + grid_size + ' --attack_label ' + label_type + ' --exp_id ' + exp_id + ' --gpu_id ' + str(gpu_id) + ' --attack ' + attack + ' --user_rate ' + urate + ' --domain ' + domain + ' --attack_position ' + atkpos + ' --attack_ratio ' + atkratio + ' --deltaS ' + deltaS + ' --directionS ' + directionS + output_prefix + str(i) + '.txt 2>&1 &'
    #                                             f.write(cmd)
    #                                             f.write('\n')
    #                                             i += 1
    #                                         f.write('\n')
    #                                 if attack == 'Stretch':
    #                                     for stretch_length in attack_parameters_shortshort[attack]:
    #                                         gpu_id += 1
    #                                         if gpu_id >= len(gpu_ids):
    #                                             gpu_id = 0
    #                                         ii = str(i)
    #                                         ii = '0' * (4 - len(ii)) + ii
    #                                         exp_id = exp_id_prefix + ii
    #                                         f.write('echo ' + str(i) + '\n')
    #                                         # f.write('conda activate TSCS')
    #                                         cmd = 'python -u ' + pyfile + ' --dataset ' + dataset + ' --grid_size ' + grid_size + ' --attack_label ' + label_type + ' --exp_id ' + exp_id + ' --gpu_id ' + str(gpu_id) + ' --attack ' + attack + ' --user_rate ' + urate + ' --domain ' + domain + ' --attack_position ' + atkpos + ' --attack_ratio ' + atkratio + ' --stretch_length ' + stretch_length + output_prefix + str(i) + '.txt 2>&1 &'
    #                                         f.write(cmd)
    #                                         f.write('\n')
    #                                         i += 1
    #                                     f.write('\n')
    #                                 elif attack == 'Trigger':
    #                                     for trigger_shape, trigger_position, trigger_size in attack_parameters_shortshort[attack]:
    #                                         gpu_id += 1
    #                                         if gpu_id >= len(gpu_ids):
    #                                             gpu_id = 0
    #                                         ii = str(i)
    #                                         ii = '0' * (4 - len(ii)) + ii
    #                                         exp_id = exp_id_prefix + ii
    #                                         f.write('echo ' + str(i) + '\n')
    #                                         # f.write('conda activate TSCS')
    #                                         cmd = 'python -u ' + pyfile + ' --dataset ' + dataset + ' --grid_size ' + grid_size + ' --attack_label ' + label_type + ' --exp_id ' + exp_id + ' --gpu_id ' + str(gpu_id) + ' --attack ' + attack + ' --user_rate ' + urate + ' --domain ' + domain + ' --attack_position ' + atkpos + ' --attack_ratio ' + atkratio + ' --trigger_shape ' + trigger_shape + ' --trigger_position ' + trigger_position + ' --trigger_size ' + trigger_size + output_prefix + str(i) + '.txt 2>&1 &'
    #                                         f.write(cmd)
    #                                         f.write('\n')
    #                                         if i % 4 == 0:
    #                                             f.write('\n')
    #                                         i += 1
                                            
    #                                     f.write('\n')
    #                                     f.write('\n')

        #             f.write('\n')
        #         f.write('\n')
        #     f.write('\n')
        # f.write('\n')
    
    # python run_model.py --task eta --dataset Chengdu_Taxi_Sample1 --model DeepTTE --exp_id 23092902 --gpu_id 1 --attack Stretch --stretch_length 60.0 --user_rate 0.1 --attack_position 0.5 --attack_ratio 0.5 >> /home/xuzhao/TrajectoryBA/scripts/script_logs/2.txt 2>&1 &
                
if __name__ == "__main__":
    main()

