

def load_data_atk(dataset, read_pkl, grid_size):
    """[This is a function used to load data]

    Args:
        dataset ([str]): [The name of the dataset]
        read_pkl ([bool]): [Whether to use the preprocessed data file]
        grid_size ([int]): [Grid size of preprocessing trajectory data division]

    Returns:
        [type]: [description]
    """
    if read_pkl == False:
        if dataset == "Chengdu_Sample1":
            raw_path = './data/tul/chengdu/raw/Chengdu_Sample1.dyna'
        elif dataset == "Chengdu_Sample12":
            raw_path = './data/tul/chengdu/raw/Chengdu_Sample12.dyna'
        elif dataset == "Chengdu_20140803_1":
            raw_path = './data/tul/chengdu/raw/Chengdu_20140803_1.dyna'
        elif dataset == 'gowalla-all':
            raw_path = './data/tul/gowalla/raw/gowalla-all.csv'
        elif dataset == 'shenzhen-all':
            raw_path = './data/tul/shenzhen/raw/shenzhen-all.csv'
        elif dataset == 'geolife-all':
            raw_path = './data/tul/geolife/raw/geolife-all.csv'
        else:
            print('The dataset does not exist')
            exit()
        return raw_pathget_data_and_graph(raw_path, read_pkl, grid_size)