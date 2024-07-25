import random
import json
import copy

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    dataset_index = 0
    
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_objs:list, transform, **kwargs):
        super(BaseDataset, self).__init__()
        self.data_objs = data_objs  # list
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_objs)

    def __getitem__(self, index):
        raise NotImplementedError(" Can not call '__getitem__' via base class 'BaseDataset'! ")

    @staticmethod
    def init_lists():
        data_objs = []
        return data_objs

    @staticmethod
    def data_downsampling(data_objs:list, downsampling_rate:float):
        data_objs = data_objs[::int(1 / downsampling_rate)]
        return data_objs
    
    @staticmethod
    def duplicate_list(input_list:list, upsampling_rate:int):
        delta = []
        for _ in range(0, upsampling_rate-1):
            delta += copy.deepcopy(input_list)
        input_list += delta
        return input_list
    
    @staticmethod
    def data_upsampling(data_objs:list, upsampling_rate:int):
        data_objs = BaseDataset.duplicate_list(data_objs, upsampling_rate)
        return data_objs
    
    @staticmethod
    def shuffle_lists(data_objs:list, seed=None):
        if seed is None:
            seed = random.randint(0, 100)
        random.seed(seed)  # set random seed
        random.shuffle(data_objs)
        return data_objs

    @classmethod
    def create_dataset(cls, dataset_configs:list, is_preshuffle:bool, transforms, data_obj_factory, **kwargs):
        # there is only one data_obj_factory instance in the runtime
        data_objs = cls.init_lists()
        dataset_distribution = {}
        for dataset_config in dataset_configs:
            with open(dataset_config['data_dir'], 'r') as f:
                annotation = json.load(f)
            for img_info_map in annotation['annotations']:
                data_obj = data_obj_factory.create(img_info_map['filename'])
                data_objs.append(data_obj)          
            # wether do downsampling or not
            if dataset_config['downsampling_rate'] > 0:
                data_objs, = cls.data_downsampling(data_objs, dataset_config['downsampling_rate'])
            # wether do downsampling or not
            if dataset_config['upsampling_rate'] > 0:
                data_objs = cls.data_upsampling(data_objs, dataset_config['upsampling_rate'])
        print(' [{} set]: '.format({0: 'train', 1: 'valid', 2: 'test'}[cls.dataset_index]))
        print('number of images: ', len(data_objs))
        # pre-shuffle dataset is necessary
        if is_preshuffle:
            data_objs = cls.shuffle_lists(data_objs)
        cls.dataset_index += 1
        return cls(data_objs, transforms, **kwargs)