import random
import json
import copy
import time
import multiprocessing

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    dataset_index = 0
    
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_objs:list, transform, **kwargs):
        super(BaseDataset, self).__init__()
        self.data_objs = data_objs  # list
        self.transform = transform
        self.custom_load_config = kwargs

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
    def init_lists_mp(manager):
        assert manager is not None
        data_objs = manager.list()
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

    @staticmethod
    def get_annotations(dataset_config:dict, data_obj_factory):
        data_objs = BaseDataset.init_lists()
        with open(dataset_config['path'], 'r') as f:
            annotation = json.load(f)
        for img_info_map in annotation['annotations']:
            data_obj = data_obj_factory.create(img_info_map['filename'])
            data_objs.append(data_obj)          
        # wether do downsampling or not
        if dataset_config['downsampling_rate'] > 0:
            data_objs = BaseDataset.data_downsampling(data_objs, dataset_config['downsampling_rate'])
        # wether do downsampling or not
        if dataset_config['upsampling_rate'] > 0:
            data_objs = BaseDataset.data_upsampling(data_objs, dataset_config['upsampling_rate'])
        return data_objs

    @staticmethod
    def get_annotations_mp(dataset_config:dict, data_obj_factory, shared_var:list):
        new_data_objs = BaseDataset.get_annotations(dataset_config, data_obj_factory)
        shared_var += new_data_objs
        return shared_var

    @classmethod
    def create_dataset(cls, dataset_configs:list, is_preshuffle:bool, transforms, data_obj_factory, \
                       parallel_gather:str, **kwargs):
        # there is only one data_obj_factory instance in the runtime
        start_time = time.time()
        if parallel_gather == 'default':
            data_objs = cls.init_lists()
            for dataset_config in dataset_configs:
                data_objs += BaseDataset.get_annotations(dataset_config, data_obj_factory)
        elif parallel_gather == 'multi_process':
            print('sub-processes started')
            pool = multiprocessing.Pool(8)
            manager = multiprocessing.Manager()
            data_objs = cls.init_lists_mp(manager)  # shared variables
            for dataset_config in dataset_configs:
                # pool.apply(func=BaseDataset.get_annotations, args=(dataset_config, data_obj_factory))  # Each process is executed before the next process is executed, which is basically not used.
                pool.apply_async(func=BaseDataset.get_annotations_mp, args=(dataset_config, data_obj_factory, data_objs))  # Here 'func' is the function called, args is the input, Bar is the callback function, the input of the callback function is the return value of func.
            pool.close()  # No new processes will be added to the pool after close, and the join function waits for all child processes to finish.
            pool.join()  # Wait for the process to finish running, call the close function first or you will get an error!
            print('sub-processes ended')
        else:
            raise ValueError(f'parallel_gather type: {parallel_gather} is not supported')
        print('[Dataset] time for loading filenames:', time.time() - start_time)
        print('[Dataset] [{} set initialized successfully]: '.format({0: 'train', 1: 'valid', 2: 'test'}[cls.dataset_index]))
        print('[Dataset] number of images: ', len(data_objs))
        # pre-shuffle dataset is necessary
        if is_preshuffle:
            data_objs = cls.shuffle_lists(data_objs)
        cls.dataset_index += 1
        return cls(data_objs, transforms, **kwargs)