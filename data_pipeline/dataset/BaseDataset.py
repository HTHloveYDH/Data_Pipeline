import random
import json
import copy
import time
import multiprocessing

from torch.utils.data import Dataset

# from utils import init_mp


class BaseDataset(Dataset):
    dataset_index = 0
    
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_objs_list:list, transform, **kwargs):
        super(BaseDataset, self).__init__()
        self.data_objs_list = data_objs_list
        self.transform = transform
        self.custom_load_config = kwargs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_objs_list[0])  # images

    def __getitem__(self, index):
        raise NotImplementedError(" Can not call '__getitem__' via base class 'BaseDataset'! ")

    @staticmethod
    def init_lists():
        img_data_objs = []
        return [img_data_objs]

    @staticmethod
    def init_lists_mp(manager):
        assert manager is not None
        img_data_objs = manager.list()
        data_objs_list = manager.list()
        data_objs_list.append(img_data_objs)
        return data_objs_list

    @staticmethod
    def data_downsampling(data_objs_list:list, downsampling_rate:float):
        for i in range(0, len(data_objs_list)):
            data_objs_list[i] = data_objs_list[i][::int(1 / downsampling_rate)]
    
    @staticmethod
    def duplicate_list(input_list:list, upsampling_rate:int):
        delta = []
        for _ in range(0, upsampling_rate-1):
            delta += copy.deepcopy(input_list)
        input_list += delta
    
    @staticmethod
    def data_upsampling(data_objs_list:list, upsampling_rate:int):
        for i in range(0, len(data_objs_list)):
            BaseDataset.duplicate_list(data_objs_list[i], upsampling_rate)
    
    @staticmethod
    def shuffle_lists(data_objs_list:list, seed=None):
        if seed is None:
            seed = random.randint(0, 100)
        # for images
        random.seed(seed)  # set random seed
        random.shuffle(data_objs_list[0])
        # for other data
        for i in range(1, len(data_objs_list)):
            random.seed(seed)  # set same random seed again and again
            random.shuffle(data_objs_list[i])
        return data_objs_list

    @staticmethod
    def get_annotations(dataset_config:dict, img_data_obj_factory):
        new_data_objs_list = BaseDataset.init_lists()
        with open(dataset_config['path'], 'r') as f:
            annotation = json.load(f)
        for img_info_map in annotation['annotations']:
            img_data_obj = img_data_obj_factory.create(img_info_map['filename'])
            new_data_objs_list[0].append(img_data_obj)          
        # wether do downsampling or not
        if dataset_config['downsampling_rate'] > 0:
            BaseDataset.data_downsampling(new_data_objs_list, dataset_config['downsampling_rate'])
        # wether do downsampling or not
        if dataset_config['upsampling_rate'] > 0:
            BaseDataset.data_upsampling(new_data_objs_list, dataset_config['upsampling_rate'])
        return new_data_objs_list

    @staticmethod
    def gather_annotations(dataset_config:dict, img_data_obj_factory, data_objs_list:list):
        new_data_objs_list = BaseDataset.get_annotations(dataset_config, img_data_obj_factory)
        data_objs_list[0] += new_data_objs_list[0]

    @staticmethod
    def gather_annotations_mp(dataset_config:dict, img_data_obj_factory, shared_vars:list, shared_lock):
        new_vars = BaseDataset.get_annotations(dataset_config, img_data_obj_factory)
        # update shared variables (shared between processes)
        shared_lock.acquire()
        shared_vars[0] += new_vars[0]
        shared_lock.release()

    @classmethod
    def create_dataset(cls, dataset_configs:list, is_preshuffle:bool, transforms, img_data_obj_factory, \
                       parallel_gather:str, **kwargs):
        # there is only one img_data_obj_factory instance in the runtime
        start_time = time.time()
        if parallel_gather == 'default':
            data_objs_list = cls.init_lists()
            for dataset_config in dataset_configs:
                BaseDataset.gather_annotations(dataset_config, img_data_obj_factory, data_objs_list)
        elif parallel_gather == 'multi_process':
            print('sub-processes started')
            # another two ways for using poll with lock: https://stackoverflow.com/questions/25557686/python-sharing-a-lock-between-processes
            # current method: https://blog.csdn.net/m0_48978908/article/details/119967943
            pool = multiprocessing.Pool(8)
            manager = multiprocessing.Manager()
            shared_lock = manager.Lock()
            shared_vars = cls.init_lists_mp(manager)  # shared variables
            for dataset_config in dataset_configs:
                # pool.apply(
                #     func=BaseDataset.gather_annotations_mp, 
                #     args=(dataset_config, img_data_obj_factory, shared_vars, shared_lock)
                # )  # Each process is executed before the next process is executed, which is basically not used.
                pool.apply_async(
                    func=BaseDataset.gather_annotations_mp, 
                    args=(dataset_config, img_data_obj_factory, shared_vars, shared_lock)
                )  # Here 'func' is the function called, args is the input, Bar is the callback function, the input of the callback function is the return value of func.
            pool.close()  # No new processes will be added to the pool after close, and the join function waits for all child processes to finish.
            pool.join()  # Wait for the process to finish running, call the close function first or you will get an error!
            print('sub-processes ended')
            data_objs_list = shared_vars
        else:
            raise ValueError(f'parallel_gather type: {parallel_gather} is not supported')
        print('[Dataset] time for loading filenames:', time.time() - start_time)
        print('[Dataset] [{} set initialized successfully]: '.format({0: 'train', 1: 'valid', 2: 'test'}[cls.dataset_index]))
        print('[Dataset] number of images: ', len(data_objs_list[0]))  # number of images
        # pre-shuffle dataset is necessary
        if is_preshuffle:
            data_objs_list = cls.shuffle_lists(data_objs_list)
        cls.dataset_index += 1
        return cls(data_objs_list, transforms, **kwargs)