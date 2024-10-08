from io import BytesIO
import pickle

import redis  # pip install redis

from data_pipeline.dataset.BaseDataset import BaseDataset
import global_vars_manager


class RedisDataset(BaseDataset):
    'Characterizes a redis optimized dataset for PyTorch'
    def __init__(self, data_objs_list:list, transform, **kwargs):
        super(RedisDataset, self).__init__(data_objs_list, transform, **kwargs)
        self.redis = redis.Redis(
            host=global_vars_manager.get_global_var('REDIS_HOST'), 
            port=global_vars_manager.get_global_var('REDIS_PORT'),  # 6379
            db=global_vars_manager.get_global_var('REDIS_DB'),
            password=global_vars_manager.get_global_var('REDIS_PASSWORD'),
            decode_responses=False,  # return bytes stream 
            retry_on_timeout=global_vars_manager.get_global_var('REDIS_RETRY_ON_TIMEOUT'),
            username=global_vars_manager.get_global_var('REDIS_USERNAME')
        )  # self.redis is a in-memory database
        ret = self.redis.ping()
        print('[DATASET] ', {True: 'redis client link established', False: 'redis client link failed'}[ret])

    def __getitem__(self, index):
        'Generates one sample of data'
        # load image
        img_data_loader = self.data_objs_list[0][index]
        data = self.redis.get(img_data_loader.key)  # type: bytes
        assert isinstance(data, bytes)
        image = img_data_loader.load_data(data)
        # image augmentation and rescale
        image = self.transform(image, **self.custom_load_config)
        return image
    
class RedisDatasetV2(BaseDataset):
    'Characterizes a redis optimized dataset for PyTorch'
    def __init__(self, data_objs_list:list, transform, **kwargs):
        super(RedisDatasetV2, self).__init__(data_objs_list, transform, **kwargs)
        self.redis = redis.Redis(
            host=global_vars_manager.get_global_var('REDIS_HOST'), 
            port=global_vars_manager.get_global_var('REDIS_PORT'),  # 6379
            db=global_vars_manager.get_global_var('REDIS_DB'),
            password=global_vars_manager.get_global_var('REDIS_PASSWORD'),
            decode_responses=False,  # return bytes stream 
            retry_on_timeout=global_vars_manager.get_global_var('REDIS_RETRY_ON_TIMEOUT'),
            username=global_vars_manager.get_global_var('REDIS_USERNAME')
        )  # self.redis is a in-memory database
        ret = self.redis.ping()
        print('[DATASET] ', {True: 'redis client link established', False: 'redis client link failed'}[ret])

    def __getitem__(self, index):
        'Generates one sample of data'
        # load image
        img_data_loader = self.data_objs_list[0][index]
        image_bytes = self.redis.get(img_data_loader.key)  # type: bytes
        assert isinstance(image_bytes, bytes)
        data = pickle.loads(image_bytes)  # numpy.ndarray
        image = img_data_loader.load_data(data)
        # image augmentation and rescale
        image = self.transform(image, **self.custom_load_config)
        return image