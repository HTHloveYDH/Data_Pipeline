from io import BytesIO

import redis  # pip install redis

from data_pipeline.dataset.BaseDataset import BaseDataset
import global_vars_manager


class RedisDataset(BaseDataset):
    'Characterizes a redis optimized dataset for PyTorch'
    def __init__(self, data_objs:list, transform, **kwargs):
        super(RedisDataset, self).__init__(data_objs, transform, **kwargs)
        self.redis = redis.Redis(
            host=global_vars_manager.get_global_var('REDIS_HOST'), 
            port=global_vars_manager.get_global_var('REDIS_PORT'),  # 6379
            decode_responses=False,  # return bytes stream 
            db=0
        )  # self.redis is a in-memory database

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_objs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # load image
        img_data_loader = self.data_objs[index]
        img_data_loader.data = self.redis.get(img_data_loader.data)  # bytes stream
        image = img_data_loader.load_data()
        # image augmentation and rescale
        image = self.transform(image, **{'random_aug_config': self.random_aug_config})
        return image