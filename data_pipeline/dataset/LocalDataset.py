from io import BytesIO

import redis  # pip install redis

from data_pipeline.dataset.BaseDataset import BaseDataset
import global_vars_manager


class LocalDataset(BaseDataset):
    'Characterizes a local dataset for PyTorch'
    def __init__(self, data_objs:list, transform, **kwargs):
        super(LocalDataset, self).__init__(data_objs, transform, **kwargs)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_objs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # load image
        img_data_loader = self.data_objs[index]
        image = img_data_loader.load_data()
        # image augmentation and rescale
        image = self.transform(image, **{'random_aug_config': self.random_aug_config})
        return image