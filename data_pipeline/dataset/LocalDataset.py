from io import BytesIO

from data_pipeline.dataset.BaseDataset import BaseDataset


class LocalDataset(BaseDataset):
    'Characterizes a local dataset for PyTorch'
    def __init__(self, data_objs:list, transform, **kwargs):
        super(LocalDataset, self).__init__(data_objs, transform, **kwargs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # load image
        img_data_loader = self.data_objs[index]
        image = img_data_loader.load_data(img_data_loader.key)
        # image augmentation and rescale
        image = self.transform(image, **self.custom_load_config)
        return image