from data_pipeline.dataset.BaseDataset import BaseDataset


class GeneralDataset(BaseDataset):
    'Characterizes a general dataset for PyTorch'
    def __init__(self, data_objs:list, transform, **kwargs):
        super(GeneralDataset, self).__init__(data_objs, transform, **kwargs)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_objs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # load image
        filename_obj = self.data_objs[index]
        image = filename_obj.load()
        # image augmentation and rescale
        image = self.transform(image, **{'random_aug_config': self.random_aug_config})
        return image