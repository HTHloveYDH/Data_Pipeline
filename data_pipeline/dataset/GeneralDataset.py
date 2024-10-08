from data_pipeline.dataset.BaseDataset import BaseDataset


class GeneralDataset(BaseDataset):
    'Characterizes a general dataset for PyTorch'
    def __init__(self, data_objs_list:list, transform, **kwargs):
        super(GeneralDataset, self).__init__(data_objs_list, transform, **kwargs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # load image
        filename_obj = self.data_objs_list[0][index]
        image = filename_obj.load()
        # image augmentation and rescale
        image = self.transform(image, **self.custom_load_config)
        return image