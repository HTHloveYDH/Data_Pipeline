from EngineeringDataset import EngineeringDataset


class RedisEngineeringDataset(EngineeringDataset):
    'Characterizes a redis optimized dataset for PyTorch'
    def __init__(
        self, filename_objs:list, input_size:tuple, default_img_size:tuple, transform, **kwargs):
        super(RedisEngineeringDataset, self).__init__(
            filename_objs, input_size, default_img_size, transform, **kwargs
        )
        self.redis = None

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.filename_objs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # load image
        filename_obj = self.filename_objs[index]
        image = self.load_image(filename_obj)
        # do not need to convert to tensor (torch.Tensor) here, or you may get TypeError: 
        # pic should be PIL Image or ndarray. Got <class 'torch.Tensor'> when your transform is of 'V1'
        # image = T.functional.to_tensor(image)  # should not convert to tensor here
        # image augmentation and rescale
        image = self.transform(image, **{'random_aug_config': self.random_aug_config})
        return image