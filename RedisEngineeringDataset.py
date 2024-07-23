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

    @classmethod
    def create_dataset(cls, dataset_configs:list, input_size:tuple, default_img_size:tuple, is_preshuffle:bool, \
                       transforms, **kwargs):
        filename_obj_factory = FilenameObjFactory()
        filename_objs, labels, string_labels, class_weights, _ = EngineeringDataset.init_lists(True, output_class_map)
        dataset_distribution = {}
        for dataset_config in dataset_configs:
            with open(dataset_config['data_dir'], 'r') as f:
                annotation = json.load(f)
            for img_info_map in annotation['annotations']:
                filename_obj = filename_obj_factory.create(img_info_map['filename'])
                filename_objs.append(filename_obj)          
            # wether do downsampling or not
            if dataset_config['downsampling_rate'] > 0:
                filename_objs, = EngineeringDataset.data_downsampling(filename_objs, dataset_config['downsampling_rate'])
            # wether do downsampling or not
            if dataset_config['upsampling_rate'] > 0:
                filename_objs = EngineeringDataset.data_upsampling(filename_objs, dataset_config['upsampling_rate'])
        print(' [{} set]: '.format({0: 'train', 1: 'valid', 2: 'test'}[EngineeringDataset.dataset_index]))
        print('number of images: ', len(filename_objs))
        # pre-shuffle dataset is necessary
        if is_preshuffle:
            filename_objs = EngineeringDataset.shuffle_lists(filename_objs)
        EngineeringDataset.dataset_index += 1
        return cls(filename_objs, input_size, default_img_size, transforms, **kwargs)
