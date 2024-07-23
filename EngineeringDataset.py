from torch.utils.data import DataLoader, Dataset

from FilenameObjFactory import FilenameObjFactory


class EngineeringDataset(Dataset):
    dataset_index = 0
    'Characterizes a dataset for PyTorch'
    def __init__(
        self, filename_objs:list, input_size:tuple, default_img_size:tuple, img_mode:str, transform, 
        **kwargs):
        super(EngineeringDataset, self).__init__()
        self.filename_objs = filename_objs
        self.input_size = input_size
        self.default_img_size = default_img_size
        self.transform = transform
        self.random_aug_config = kwargs.get('random_aug_config', {})

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.filename_objs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # load image
        filename_obj = self.filename_objs[index]
        image = self.load_image(filename_obj, img_mode)
        # do not need to convert to tensor (torch.Tensor) here, or you may get TypeError: 
        # pic should be PIL Image or ndarray. Got <class 'torch.Tensor'> when your transform is of 'V1'
        # image = T.functional.to_tensor(image)  # should not convert to tensor here
        # image augmentation and rescale
        image = self.transform(image, **{'random_aug_config': self.random_aug_config})
        return image

    def load_image(self, filename_obj, img_mode:str):
        return filename_obj.load(img_mode)

    @staticmethod
    def data_downsampling(filename_objs:list, downsampling_rate:float):
        filename_objs = filename_objs[::int(1 / downsampling_rate)]
        return filename_objs
    
    @staticmethod
    def duplicate_list(input:list, upsampling_rate:int):
        delta = []
        for _ in range(0, upsampling_rate-1):
            delta += copy.deepcopy(input)
        input += delta
        return input
    
    @staticmethod
    def data_upsampling(filename_objs:list, upsampling_rate:int):
        filename_objs = EngineeringDataset.duplicate_list(filename_objs, upsampling_rate)
        return filename_objs
    
    @staticmethod
    def shuffle_lists(filename_objs:list, seed=None):
        if seed is None:
            seed = random.randint(0, 100)
        random.seed(seed)  # set random seed
        random.shuffle(filename_objs)
        return filename_objs

    @classmethod
    def create_dataset(cls, dataset_configs:list, input_size:tuple, default_img_size:tuple, img_mode:str, \
                       is_preshuffle:bool, transforms, **kwargs):
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
        return cls(filename_objs, input_size, default_img_size, img_mode, transforms, **kwargs)
