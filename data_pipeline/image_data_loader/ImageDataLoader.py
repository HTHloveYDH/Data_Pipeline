from PIL import Image

import numpy as np

import global_vars_manager


class ImageDataLoader:
    '''base class'''
    img_mode = global_vars_manager.get_global_var('IMG_MODE')

    def __init__(self):
        pass

    def load_data(self, data):
        raise NotImplementedError(" Can not call this member function via base class 'ImageFileLoader'! ")

class NormalImageDataLoader(ImageDataLoader):
    def __init__(self):
        super(NormalImageDataLoader, self).__init__()
    
    def load_data(self, data):
        # data is either type 'byte' or type 'str'
        return Image.open(data, mode=NormalImageDataLoader.img_mode)  # PIL image, data can be either string or bytes stream

class ArrayImageDataLoader(ImageDataLoader):
    def __init__(self):
        super(ArrayImageDataLoader, self).__init__()
    
    def load_data(self, data):
        # data is type 'np.ndarray'
        return Image.fromarray(data, mode=ArrayImageDataLoader.img_mode)  # PIL image

class NpyImageDataLoader(ImageDataLoader):
    def __init__(self):
        super(NpyImageDataLoader, self).__init__()
    
    def load_data(self, data):
        # data is type 'str'
        numpy_data = np.load(data).astype(np.uint8)
        return Image.fromarray(numpy_data, mode=NpyImageDataLoader.img_mode)  # PIL image

class NpyImageDataLoaderV2(ImageDataLoader):
    def __init__(self):
        super(NpyImageDataLoaderV2, self).__init__()
    
    def load_data(self, data):
        # data is type 'byte'
        numpy_data = np.frombuffer(data, dtype=np.uint8)
        return Image.fromarray(numpy_data, mode=NpyImageDataLoaderV2.img_mode)  # PIL image