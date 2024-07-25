from PIL import Image

import numpy as np


class ImageDataLoader:
    '''base class'''
    def __init__(self):
        pass

    def load_data(self, data, img_mode:str):
        raise NotImplementedError(" Can not call this member function via base class 'ImageFileLoader'! ")

class NormalImageDataLoader(ImageDataLoader):
    def __init__(self):
        super(NormalImageDataLoader, self).__init__()
    
    def load_data(self, data, img_mode:str):
        # data is either type 'byte' or type 'str'
        return Image.open(data, mode=img_mode)  # PIL image, data can be either string or bytes stream

class ArrayImageDataLoader(ImageDataLoader):
    def __init__(self):
        super(ArrayImageDataLoader, self).__init__()
    
    def load_data(self, data, img_mode:str):
        # data is type 'np.ndarray'
        return Image.fromarray(data, mode=img_mode)  # PIL image

class NpyImageDataLoader(ImageDataLoader):
    def __init__(self):
        super(NpyImageDataLoader, self).__init__()
    
    def load_data(self, data, img_mode:str):
        # data is type 'str'
        numpy_data = np.load(data).astype(np.uint8)
        return Image.fromarray(numpy_data, mode=img_mode)  # PIL image

class NpyImageDataLoaderV2(ImageDataLoader):
    def __init__(self):
        super(NpyImageDataLoaderV2, self).__init__()
    
    def load_data(self, data, img_mode:str):
        # data is type 'byte'
        numpy_data = np.frombuffer(data, dtype=np.uint8)
        return Image.fromarray(numpy_data, mode=img_mode)  # PIL image