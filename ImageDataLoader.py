from PIL import Image

import numpy as np


class ImageDataLoader:
    '''base class'''
    def __init__(self, data):
        self.data = data

    def load_data(self):
        raise NotImplementedError(" Can not call this member function via base class 'ImageFileLoader'! ")

class NormalImageDataLoader(ImageDataLoader):
    def __init__(self, data):
        super(NormalImageDataLoader, self).__init__(data)
    
    def load_data(self):
        return Image.open(self.data)  # self.data can be either string or bytes stream

class ArrayImageDataLoader(ImageDataLoader):
    def __init__(self, data):
        super(ArrayImageDataLoader, self).__init__(data)
    
    def load_data(self):
        return Image.fromarray(img, mode='RGB')

class NpyImageDataLoader(ImageDataLoader):
    def __init__(self, data):
        super(NpyImageDataLoader, self).__init__(data)
    
    def load_data(self):
        return np.load(self.data).astype(np.uint8)

class NpyImageDataLoaderV2(ImageDataLoader):
    def __init__(self, data):
        super(NpyImageDataLoaderV2, self).__init__(data)
    
    def load_data(self):
        return np.frombuffer(self.data, dtype=np.uint8)
