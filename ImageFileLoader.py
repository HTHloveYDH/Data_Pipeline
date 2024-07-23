from PIL import Image

import numpy as np


class ImageDataLoader:
    def __init__(self, filename):
        self.filename = filename

    def load_file(self):
        raise NotImplementedError(" Can not call this member function via base class 'ImageFileLoader'! ")

class NormalImageDataLoader(ImageDataLoader):
    def __init__(self, filename):
        super(NormalImageDataLoader, self).__init__(filename)
    
    def load_file(self):
        return Image.open(self.filename)  # self.filename can be either string or bytes stream

class NpyImageDataLoader(ImageDataLoader):
    def __init__(self, filename):
        super(NpyImageDataLoader, self).__init__(filename)
    
    def load_file(self):
        return np.load(self.filename).astype(np.uint8)
