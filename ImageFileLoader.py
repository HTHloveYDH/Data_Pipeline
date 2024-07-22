from PIL import Image

import numpy as np


class ImageFileLoader:
    def __init__(self, filename:str):
        self.filename = filename

    def load_file(self):
        raise NotImplementedError(" Can not call this member function via base class 'Filename'! ")

class NormalImageFileLoader(ImageFileLoader):
    def __init__(self, filename):
        super(NormalImageFileLoader, self).__init__(filename)
    
    def load_file(self):
        return Image.open(self.filename)

class NpyImageFileLoader(ImageFileLoader):
    def __init__(self, filename):
        super(NpyImageFileLoader, self).__init__(filename)
    
    def load_file(self):
        return np.load(self.filename).astype(np.uint8)