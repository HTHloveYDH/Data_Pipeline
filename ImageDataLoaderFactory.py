import os

from ImageDataLoader import NormalImageDataLoader, NpyImageDataLoader, NpyImageDataLoaderV2, \
    ArrayImageDataLoader
from utils import filename2loc


classname_map = {
    's3': {
        'npy': NpyImageDataLoaderV2, 
        'jpeg': NormalImageDataLoader, 
        'jpg': NormalImageDataLoader, 
        'png': NormalImageDataLoader,
        'bmp': NormalImageDataLoader,
    }, 
    'gcs': {
        'npy': NpyImageDataLoaderV2, 
        'jpeg': NormalImageDataLoader, 
        'jpg': NormalImageDataLoader, 
        'png': NormalImageDataLoader,
        'bmp': NormalImageDataLoader,
    }, 
    'local_disk': {
        'npy': NpyImageDataLoader, 
        'jpeg': NormalImageDataLoader, 
        'jpg': NormalImageDataLoader, 
        'png': NormalImageDataLoader,
        'bmp': NormalImageDataLoader,
    }
    'redis' {
        '': NormalImageDataLoader
    },
    'redisv2' {
        '': ArrayImageDataLoader
    }
}

class ImageDataLoaderFactory:
    def __init__(self, img_mode:str):
        self.valid_classname_list = [
            'NormalImageDataLoader', 
            'NpyImageDataLoader', 
            'NpyImageDataLoaderV2',
            'ArrayImageDataLoader'
        ]
        self.img_mode = img_mode
    
    def create(self, filename:str):
        loc = filename2loc(filename)
        path_delimiter = {'posix': '/', 'nt': '\\'}[os.name]
        suffix = filename.split(path_delimiter)[-1].split('.')[-1]
        classname = classname_map[loc][suffix]
        return classname(filename, self.img_mode)

    def create_v2(self, filename:str, classname:str):
        assert classname in self.valid_classname_list
        return eval(classname)(filename, self.img_mode)
