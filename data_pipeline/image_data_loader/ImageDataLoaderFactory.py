import os

from data_pipeline.image_data_loader.ImageDataLoader import NormalImageDataLoader, NpyImageDataLoader, \
    NpyImageDataLoaderV2, ArrayImageDataLoader
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
    'local': {
        'npy': NpyImageDataLoader, 
        'jpeg': NormalImageDataLoader, 
        'jpg': NormalImageDataLoader, 
        'png': NormalImageDataLoader,
        'bmp': NormalImageDataLoader,
    },
    'redisv1': {
        '': NormalImageDataLoader
    },
    'redisv2': {
        '': ArrayImageDataLoader
    }
}

class ImageDataLoaderFactory:
    def __init__(self):
        self.valid_classname_list = [
            'NormalImageDataLoader', 
            'NpyImageDataLoader', 
            'NpyImageDataLoaderV2',
            'ArrayImageDataLoader'
        ]
        print('ImageDataLoaderFactory built successfully')
    
    def create(self, filename:str):
        loc = filename2loc(filename)
        path_delimiter = {'posix': '/', 'nt': '\\'}[os.name]
        suffix = filename.split(path_delimiter)[-1].split('.')[-1]
        classname = classname_map[loc][suffix]
        return classname()

    def create_v2(self, classname:str):
        assert classname in self.valid_classname_list
        return eval(classname)()
    
class ImageDataLoaderFactoryV2:
    def __init__(self):
        self.valid_classname_list = [
            'NormalImageDataLoader', 
            'NpyImageDataLoader', 
            'NpyImageDataLoaderV2',
            'ArrayImageDataLoader'
        ]
        print('ImageDataLoaderFactoryV2 built successfully')
    
    def create(self, filename:str):
        loc = filename2loc(filename)
        path_delimiter = {'posix': '/', 'nt': '\\'}[os.name]
        suffix = filename.split(path_delimiter)[-1].split('.')[-1]
        classname = classname_map[loc][suffix]
        return classname(filename)

    def create_v2(self, filename:str, classname:str):
        assert classname in self.valid_classname_list
        return eval(classname)(filename)