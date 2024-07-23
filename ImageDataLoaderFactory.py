import os

from ImageFileLoader import NormalImageFileLoader, NpyImageFileLoader, NpyImageFileLoaderV2
from utils import filename2loc


classname_map = {
    's3': {
        'npy': NpyImageFileLoaderV2, 
        'jpeg': NormalImageFileLoader, 
        'jpg': NormalImageFileLoader, 
        'png': NormalImageFileLoader
    }, 
    'gcs': {
        'npy': NpyImageFileLoaderV2, 
        'jpeg': NormalImageFileLoader, 
        'jpg': NormalImageFileLoader, 
        'png': NormalImageFileLoader
    }, 
    'local_disk': {
        'npy': NpyImageFileLoader, 
        'jpeg': NormalImageFileLoader, 
        'jpg': NormalImageFileLoader, 
        'png': NormalImageFileLoader
    }
}

class ImageDataLoaderFactory:
    def __init__(self):
        self.valid_classname_list = [
            'NormalImageFileLoader', 
            'NpyImageFileLoader', 
            'NpyImageFileLoaderV2'
        ]
    
    def create(self, filename:str):
        loc = filename2loc(filename)
        path_delimiter = {'posix': '/', 'nt': '\\'}[os.name]
        suffix = filename.split(path_delimiter)[-1].split('.')
        classname = classname_map[loc][suffix]
        return classname(filename)

    def create_v2(self, filename:str, classname:str):
        assert classname in self.valid_classname_list
        return eval(classname)(filename)