from data_pipeline.dataset.GeneralDataset import GeneralDataset
from data_pipeline.dataset.RedisDataset import RedisDataset, RedisDatasetV2
from data_pipeline.dataset.S3Dataset import S3Dataset
from data_pipeline.dataset.GCSDataset import GCSDataset
from data_pipeline.dataset.LocalDataset import LocalDataset


classname_map = {
    'general': GeneralDataset, 
    'redisv1': RedisDataset,
    'redisv2': RedisDatasetV2,
    's3': S3Dataset,
    'gcs': GCSDataset,
    'local': LocalDataset
}

class DatasetFactory:
    def __init__(self):
        self.valid_classname_list = [
            'GeneralDataset', 
            'RedisDataset',
            'RedisDatasetV2',
            'S3Dataset',
            'GCSDataset',
            'LocalDataset'
        ]
        print('DatasetFactory built successfully')
    
    def create(self, dataset_type:str):
        classname = classname_map[dataset_type]
        return classname

    def create_v2(self, classname:str):
        assert classname in self.valid_classname_list
        return eval(classname)