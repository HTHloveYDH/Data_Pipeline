from EngineeringDataset import EngineeringDataset
from RedisEngineeringDataset import RedisEngineeringDataset


classname_map = {
    'normal': EngineeringDataset, 
    'redis': RedisEngineeringDataset
}

class DatasetFactory:
    def __init__(self):
        self.valid_classname_list = [
            'EngineeringDataset', 
            'RedisEngineeringDataset'
        ]
    
    def create(self, dataset_type:str):
        classname = classname_map[dataset_type]
        return classname

    def create_v2(self, classname:str):
        assert classname in self.valid_classname_list
        return eval(classname)