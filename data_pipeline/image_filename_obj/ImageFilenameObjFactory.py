from data_pipeline.image_filename_obj.ImageFilenameObj import S3ImageFilename, LocalImageFilename, RedisImageFilename, \
    RedisImageFilenameV2#, GCSImageFilename
from utils.data_pipeline_utils import filename2loc


classname_map = {
    's3': S3ImageFilename, 
    # 'gcs': GCSFilename, 
    'local': LocalImageFilename, 
    'redisv1': RedisImageFilename,
    'redisv2': RedisImageFilenameV2
}

class ImageFilenameObjFactory:
    def __init__(self):
        self.valid_classname_list = [
            'S3ImageFilename', 
            # 'GCSImageFilename', 
            'LocalImageFilename', 
            'RedisImageFilename', 
            'RedisImageFilenameV2'
        ]
        print('FilenameObjFactory built successfully')
    
    def create(self, filename:str):
        loc = filename2loc(filename)
        classname = classname_map[loc]
        return classname(filename)

    def create_v2(self, filename:str, classname:str):
        assert classname in self.valid_classname_list
        return eval(classname)(filename)