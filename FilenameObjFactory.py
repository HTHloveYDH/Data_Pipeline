from FilenameObj import S3Filename, GCSFilename, LocalFilename
from utils import filename2loc


classname_map = {
    's3': S3Filename, 'gcs': GCSFilename, 'local_disk': LocalFilename
}

class FilenameObjFactory:
    def __init__(self):
        self.valid_classname_list = ['S3Filename', 'GCSFilename', 'LocalFilename']
        print('FilenameObjFactory built successfully')
    
    def create(self, filename:str):
        loc = filename2loc(filename)
        classname = classname_map[loc]
        return classname(filename)

    def create_v2(self, filename:str, classname:str):
        assert classname in self.valid_classname_list
        return eval(classname)(filename)
