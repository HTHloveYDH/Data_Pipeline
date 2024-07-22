from FilenameObjs import S3Filename, GCSFilename, LocalFilename


classname_map = {
    's3': S3Filename, 'gcs': GCSFilename, 'local_disk': LocalFilename
}

class FilenameObjFactory:
    def __init__(self, cloud_storage:str):
        print('FilenameObjFactory built successfully')
    
    def create(self, filename:str):
        loc = self.filename2loc(filename)
        classname = classname_map[loc]
        return eval(classname)(filename)

    def filename2loc(self, filename:str):
        return loc