from FilenameObjs import S3Filename, GCSFilename, LocalFilename


classname_map = {
    's3': S3Filename, 'gcs': GCSFilename, 'local_disk': LocalFilename
}

class FilenameObjFactory:
    def __init__(self):
        self.valid_cloud_storage_list = ['S3Filename', 'GCSFilename', 'LocalFilename']
        print('FilenameObjFactory built successfully')
    
    def create(self, filename:str):
        loc = self.filename2loc(filename)
        classname = classname_map[loc]
        return classname(filename)

    def create_v2(self, filename:str, classname:str):
        assert classname in self.valid_cloud_storage_list
        return eval(classname)(filename)

    def filename2loc(self, filename:str):
        if 's3://' in filename:
            return 's3'
        elif 'storage.googleapis.com' in filename:
            return 'gcs'
        else:
            return 'local_disk'
