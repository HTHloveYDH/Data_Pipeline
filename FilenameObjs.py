import os
from io import BytesIO

import boto3

from ImageFileLoader import NormalImageFileLoader, NpyImageFileLoader


def create_img_file_loader(filename:str):
    path_delimiter = {'posix': '/', 'nt': '\\'}[os.name]
    suffix = filename.split(path_delimiter)[-1].split('.')
    if suffix in ['jpeg', 'jpg', 'png', 'bmp']:
        return NormalImageFileLoader(filename)
    elif suffix == 'npy':
        return NpyImageFileLoader(filename)
    else:
        raise ValueError(f'.{suffix} is not supported')

class Filename:
    def __init__(self, filename:str):
        self.filename = filename 
        self.img_file_loader = create_img_file_loader(filename)

    def load(self):
        raise NotImplementedError(" Can not call this member function via base class 'Filename'! ")

class LocalFilename(Filename):
    def __init__(self, filename):
        super(LocalFilename, self).__init__(filename)
    
    def load(self):
        image = self.img_file_loader.load_file()  # PIL Image, in 'RGB' order or npy file
        return image

class S3Filename:
    def __init__(self, filename:str):
        super(S3Filename, self).__init__(filename)
        self.s3 = boto3.client(
            's3', 
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
            region_name=os.environ['REGION_NAME']
        )
        self.s3_bucket_name = os.environ['S3_BUCKET_NAME']
    
    def load(self):
        image_byte_string = self.s3.get_object(
            Bucket=self.s3_bucket_name, Key=self.filename
        )['Body'].read()
        image = self.img_file_loader.load_file()  # PIL Image, in 'RGB' order or npy file
        return image

class GCSFilename:
    def __init__(self, filename:str):
        super(GCSFilename, self).__init__(filename)
        self.gcs_bucket_name = os.environ['GCS_BUCKET_NAME']
    
    def load(self):
        image = self.img_file_loader.load_file()  # PIL Image, in 'RGB' order or npy file
        return image
