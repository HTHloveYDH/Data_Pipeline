import os
from io import BytesIO, StringIO
import pickle

import boto3
from google.cloud import storage
import redis  # pip install redis

from data_pipeline.image_data_loader.ImageDataLoaderFactory import ImageDataLoaderFactory
import global_vars_manager


image_data_loader_factory = ImageDataLoaderFactory(global_vars_manager.get_global_var('IMG_MODE'))  # singleton

class Filename:
    '''base class'''
    def __init__(self, filename:str):
        self.filename = filename 
        self.img_data_loader = image_data_loader_factory.create(filename)

    def load(self):
        raise NotImplementedError(" Can not call this member function via base class 'Filename'! ")

class LocalFilename(Filename):
    def __init__(self, filename):
        super(LocalFilename, self).__init__(filename)
    
    def load(self):
        image = self.img_data_loader.load_data()  # PIL Image, in 'RGB' order or npy file
        return image

class S3Filename:
    s3 = boto3.client(
        's3', aws_access_key_id=global_vars_manager.get_global_var('AWS_ACCESS_KEY_ID'), 
        aws_secret_access_key=global_vars_manager.get_global_var('AWS_SECRET_ACCESS_KEY'), 
        region_name=global_vars_manager.get_global_var('REGION_NAME')
    )

    def __init__(self, filename:str):
        super(S3Filename, self).__init__(filename)
        self.s3_bucket_name = global_vars_manager.get_global_var('S3_BUCKET_NAME')
    
    def load(self):
        image_byte_string = S3Filename.s3.get_object(
            Bucket=self.s3_bucket_name, Key=self.filename
        )['Body'].read()  # 
        self.img_data_loader.data = BytesIO(image_byte_string)  # bytes stream
        image = self.img_data_loader.load_data()  # PIL Image, in 'RGB' order or npy file
        return image

class GCSFilename:
    bucket = storage.Client().get_bucket(global_vars_manager.get_global_var('GCS_BUCKET_NAME'))

    def __init__(self, filename:str):
        super(GCSFilename, self).__init__(filename)
    
    def load(self):
        blob = GCSFilename.bucket.blob(self.filename)
        blob = blob.download_as_string()  # 
        # blob = blob.decode('utf-8')
        self.img_data_loader.data = BytesIO(blob)  # bytes stream
        image = self.img_data_loader.load_data()  # PIL Image, in 'RGB' order or npy file
        return image

class RedisFilename(Filename):
    redis = redis.Redis(
        host=global_vars_manager.get_global_var('REDIS_HOST'), 
        port=global_vars_manager.get_global_var('REDIS_PORT'),  # 6379
        decode_responses=False,  # return bytes stream 
        db=0
    )  # self.redis is a in-memory database

    def __init__(self, filename):
        super(RedisFilename, self).__init__(filename)
    
    def load(self):
        self.img_data_loader.data = RedisFilename.redis.get(self.filename)  # bytes stream
        image = self.img_data_loader.load_data()
        return image

class RedisFilenameV2(Filename):
    redis = redis.Redis(
        host=global_vars_manager.get_global_var('REDIS_HOST'), 
        port=global_vars_manager.get_global_var('REDIS_PORT'),  # 6379
        decode_responses=False,  # return bytes stream 
        db=0
    )  # self.redis is a in-memory database
    
    def __init__(self, filename):
        super(RedisFilename, self).__init__(filename)
    
    def load(self):
        retrieved_data = RedisFilenameV2.redis.get(self.filename)  # bytes stream
        self.img_data_loader.data = pickle.loads(retrieved_data)  # string
        image = self.img_data_loader.load_data()
        return image