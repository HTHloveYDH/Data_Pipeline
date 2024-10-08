from io import BytesIO

import boto3

from data_pipeline.dataset.BaseDataset import BaseDataset
import global_vars_manager


class S3Dataset(BaseDataset):
    'Characterizes a s3 dataset for PyTorch'
    def __init__(self, data_objs_list:list, transform, **kwargs):
        super(S3Dataset, self).__init__(data_objs_list, transform, **kwargs)
        self.aws_access_key_id = global_vars_manager.get_global_var('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = global_vars_manager.get_global_var('AWS_SECRET_ACCESS_KEY')
        self.region_name = global_vars_manager.get_global_var('REGION_NAME')
        self.s3 = boto3.client(
            's3', aws_access_key_id=self.aws_access_key_id, 
            aws_secret_access_key=self.aws_secret_access_key, 
            region_name=self.region_name
        )
        self.s3_bucket_name = global_vars_manager.get_global_var('S3_BUCKET_NAME')
        self.start_idx = len(self.s3_bucket_name) + 6  # 6 = len('s3://') + len('/')

    def __getitem__(self, index):
        'Generates one sample of data'
        # load image
        img_data_loader = self.data_objs_list[0][index]
        image_bytes = self.s3.get_object(
            Bucket=self.s3_bucket_name, Key=img_data_loader.key[self.start_idx:]
        )['Body'].read()  # type: bytes
        assert isinstance(image_bytes, bytes)
        data = BytesIO(image_bytes)  # type: BytesIO
        image = img_data_loader.load_data(data)
        # image augmentation and rescale
        image = self.transform(image, **self.custom_load_config)
        return image