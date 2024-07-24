from io import BytesIO

import boto3

from data_pipeline.dataset.BaseDataset import BaseDataset
import global_vars_manager


class S3Dataset(BaseDataset):
    'Characterizes a s3 dataset for PyTorch'
    def __init__(self, data_objs:list, transform, **kwargs):
        super(S3Dataset, self).__init__(data_objs, transform, **kwargs)
        self.aws_access_key_id = global_vars_manager.get_global_var('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = global_vars_manager.get_global_var('AWS_SECRET_ACCESS_KEY')
        self.region_name = global_vars_manager.get_global_var('REGION_NAME')
        self.s3 = boto3.client(
            's3', aws_access_key_id=self.aws_access_key_id, 
            aws_secret_access_key=self.aws_secret_access_key, 
            region_name=self.region_name
        )
        self.s3_bucket_name = global_vars_manager.get_global_var('S3_BUCKET_NAME')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # load image
        img_data_loader = self.data_objs[index]
        image_byte_string = self.s3.get_object(Bucket=self.s3_bucket_name, Key=img_data_loader.data)['Body'].read()  # 
        img_data_loader.data = BytesIO(image_byte_string)
        image = img_data_loader.load_data()
        # image augmentation and rescale
        image = self.transform(image, **{'random_aug_config': self.random_aug_config})
        return image