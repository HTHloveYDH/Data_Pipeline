import os
import json

import global_vars_manager

def cloud_storage_init():
    aws_config_file_path = os.path.expanduser('~/.aws/config')
    aws_credentials_file_path = os.path.expanduser('~/.aws/credentials')
    config_file_path = '~/.aws/config'
    with open(aws_config_file_path, 'r') as file:
        config_data = file.read()
    config = json.loads(config_data)
    with open(aws_credentials_file_path, 'r') as file:
        credentials_data = file.read()
    credentials = json.loads(credentials_data)
    global_vars_manager.set_global_var('REGION_NAME', config['region'])
    global_vars_manager.set_global_var('AWS_ACCESS_KEY_ID', credentials['aws_access_key_id'])
    global_vars_manager.set_global_var('AWS_SECRET_ACCESS_KEY', credentials['aws_secret_access_key'])
    global_vars_manager.set_global_var('S3_BUCKET_NAME', os.environ['S3_BUCKET_NAME'])
    global_vars_manager.set_global_var('GCS_BUCKET_NAME', '')

def set_global_vars(**kwargs):
    cloud_storage_init()
    global_vars_manager.set_global_var('img_mode', kwargs['img_mode'])
