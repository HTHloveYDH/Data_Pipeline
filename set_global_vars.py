import os
import json

from utils import load_json
import global_vars_manager

def cloud_storage_init():
    aws_config_file_path = os.path.expanduser('~/.aws/config')
    aws_credentials_file_path = os.path.expanduser('~/.aws/credentials')
    config_file_path = '~/.aws/config'
    with open(aws_config_file_path, 'r') as file:
        # config_data = file.read()
        config_data = file.readlines()
    _, region = config_data[1].strip().split(' = ')
    with open(aws_credentials_file_path, 'r') as file:
        # credentials_data = file.read()
        credentials_data = file.readlines()
    _, aws_access_key_id = credentials_data[1].strip().split(' = ')
    _, aws_secret_access_key = credentials_data[2].strip().split(' = ')
    global_vars_manager.set_global_var('REGION_NAME', region)
    global_vars_manager.set_global_var('AWS_ACCESS_KEY_ID', aws_access_key_id)
    global_vars_manager.set_global_var('AWS_SECRET_ACCESS_KEY', aws_secret_access_key)

    cloud_config = load_json(os.path.join('.', 'config', 'cloud_config.json'))
    global_vars_manager.set_global_var('S3_BUCKET_NAME', cloud_config['s3_bucket_name'])
    global_vars_manager.set_global_var('GCS_BUCKET_NAME', cloud_config['gcs_bucket_name'])

def redis_init():
    redis_config_file_path = os.path.join('.', 'config', 'redis_config.json')
    with open(redis_config_file_path, 'r') as f:
        redis_config = json.load(f)
    global_vars_manager.set_global_var('REDIS_HOST', redis_config['redis_host'])
    global_vars_manager.set_global_var('REDIS_PORT', redis_config['redis_port'])
    global_vars_manager.set_global_var('REDIS_DB', redis_config['redis_db'])
    global_vars_manager.set_global_var('REDIS_PASSWORD', redis_config['redis_password'])
    global_vars_manager.set_global_var('REDIS_RETRY_ON_TIMEOUT', redis_config['redis_retry_on_timeout'])
    global_vars_manager.set_global_var('REDIS_USERNAME', redis_config['redis_username'])

def set_global_vars(**kwargs):
    cloud_storage_init()
    redis_init()
    global_vars_manager.set_global_var('IMG_MODE', kwargs['img_mode'])