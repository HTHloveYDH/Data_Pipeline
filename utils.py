import os
import json
import pickle

import redis
import cv2


def filename2loc(filename:str):
    if 's3://' in filename:
        return 's3'
    elif 'storage.googleapis.com' in filename:
        return 'gcs'
    elif 'redis' in filename:
        return 'redis'
    else:
        return 'local_disk'

def load_json(filename:str):
    with open(os.path.join('.', f'{filename}.json'), 'r') as f:
        content = json.load(f)
    return content

def load_configs():
    dataset_config = load_json(os.path.join('.', 'dataset_config.json'))
    training_config = load_json(os.path.join('.', 'training_config.json'))
    dist_config = load_json(os.path.join('.', 'dist_config.json'))
    return dataset_config, training_config

def create_redis_keys():
    redis_config_file_path = os.path.join('.', 'redis_config.json')
    with open(redis_config_file_path, 'r') as f:
        redis_config = json.load(f)
    db = redis.Redis(
        host=redis_config['redis_host'], port=redis_config['redis_port'], decode_responses=False
    )  
    dataset_config_map = load_json(os.path.join('.', 'dataset_config.json'))
    for key in dataset_config_map.keys():
        index = 0
        for dataset_config in dataset_config_map[key]:
            with open(dataset_config['path'], 'r') as f:
                json_content = json.load(f)
            for img_info in json_content['annotations']:
                img_data = cv2.imread(img_info['filename'])
                img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                serialized_img_data = pickle.dumps(img_data)
                db.set(f'redis_image_{index}', serialized_img_data)  # set key - value pair
                index += 1
