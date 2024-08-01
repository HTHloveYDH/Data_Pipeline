import os
import json
import pickle

import torchvision.transforms as T
import redis
import numpy as np
# import cv2


def filename2loc(filename:str):
    if 's3://' in filename:
        return 's3'
    elif 'storage.googleapis.com' in filename:
        return 'gcs'
    elif 'redisv1' in filename:
        return 'redisv1'
    elif 'redisv2' in filename:
        return 'redisv2'
    else:
        return 'local'

def check_config(train_config:dict):
    pass

def load_json(filename:str):
    with open(filename, 'r') as f:
        content = json.load(f)
    return content

def load_configs():
    dataset_config = load_json(os.path.join('.', 'config', 'dataset_config.json'))
    train_config = load_json(os.path.join('.', 'config', 'train_config.json'))
    cloud_config = load_json(os.path.join('.', 'config', 'cloud_config.json'))
    dist_config = load_json(os.path.join('.', 'config', 'dist_config.json'))
    check_config(train_config)
    return dataset_config, train_config, cloud_config, dist_config    

def create_redis_keys(prefix:str, suffix='.'):
    assert prefix in ['redisv1', 'redisv2']
    redis_config_file_path = os.path.join('.', 'config', 'redis_config.json')
    with open(redis_config_file_path, 'r') as f:
        redis_config = json.load(f)
    train_config_file_path = os.path.join('.', 'config', 'train_config.json')
    with open(train_config_file_path, 'r') as f:
        train_config = json.load(f)
    db = redis.Redis(
        host=redis_config['redis_host'], port=redis_config['redis_port'], decode_responses=False
    )
    dataset_config_map = load_json(os.path.join('.', 'config', 'dataset_config.json'))
    for key in dataset_config_map.keys():
        index = 0
        for dataset_config in dataset_config_map[key]:
            with open(dataset_config['path'], 'r') as f:
                json_content = json.load(f)
            for img_info in json_content['annotations']:
                # img_data = cv2.imread(img_info['filename'])
                # img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                img_data = Image.open(data, mode=train_config['img_mode'])
                img_data = np.array(img_data)
                serialized_img_data = pickle.dumps(img_data)
                db.set(f'{prefix}_image_{index}{suffix}', serialized_img_data)  # set key - value pair
                index += 1
    db.save(os.path.join('.', 'data', 'redis', f'{prefix}.db'))

def check_dataset(dataset, save_dir:str, num:int, mode:str):
    index = 0
    for data in dataset:
        # print(type(data), data)
        print(type(data))
        image = T.functional.to_pil_image(data, mode='RGB')
        # print(type(image), image)
        print(type(image))
        image.save(os.path.join(save_dir, f'{index}.jpg'))
        index += 1
        if index >= num:
            break

def init_mp(l):
    global lock
    lock = l