import os
import json


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
