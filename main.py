import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from DatasetFactory import DatasetFactory
from get_transform import get_custom_test_transform, get_custom_valid_transform, get_custom_train_transform
from utils import load_configs
from set_global_vars import set_global_vars


def main():
    ''' __________________________________________ setup _____________________________________________ '''
    parser = argparse.ArgumentParser('Pytorch training arguments')
    parser.add_argument('--global_batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--s3_bucket_name', type=str, default='')
    args = parser.parse_args()
    # load config
    dataset_config, training_config = load_configs()
    # initialize custom configs
    custom_load_config = {}
    custom_model_config = {}
    custom_train_config = {}
    # set global variables
    runtime_global_vars_map = {}
    runtime_global_vars_map.update({'s3_bucket_name': args.s3_bucket_name})
    set_global_vars(**runtime_global_vars_map)

    ''' ________________________________________ load dataset ________________________________________ '''
    # create a 'DatasetFactory' instance
    dataset_factory = DatasetFactory()
    # load configuration
    input_size = (training_config['input_width'], training_config['input_height'])
    default_img_size = (training_config['default_img_width'], training_config['default_img_height'])
    # get image transformation for data loading
    train_transform = get_custom_train_transform(input_size, training_config['transform_version'])
    valid_transform = get_custom_valid_transform(input_size, training_config['transform_version'])
    test_transform = get_custom_test_transform(input_size, training_config['transform_version'])
    custom_load_config.update({'random_aug_config': config['random_aug_config']})
    trainset = dataset_factory.create(training_config['dataset_type']).create_dataset(
        dataset_config['train'], input_size, default_img_size, is_preshuffle, **custom_load_config
    )
    trainset_loader = DataLoader(
        trainset, args.global_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, 
        drop_last=True, prefetch_factor=args.prefetch_factor
    )
    validset = dataset_factory.create(training_config['dataset_type']).create_dataset(
        dataset_config['valid'], input_size, default_img_size, False
    )
    validset_loader = DataLoader(
        validset, args.global_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, 
        drop_last=True, prefetch_factor=args.prefetch_factor
    )
    testset = dataset_factory.create(training_config['dataset_type']).create_dataset(
        dataset_config['test'], input_size, default_img_size, False
    )
    testset_loader = DataLoader(
        testset, args.global_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, 
        drop_last=True, prefetch_factor=args.prefetch_factor
    )

    ''' ____________________________________ build & compile model ___________________________________ '''
    # custom_model_config.update()
    
    ''' ____________________________________________ train ___________________________________________ '''
    # custom_train_config.update()
    # for epoch in range(1, training_config['epochs'] + 1):
    #     train_on_epoch()
    #     valid_on_epoch()
        
    ''' __________________________________________ test ______________________________________________ '''


if __name__ == '__main__':
    main()
