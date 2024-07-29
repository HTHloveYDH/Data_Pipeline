import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from data_pipeline.dataset.DatasetFactory import DatasetFactory
from data_pipeline.transform.get_transform import get_custom_test_transform, get_custom_valid_transform, get_custom_train_transform
from utils import load_configs, check_dataset
import global_vars_manager
from set_global_vars import set_global_vars


def main():
    ''' __________________________________________ setup _____________________________________________ '''
    # load config
    dataset_config, train_config, cloud_config, dist_config = load_configs()
    # initialize custom configs
    custom_load_config = {}
    custom_model_config = {}
    custom_train_config = {}
    runtime_global_vars_map = {}
    # set global variables values which do not change during runtime
    global_vars_manager.init()
    runtime_global_vars_map.update(
        {'s3_bucket_name': cloud_config['s3_bucket_name'], 'img_mode': train_config['img_mode']}
    )
    set_global_vars(**runtime_global_vars_map)

    ''' ________________________________________ load dataset ________________________________________ '''
    from data_pipeline.filename_obj.FilenameObjFactory import FilenameObjFactory
    from data_pipeline.image_data_loader.ImageDataLoaderFactory import ImageDataLoaderFactoryV2
    # create a 'DatasetFactory' instance
    dataset_factory = DatasetFactory()
    # load configuration
    input_size = (train_config['input_width'], train_config['input_height'])
    # get image transform function for data loading
    train_transform = get_custom_train_transform(input_size, train_config['transform_version'])
    valid_transform = get_custom_valid_transform(input_size, train_config['transform_version'])
    test_transform = get_custom_test_transform(input_size, train_config['transform_version'])
    # set 'data object factory for creating dataset instance
    filename_obj_factory = FilenameObjFactory()
    image_data_loader_factory = ImageDataLoaderFactoryV2()
    data_obj_factory = {'general': filename_obj_factory}.get(train_config['dataset_type'], image_data_loader_factory)
    custom_load_config.update(
        {
            'random_aug_config': train_config['random_aug_config'], 
            'rescale_config': train_config['rescale_config']
        }
    )
    trainset = dataset_factory.create(train_config['dataset_type']).create_dataset(
        dataset_config['train'], train_config['is_preshuffle'], train_transform, data_obj_factory, 
        train_config['parallel_gather'], **custom_load_config
    )
    del custom_load_config['random_aug_config']
    validset = dataset_factory.create(train_config['dataset_type']).create_dataset(
        dataset_config['valid'], False, valid_transform, data_obj_factory, train_config['parallel_gather'], 
        **custom_load_config
    )
    testset = dataset_factory.create(train_config['dataset_type']).create_dataset(
        dataset_config['test'], False, test_transform, data_obj_factory, train_config['parallel_gather'], 
        **custom_load_config
    )
    trainset_loader = DataLoader(
        trainset, train_config['global_batch_size'], shuffle=True, num_workers=train_config['num_workers'], 
        pin_memory=False, drop_last=True, prefetch_factor=train_config['prefetch_factor']
    )
    validset_loader = DataLoader(
        validset, train_config['global_batch_size'], shuffle=True, num_workers=train_config['num_workers'], 
        pin_memory=False, drop_last=True, prefetch_factor=train_config['prefetch_factor']
    )
    testset_loader = DataLoader(
        testset, train_config['global_batch_size'], shuffle=True, num_workers=train_config['num_workers'], 
        pin_memory=False, drop_last=True, prefetch_factor=train_config['prefetch_factor']
    )

    ''' ______________________ check dataset output and generate annotation file _____________________ '''
    if train_config['training_type'] == 'check_dataset_output':
        check_dataset(testset, os.path.join('.', 'check_dataset_output'), 200, train_config['img_mode'])
        print('------------------- check dataset output completed -------------------')
        exit()
    if train_config['training_type'] == 'generate_annotation_file_only':
        print('----------------- generate annotation file completed -----------------')
        exit()

    ''' ____________________________________ build & compile model ___________________________________ '''
    # custom_model_config.update()
    
    ''' ____________________________________________ train ___________________________________________ '''
    # custom_train_config.update()
    # for epoch in range(1, train_config['epochs'] + 1):
    #     train_on_epoch(trainset_loader)
    #     valid_on_epoch(validset_loader)
        
    ''' __________________________________________ test ______________________________________________ '''


if __name__ == '__main__':
    main()
