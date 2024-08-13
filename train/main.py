import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

sys.path.append(os.getcwd())
from dist.distribute import init_dist, ternimate_dist
from data_pipeline.dataset.DatasetFactory import DatasetFactory
from data_pipeline.transform.get_transform import get_custom_test_transform, get_custom_valid_transform, \
    get_custom_train_transform
from models.get_model import get_model
from train.train_funcs import train_on_epoch_hp, train_on_epoch_amp, valid_on_epoch, resume_from_ckpt
from train.optimizer import set_optimizer_lr_scheduler
from utils.model_utils import init_qat
from utils.data_pipeline_utils import load_configs, check_dataset
from utils.logger import Logger
import global_vars_manager
from set_global_vars import set_global_vars


def main(dp_local_rank=0, dp_world_size=1, torch_mp_launch=False):
    ''' __________________________________________ setup _____________________________________________ '''
    # load config
    dataset_config, train_config, cloud_config, dist_config = load_configs()
    save_model_dir = train_config['save_model_dir']
    load_optimizer_dir = train_config['load_optimizer_dir']
    # initialize custom configs
    custom_load_config = {}
    custom_model_config = {}
    custom_optimizer_lr_scheduler_config = {}
    custom_train_config = {}
    runtime_global_vars_map = {}
    # save log file
    sys.stdout = Logger(os.path.join(save_model_dir, f'train.log'), sys.stdout)
    # setup distribute process group
    dp = dist_config['dist_strategy'] in ['ddp', 'fsdp']
    dp_global_rank, dp_local_rank, dp_world_size, master_process, device, device_type = init_dist(
        dist_config['dist_strategy'], torch_mp_launch, dp_local_rank, dp_world_size
    )
    # qat setup
    init_qat(train_config['model_type'])
    # set global variables values which do not change during runtime
    global_vars_manager.init()
    runtime_global_vars_map.update(
        {'s3_bucket_name': cloud_config['s3_bucket_name'], 'img_mode': train_config['img_mode']}
    )
    set_global_vars(**runtime_global_vars_map)

    ''' ________________________________________ load dataset ________________________________________ '''
    from data_pipeline.image_filename_obj.ImageFilenameObjFactory import ImageFilenameObjFactory
    from data_pipeline.image_data_loader.ImageDataLoaderFactory import ImageDataLoaderFactoryV2
    # create a 'DatasetFactory' instance
    dataset_factory = DatasetFactory()
    # load configuration
    input_size = (train_config['input_width'], train_config['input_height'])
    # shuffle configuration
    is_preshuffle = shuffle = False if dist_config['dist_strategy'] in ['ddp', 'fsdp'] else True
    # get image transform function for data loading
    train_transform = get_custom_train_transform(input_size, train_config['transform_version'])
    valid_transform = get_custom_valid_transform(input_size, train_config['transform_version'])
    test_transform = get_custom_test_transform(input_size, train_config['transform_version'])
    # set 'data object factory for creating dataset instance
    image_filename_obj_factory = ImageFilenameObjFactory()
    image_data_loader_factory = ImageDataLoaderFactoryV2()
    image_data_obj_factory = {'general': image_filename_obj_factory}.get(train_config['dataset_type'], image_data_loader_factory)
    custom_load_config.update(
        {
            'random_aug_config': train_config['random_aug_config'], 
            'rescale_config': train_config['rescale_config']
        }
    )
    trainset = dataset_factory.create(train_config['dataset_type']).create_dataset(
        dataset_config['train'], is_preshuffle, train_transform, image_data_obj_factory, 
        train_config['parallel_gather'], **custom_load_config
    )
    del custom_load_config['random_aug_config']
    validset = dataset_factory.create(train_config['dataset_type']).create_dataset(
        dataset_config['valid'], False, valid_transform, image_data_obj_factory, 
        train_config['parallel_gather'], **custom_load_config
    )
    testset = dataset_factory.create(train_config['dataset_type']).create_dataset(
        dataset_config['test'], False, test_transform, image_data_obj_factory, 
        train_config['parallel_gather'], **custom_load_config
    )
    # set dataset sampler distribute traning
    trainset_sampler = None
    validset_sampler = None
    testset_sampler = None
    if dist_config['dist_strategy'] in ['ddp', 'fsdp']:
        trainset_sampler = DistributedSampler(trainset)
        validset_sampler = DistributedSampler(validset)
        testset_sampler = DistributedSampler(testset)
    trainset_loader = DataLoader(
        trainset, train_config['batch_size'], shuffle=shuffle, sampler=trainset_sampler, 
        num_workers=train_config['num_workers'], 
        pin_memory=False, drop_last=True, prefetch_factor=train_config['prefetch_factor']
    )
    validset_loader = DataLoader(
        validset, train_config['batch_size'], shuffle=shuffle, sampler=validset_sampler, 
        num_workers=train_config['num_workers'], 
        pin_memory=False, drop_last=True, prefetch_factor=train_config['prefetch_factor']
    )
    testset_loader = DataLoader(
        testset, train_config['batch_size'], shuffle=shuffle, sampler=testset_sampler, 
        num_workers=train_config['num_workers'], 
        pin_memory=False, drop_last=True, prefetch_factor=train_config['prefetch_factor']
    )

    ''' ______________________ check dataset output and generate annotation file _____________________ '''
    if train_config['training_type'] == 'check_dataset_output':
        check_dataset(trainset, os.path.join('.', 'check_dataset_output'), 200, train_config['img_mode'])
        print('------------------- check dataset output completed -------------------')
        exit()
    if train_config['training_type'] == 'generate_annotation_file_only':
        print('----------------- generate annotation file completed -----------------')
        exit()

    ''' ____________________________________ build & compile model ___________________________________ '''
    # get model
    custom_model_config.update({})
    device_ids = [dp_local_rank]
    model = get_model(
        dist_config['dist_strategy'], device, device_ids, dp_local_rank, **custom_model_config)
    # 
    custom_optimizer_lr_scheduler_config.update({'load_optimizer_dir': load_optimizer_dir})
    optimizer, learning_rate_scheduler = set_optimizer_lr_scheduler(
        train_config['optimizer_type'], train_config['initial_learning_rates'], 
        train_config['learning_rate_scheduler_type'], model, **custom_optimizer_lr_scheduler_config
    )
    
    ''' ____________________________________________ train ___________________________________________ '''
    train_losses = []
    train_counter = []
    valid_losses = []
    valid_counter = []
    min_val_loss = float("inf")  # min_val_loss is set to be positive infinity at the begining of training
    consecutive_epochs_without_improvement = 0
    max_consecutive_epochs_without_improvement = 50
    # whether to enable 'AUTOMATIC MIXED PRECISION' mode or not
    if train_config['is_mix_precision']:
        # Creates a GradScaler once at the beginning of training.
        custom_train_config['grad_scaler'] = torch.cuda.amp.GradScaler()
        custom_train_config['amp_dtype'] = {'cuda': torch.float16, 'cpu': torch.bfloat16}[device.type]
    train_on_epoch = {False: train_on_epoch_hp, True: train_on_epoch_amp}[train_config['is_mix_precision']]
    model, optimizer = resume_from_ckpt(model, optimizer, save_model_dir)
    for epoch in range(1, train_config['epochs'] + 1):  
        train_on_epoch(
            model, trainset_loader, optimizer, learning_rate_scheduler, train_config['batch_size'], 
            train_config['accum_steps'], train_losses, train_counter, save_model_dir, 
            train_config['log_interval'], epoch, train_config['epochs'], dp_global_rank, device, dp, 
            **custom_train_config
        )
        valid_on_epoch(
            model, validset_loader, valid_losses, valid_counter, len(trainset_loader), epoch, 
            dp_global_rank, device, dp
        )
        if valid_losses[-1] < min_val_loss:
            print(f'val loss improved from {min_val_loss} to {valid_losses[-1]}, saving model to {save_model_dir}')
            min_val_loss = valid_losses[-1]
            consecutive_epochs_without_improvement = 0
            # save best model and the corresponding optimizer at the end of every epoch (only in process #0)
            torch.save(model.state_dict(), f'{save_model_dir}/model.pth')
            torch.save(optimizer.state_dict(), f'{save_model_dir}/optimizer.pth')
        else:
            print(f'val loss did not improved from {min_val_loss}, go to next epoch')
            consecutive_epochs_without_improvement += 1
            if consecutive_epochs_without_improvement > max_consecutive_epochs_without_improvement:
                print('after {epoch} epochs, early stop activated')
                break
    # destroy current process group
    ternimate_dist(dist_config['dist_strategy'])
        
    ''' __________________________________________ test ______________________________________________ '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--dp_world_size', type=int, help='Total processes to train the model')
    parser.add_argument('--torch_mp_launch', action='store_true')
    args = parser.parse_args()
    # launch by torch.multiprocessing
    if args.torch_mp_launch:
        mp.spawn(main, args=(args.dp_world_size, args.torch_mp_launch), nprocs=args.dp_world_size)
    # launch by torchrun or python
    else:
        main()