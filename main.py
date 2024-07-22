import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from EngineeringDataset import EngineeringDataset
from set_global_vars import set_global_vars


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pytorch training arguments')
    parser.add_argument('--global_batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--s3_bucket_name', type=str, default='')
    args = parser.parse_args()
    # set global variables
    runtime_global_vars_map = {}
    runtime_global_vars_map.update({'s3_bucket_name': , args.s3_bucket_name})
    set_global_vars(**runtime_global_vars_map)
    # load configuration
    dataset = EngineeringDataset.create_dataset()
    trainset_loader = DataLoader(
        dataset, args.global_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, 
        drop_last=True, prefetch_factor=args.prefetch_factor
    )