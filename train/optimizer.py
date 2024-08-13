import os

import torch
import torch.optim as optim


def select_optimizer(optimizer_type:str, learning_rate:float, model, **kwargs):
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(
            model if isinstance(model, list) else model.parameters(), lr=learning_rate, weight_decay=0.01
        )
    elif optimizer_type == 'AdamW':
        optimizer = optim.AdamW(
            model if isinstance(model, list) else model.parameters(), lr=learning_rate, weight_decay=0.01
        )
    elif optimizer_type == 'Adamx':
        optimizer = optim.Adamx(
            model if isinstance(model, list) else model.parameters(), lr=learning_rate, weight_decay=0.01
        )
    elif optimizer_type == 'NAdam':
        optimizer = optim.NAdam(
            model if isinstance(model, list) else model.parameters(), lr=learning_rate, weight_decay=0.01
        )
    elif optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(
            model if isinstance(model, list) else model.parameters(), lr=learning_rate, momentum=0.0, 
            weight_decay=0.01
        )
    elif optimizer_type == 'Adadelta':
        optimizer = optim.Adadelta(
            model if isinstance(model, list) else model.parameters(), lr=learning_rate, weight_decay=0.01
        )
    elif optimizer_type == 'Adagrad':
        optimizer = optim.Adagrad(
            model if isinstance(model, list) else model.parameters(), lr=learning_rate, weight_decay=0.01
        )
    elif optimizer_type == 'RAdam':
        optimizer = optim.RAdam(
            model if isinstance(model, list) else model.parameters(), lr=learning_rate, weight_decay=0.01
        )
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(
            model if isinstance(model, list) else model.parameters(), lr=learning_rate, momentum=0.0, 
            weight_decay=0.01
        )
    else:
        raise ValueError(f" {optimizer_type} is not supported ")
    if kwargs['load_optimizer_dir'] is not None:
        optimizer_state_dict = torch.load(os.path.join(kwargs['load_optimizer_dir'], 'optimizer.pth'))
        optimizer.load_state_dict(optimizer_state_dict)
    return optimizer

def select_lr_scheduler(lr_scheduler_type:str, optimizer, **kwargs):
    if lr_scheduler_type == 'cosine_decay':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-9)
    elif lr_scheduler_type == 'linear':
        lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=5)
    elif lr_scheduler_type == 'constant':
        lr_scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=0.3333333333333333, total_iters=5)
    elif lr_scheduler_type == 'exp_decay':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    elif lr_scheduler_type == 'cosine_restart_decay':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=0)
    elif lr_scheduler_type == 'poly_decay':
        lr_scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=4, power=1.0)
    else:
        raise ValueError(f" {lr_scheduler_type} is not supported ")
    return lr_scheduler

def set_optimizer_lr_scheduler(optimizer_type:str, learning_rates:list, lr_scheduler_type:str, model, **kwargs):
    # multi-optimizer
    if len(learning_rates) == 3:
        model_segs = [{'params': model.parameters()}]
        optimizer = select_optimizer(optimizer_type, learning_rates[0], model_segs, **kwargs)
    # single-optimizer
    elif len(learning_rates) == 1:
        optimizer = select_optimizer(optimizer_type, learning_rates[0], model, **kwargs)
    else:
        raise ValueError(f' the number of learning rate: {len(learning_rates)} is invalid! ')
    lr_scheduler = select_lr_scheduler(lr_scheduler_type, optimizer)
    return optimizer, lr_scheduler