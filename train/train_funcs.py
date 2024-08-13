import os
from datetime import datetime

import torch
# from torchvision.transforms import v2
import torch.distributed as dist
from tqdm import tqdm

from train.epoch_log import train_batch_log, valid_epoch_log


def train_on_epoch_hp(model, trainset_loader, optimizer, learning_rate_scheduler, batch_size:int, \
                      accum_steps:int, train_losses:list, train_counter:list, save_model_dir:str, \
                      log_interval:int, epoch:int, epochs:int, dp_global_rank:int, device, dp:bool, \
                      **kwargs):
    print(f'Epoch: {epoch}: ')
    pbar = tqdm(enumerate(trainset_loader), total=len(trainset_loader), leave=True, ncols=None)
    pbar.set_description(f"Epoch [{epoch}/{epochs}]")
    # set model to train mode
    model.train()
    # clear optimizer at the very beginning for new epoch
    optimizer.zero_grad()
    for batch_idx, images in pbar:
        loss_accum = 0.0
        output = model(images.to(device))
        loss = output.mean()
        loss_accum = loss.detach()
        # loss.backward()
        if dp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)  # all_reduce (mean)
        # update weights at the 'accum_steps'st step of every 'accum_steps' steps
        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
            # clear history gradients before the next gradients accumulation period starts
            optimizer.zero_grad()
        # update progress bar
        pbar.set_postfix(loss=loss_accum.item())
        # print log and save training history
        if batch_idx % log_interval == 0:
            # train_batch_log(batch_idx, batch_size, len(trainset_loader.dataset), loss_accum.item())
            train_losses.append(loss_accum.item())  # .item(): to scalar
            train_counter.append(batch_idx + int((epoch - 1) * len(trainset_loader)))
    # update learning rate
    learning_rate_scheduler.step()
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    # save model and optimizer at the end of every epoch (only in process #0)
    if dp_global_rank == 0:
        torch.save(model.state_dict(), f'{save_model_dir}/history/model_{date_time}.pth')
        torch.save(optimizer.state_dict(), f'{save_model_dir}/history/optimizer_{date_time}.pth')

def train_on_epoch_amp(model, trainset_loader, optimizer, learning_rate_scheduler, batch_size:int, \
                       accum_steps:int, train_losses:list, train_counter:list, save_model_dir:str, \
                       log_interval:int, epoch:int, epochs:int, dp_global_rank:int, device, dp:bool, \
                       **kwargs):
    print(f'Epoch: {epoch}: ')
    pbar = tqdm(enumerate(trainset_loader), total=len(trainset_loader), leave=True, ncols=None)
    pbar.set_description(f"Epoch [{epoch}/{epochs}]")
    # set model to train mode
    model.train()
    # clear optimizer at the very beginning for new epoch
    optimizer.zero_grad()
    grad_scaler = kwargs['grad_scaler']
    for batch_idx, (images, labels, sample_weights) in pbar:
        # Runs the forward pass with autocasting.
        with torch.autocast(device_type=device.type, dtype=kwargs['amp_dtype']):
            loss_accum = 0.0
            output = model(images.to(device))
            loss = output.mean()
        loss_accum = loss.detach()
        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        # grad_scaler.scale(loss).backward()
        if dp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)  # all_reduce (mean)
        if (batch_idx + 1) % accum_steps == 0:
            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            grad_scaler.step(optimizer)
            # Updates the scale for next iteration.
            grad_scaler.update()
            optimizer.zero_grad()
        # update progress bar
        pbar.set_postfix(loss=loss_accum.item())
        # print log and save training history
        if batch_idx % log_interval == 0:
            # train_batch_log(batch_idx, batch_size, len(trainset_loader.dataset), loss_accum.item())
            train_losses.append(loss_accum.item())  # .item(): to scalar
            train_counter.append(batch_idx + int((epoch - 1) * len(trainset_loader)))
    # update learning rate
    learning_rate_scheduler.step()
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    # save model and optimizer at the end of every epoch (only in process #0)
    if dp_global_rank == 0:
        torch.save(model.state_dict(), f'{save_model_dir}/history/model_{date_time}.pth')
        torch.save(optimizer.state_dict(), f'{save_model_dir}/history/optimizer_{date_time}.pth')

def valid_on_epoch(model, validset_loader, valid_losses:list, valid_counter:list, per_epoch_batches:int, \
                   epoch:int, dp_global_rank:int, device, dp:bool):
    # set model to eval mode
    model.eval()
    loss_accum = 0.0
    with torch.no_grad():
        for images in validset_loader:
            # RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) 
            # should be the same or input should be a MKLDNN tensor and weight is a dense tensor
            output = model(images.to(device))
            loss = output.mean()
            # loss_accum = loss.detach()
            loss_accum = loss
    if dp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)  # all_reduce (mean)
    loss_accum /= len(validset_loader)
    # print current valid log
    # valid_epoch_log(epoch, len(validset_loader.dataset), loss_accum.item())
    valid_losses.append(loss_accum.item())
    valid_counter.append(per_epoch_batches * epoch)
    
def resume_from_ckpt(model, optimizer, save_model_dir:dir):
    model_ckpt_path = os.path.join(save_model_dir, 'model.pth')
    optimizer_ckpt_path = os.path.join(save_model_dir, 'optimizer.pth')
    if os.path.exists(model_ckpt_path):
        model_ckpt = torch.load(model_ckpt_path)
        model.load_state_dict(model_ckpt)
    if os.path.exists(optimizer_ckpt_path):
        optimizer_ckpt = torch.load(optimizer_ckpt_path)
        optimizer.load_state_dict(optimizer_ckpt)
    return model, optimizer