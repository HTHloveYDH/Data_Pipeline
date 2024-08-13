import torch
import torchvision

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)


def get_model(dist_strategy:str, device, device_ids:list, dp_local_rank:int, **kwargs):
    backbone = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1', progress=True)
    model = backbone
    model.to(device)
    use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
    if use_compile:
        model = torch.compile(model)
    if dist_strategy == 'ddp':
        model = DDP(model, device_ids=device_ids)
    elif dist_strategy == 'fsdp':
        # reference: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html#how-to-use-fsdp
        model = FSDP(model)
        # my_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100)
        # model = FSDP(
        #     model, auto_wrap_policy=my_auto_wrap_policy, cpu_offload=CPUOffload(offload_params=True)
        # )
    print(f'distribute strategy is set to {dist_strategy}')
    return model