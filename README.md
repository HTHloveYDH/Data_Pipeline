# Data_Pipeline


### environment configuration
pip install boto3==1.34.148

pip install google-cloud-storage==2.18.0

pip install redis==5.0.7

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # pytorch 2.4.0 on cuda 11.8 (stable)
reference url: https://pytorch.org/get-started/locally/

pip install opencv-python==4.8.1.78 (optional)

pip install tqdm==4.66.5


### some api details
https://chatgpt.com/share/34c331c8-ad3f-4ebe-ad7c-14b4d6e09fc2


## lanuch training task
```bash
# [reference]: https://www.youtube.com/watch?v=KaAJtI1T2x4&list=PL_lsbAsL_o2CSuhUhJIiW0IkdT5C2wGWj
```
### simple launch on one node:
```bash
python train/main.py --dp_world_size 1
```

### DDP (FSDP) launch on one node by torch.multiprocessing (e.g. 8 GPUs):
```bash
python train/main.py --dp_world_size 8 --torch_mp_launch
```

### DDP (FSDP) launch on one node by torchrun (e.g. 8 GPUs):
```bash
torchrun --standalone --nproc_per_node=8 train/main.py --dp_world_size 8
```

### DDP (FSDP) launch on multi node by torchrun (e.g. 2 * 8 GPUs, two nodes):
```bash
# on node 0#
torchrun --nproc_per_node=8 --nnodes=2 -node_rank=0 --rdzv_backend=c10d --rdzv_endpoint=xxx.xxx.xxx.xxx:xxxx train/main.py --dp_world_size 16
```

```bash
# on node 1#
torchrun --nproc_per_node=8 --nnodes=2 -node_rank=1 --rdzv_backend=c10d --rdzv_endpoint=xxx.xxx.xxx.xxx:xxxx train/main.py --dp_world_size 16
```

## lanuch evaluation task
```bash
python eval/main.py
```
