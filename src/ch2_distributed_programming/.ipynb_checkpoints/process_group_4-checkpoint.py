"""
src/process_group_4.py

$ salloc --partition=cas_v100_4 -J debug --nodes=2 --time=12:00:00 --gres=gpu:1 --comment=python
salloc: Granted job allocation 299468
salloc: Waiting for resource configuration
salloc: Nodes gpu[15,19] are ready for job

$srun torchrun --nnodes=2 --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint gpu15:12345 process_group_4.py
"""

import torch.distributed as dist

dist.init_process_group(backend="nccl")
# dist.init_process_group(backend="gloo")
# Initialize process group

group = dist.new_group([_ for _ in range(dist.get_world_size())])
# Create process group

print(f"{group} - rank: {dist.get_rank()}\n")

dist.destroy_process_group()
