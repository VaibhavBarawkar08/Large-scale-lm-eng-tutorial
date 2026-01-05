"""
src/broadcast.py
"""

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
#dist.init_process_group("gloo")
#dist.init_process_group("mpi")
rank = dist.get_rank()
torch.cuda.set_device(rank)


if rank == 0:
    tensor = torch.randn(2, 2).to(torch.cuda.current_device())
else:
    tensor = torch.zeros(2, 2).to(torch.cuda.current_device())

print(f"before rank {rank}: {tensor}\n")
dist.broadcast(tensor, src=0)
print(f"after rank {rank}: {tensor}\n")
