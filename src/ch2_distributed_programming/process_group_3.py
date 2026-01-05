"""
src/process_group_3.py
"""

import torch.multiprocessing as mp
import torch.distributed as dist
import os


# Area executed concurrently in subprocesses
def fn(rank, world_size):
    # `rank` is passed automatically. `world_size` is provided as input.
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # Initialize process group
    group = dist.new_group([_ for _ in range(world_size)])
    # Create process group
    print(f"{group} - rank: {rank}")
    dist.destroy_process_group()


# Main process
if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = "4"

    mp.spawn(
        fn=fn,
        args=(4,),  # Input world_size
        nprocs=4,  # Number of processes to create
        join=True,  # Whether to join processes
        daemon=False,  # Daemon flag
        start_method="spawn",  # Set start method
    )
