"""
src/test_mpu.py
"""
import torch
import torch.distributed as dist
from mpu import MPU

mpu = MPU(
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=2,
    #pipeline_model_parallel_size=4,
    backend="nccl",
    master_port=5678,
)

global_rank = dist.get_rank()


print(f"{global_rank}: TP group: {mpu.get_tensor_model_parallel_group()}\n")
print(f"{global_rank}: TP wsz: {mpu.get_tensor_model_parallel_world_size()}\n")
print(f"{global_rank}: TP rank: {mpu.get_tensor_model_parallel_rank()}\n")
dist.barrier()
print("\n")

print(f"{global_rank}: PP group: {mpu.get_pipeline_model_parallel_group()}\n")
print(f"{global_rank}: PP wsz: {mpu.get_pipeline_model_parallel_world_size()}\n")
print(f"{global_rank}: PP rank: {mpu.get_pipeline_model_parallel_rank()}\n")
dist.barrier()
print("\n")


print(f"{global_rank}: DP group: {mpu.get_data_parallel_group()}\n")
print(f"{global_rank}: DP wsz: {mpu.get_data_parallel_world_size()}\n")
print(f"{global_rank}: DP rank: {mpu.get_data_parallel_rank()}\n")
dist.barrier()
print("\n")


#tensor = torch.tensor([2, 3, 4, 5]).cuda() * global_rank
tensor = torch.tensor([1, 2, 3, 4]).cuda() * global_rank
tensor = torch.tensor([1., 2., 3., 4.], requires_grad=True).cuda() * global_rank
tensor = mpu.reduce(tensor)
print(f"{global_rank}: all-reduce => {tensor}")
