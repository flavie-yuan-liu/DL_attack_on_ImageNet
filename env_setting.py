import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch


import hostlist

hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
IP = hostnames[0]
gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")
world_size =  int(os.environ['SLURM_NTASKS'])
nodes =  int(os.environ['SLURM_JOB_NUM_NODES'])
rank = int(os.environ['SLURM_PROCID'])
local_rank = int(os.environ['SLURM_LOCALID'])


def dist_init(rank, world_size, port=str(12345 + int(min(gpu_ids))), host_addr=IP):
    host_addr_full = f"tcp://{host_addr}:{str(port)}"
    dist.init_process_group('nccl', init_method=host_addr_full, rank=rank,  world_size=world_size)
    assert dist.is_initialized()


def cleanup():
    dist.destroy_process_group()


