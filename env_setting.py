import torch.distributed as dist


def dist_init(host_addr, rank, world_size, port=12345):
    host_addr_full = f"tcp:/{host_addr}:{str(port)}"
    dist.init_process_group('nccl', init_method=host_addr_full, rank=rank, world_size=world_size)
    assert dist.is_initialized()


def cleanup():
    dist.destroy_process_group()

