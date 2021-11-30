import os
import torch
import torchmetrics

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from env_setting import dist_init, cleanup


def run_accuracy_computing(args, dataset, model):
    IP = os.environ['SLURM_STEP_NODELIST']
    world_size = int(os.environ['SLURM_NTASKS'])
    mp.spawn(model_accuracy_distributed(), args=(IP, world_size, dataset, model),
             nprocs=world_size, join=True)


def model_accuracy_distributed(rank, IP, world_size, dataset, model):

    # initial environment setting
    dist_init(host_addr=IP, rank=rank, world_size=world_size)
    local_rank = int(os.environ['SLURM_LOCALID'])
    device = torch.device(local_rank)
    torch.backends.cudnn.benchmark = True

    # initial distributed data parallel model
    ddp_model = DDP(model, device_ids=[local_rank])

    # parallel dataloader
    test_sampler = DistributedSampler(dataset)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=4,
                                              sampler=test_sampler)
    ddp_model.eval()
    with torch.no_grad():
        acc = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = ddp_model(x)
            acc_batch = torch.sum(pred.argmax(dim=-1) == y)

            dist.barrier()
            dist.reduce(acc_batch, op=dist.ReduceOp.SUM)
            acc += acc_batch
        acc = acc/len(dataset)
    cleanup()
    return acc


def model_accuracy(dataset, model, device='cpu'):
    metric = torchmetrics.Accuracy()
    metric.to(device)
    model.eval()
    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=128)
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            model = model.to(device)
            pred = model(x)
            acc = metric(pred, y)
        acc = metric.compute()
    metric.reset()
    return acc
