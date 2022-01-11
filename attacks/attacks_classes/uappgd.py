import os

import torch

from attacks.utils import *
import torchattacks
torch.set_default_tensor_type(torch.DoubleTensor)
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
import torch.nn as nn
from env_setting import dist_init, cleanup


class UAP_Model(nn.Module):

    def __init__(self, e):
        super(UAP_Model, self).__init__()
        self.e = nn.Parameter(e, requires_grad=True)

    def forward(self, model, x):
        output = model(x+self.e)

    def update_e(self, eps):
        self.e = torch.clamp(min=-eps, max=eps)


class UAPPGD(Attack):
    """ UAP: Universal Adversarial Perturbation from  [Shafahi et al., 2020]
     beta: clipping parameter of the cross-entropy loss (Default: 9)
     eps: radius of the ball on the adversarial noise
     batch_size:
    source: https://arxiv.org/pdf/1811.11304.pdf
     """

    def __init__(self, model, data_train=None, data_val=None, steps=10, batch_size=100, beta=9, step_size=0.01, norm='l2', eps=.1,
                 optimizer='adam', distributed=None, model_name=None):
        super().__init__("UAPPGD", model)
        self.beta = beta
        self.steps = steps
        self.step_size = step_size
        self.batch_size = batch_size
        self.norm = norm
        self.eps = eps
        self.optimizer = optimizer

        root = 'dict_model_ImageNet_version_constrained/{}_uappgd/trained_dicts'.format(model_name)
        self.model_name = os.path.join(root, 'UAPPGD_model')

        if not os.path.exists(self.model_name):
            if distributed:
                IP = os.environ['SLURM_STEP_NODELIST']
                world_size = int(os.environ['SLURM_NTASKS'])
                mp.spawn(self.learn_attack_distributed, args=(world_size, IP, data_train, data_val),
                         nprocs=world_size, join=True)
            else:
                self.learn_attack(dataset=data_train, val=data_val)

    def project(self, attack):
        if self.norm.lower() == 'l2':
            attack_norm = torch.norm(attack, p='fro', keepdim=False)
            if attack_norm > self.eps:
                return self.eps * attack / attack_norm
            else:
                return attack
        elif self.norm.lower() == 'linf':
            return torch.clamp(attack, min=-self.eps, max=self.eps)

    def learn_attack(self, dataset, val):

        # data loader
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        # variable
        x, _ = dataset[0]
        attack = torch.autograd.Variable(torch.zeros(x.shape, device=self.device).unsqueeze(0), requires_grad=True)

        # optimizer
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD([attack], lr=self.step_size)
        else:
            optimizer = torch.optim.Adam([attack], lr=self.step_size)

        fooling_rate = []
        v = torch.ones((self.batch_size, 1), device=self.device)

        for _ in trange(int(self.steps)):
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                x_attack = torch.tensordot(v, attack, dims=([1], [0]))
                loss = -1*criterion(self.model(x_attack), y)
                loss = torch.clamp_min(loss, min=-self.beta)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    attack.data = self.project(attack.data)
            fooling_rate.append(compute_fooling_rate(dataset=val, attack=attack, model=self.model, device=self.device))
            print(fooling_rate[-1])

        torch.save([attack, fooling_rate], self.model_name)

    def learn_attack_distributed(self, rank, world_size, ip, dataset):

        # set environment parameters
        dist_init(ip, rank, world_size=world_size)
        local_rank = int(os.environ['SLURM_LOCALID'])
        torch.cuda.set_device(local_rank)
        device = torch.device(local_rank)
        torch.backends.cudnn.benchmark = True

        # data loader
        sampler = DistributedSampler(dataset)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                                                  pin_memory=True, sampler=sampler)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        # variable
        x, _ = dataset[0]
        attack = torch.zeros(x.shape, device=device)
        uap_model = UAP_Model(attack).cuda(local_rank)
        ddp_model = DistributedDataParallel(uap_model, device_ids=[local_rank])

        # optimizer
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(ddp_model.parameters(), lr=self.step_size)
        else:
            optimizer = torch.optim.Adam(ddp_model.parameters(), lr=self.step_size)

        fooling_rate = []
        loss_track = []

        for _ in trange(int(self.steps)):
            fooling_sample = 0
            loss_all = 0
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = ddp_model(model=self.model, x=x)
                fooling_sample_s = torch.sum(output.argmax(dim=-1) != y)
                loss = torch.clamp_min(-1*criterion(output, y), min=-self.beta)
                loss.backward()
                optimizer.step()

                dist.barrier()
                ddp_model.module.update_e(self.eps)
                dist.reduce(loss, op=dist.ReduceOp.SUM)
                dist.reduce(fooling_sample_s, op=dist.ReduceOp.SUM)
                loss_all += loss
                fooling_sample += fooling_sample_s

            fooling_rate.append(fooling_sample/len(dataset))
            loss_track.append(loss_all)
            # print(fooling_rate[-1])

        torch.save(ddp_model.module.e.data.detach(), fooling_rate)
        cleanup()

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)

        # Check if the UAP attack has been learned
        if not os.path.exists(self.model_name):
            print('The UAP attack has not been learned. It is now being learned on the given dataset.')
            dataset = QuickAttackDataset(images=images, labels=labels)
            self.learn_attack(dataset=dataset, model=self.model)
        else:
            attack, _ = torch.load(self.model_name)

        return torch.clamp(images + attack, min=0, max=1)