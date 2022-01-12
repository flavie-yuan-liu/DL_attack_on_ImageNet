import numpy as np

from attacks.utils import *
import torch
import os
import copy

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from env_setting import dist_init, cleanup, world_size, rank, local_rank
import torch.nn as nn


class Attack_dict_model(nn.Module):

    def __init__(self, d, v, eps):
        super(Attack_dict_model, self).__init__()
        self.d = nn.Parameter(d)
        self.v = nn.Parameter(v)
        self.eps = eps

    def forward(self, x, index, model):
        dv = torch.tensordot(self.v[index, :], self.d, dims=([1], [3]))
        output = model(x + dv)
        return output

    def update_v(self):
        self.v.data = project_onto_l1_ball(self.v.data, self.eps)

    def update_d(self):
        self.d.data = torch.clamp(self.d.data, min=-1, max=1)


class ADIL(Attack):
    """
    ADiL in the paper 'Adversarial Dictionary Learning'
    Arguments:
        model (nn.Module): model to attack.
        norm (str): Lp-norm of the attack. ['Linf', 'L2'] (Default: 'L2')
        eps (float): maximum perturbation. (Default: None)
        targeted (bool): if True, attacks target the second most probable label.
                         Otherwise it performs untargeted attacks. (Default: False)
        attack (string): behavior when attacking unseen examples. (Default: 'unsupervised')
                         'unsupervised' <--> sample the coding vectors
                         'supervised' <--> optimize over the coding vectors
        data_train (Dataset): dataset used to learn the dictionary.
        steps (int): number of steps to learn the dictionary. (Default: 100).
        step_size (int): step-size (Default: 10 for norm='L2' and eps/steps for norm='Linf')
        trials (int): number of trials to find the best unsupervised attacks to unseen examples. (Default: 10)
        batch_size (int): batch size used to compute the gradients. Larger values speed up the computation at the cost
                          of a higher memory storage. It has no impact on the results. (Default: len(data_train)).
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, 'H = height`
                    and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 <= y_i <= ` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    """

    def __init__(self, model, eps=None, steps=1e2, norm='L2', targeted=True, n_atoms=10, batch_size=None,
                 data_train=None, data_val=None, trials=10, attack='supervised', model_name=None, step_size=0.01,
                 is_distributed=False, steps_in=None, loss='ce', method='gd', warm_start=False, alpha=1/255):

        super().__init__("ADIL", model.eval())
        # Attack parameters
        self.norm = norm.lower()
        self.eps = eps
        self.n_atoms = n_atoms
        self.dictionary = None
        self.targeted = targeted
        self.attack = attack
        self.trials = trials
        self.alpha = alpha
        self.step_size = step_size

        # Algorithmic parameters
        self.steps = steps
        self.steps_inner = steps_in
        self.batch_size = batch_size
        self.loss = loss
        self.model_name = model_name
        self.method = method

        # path = f"trained_dicts /"
        path = f"dict_model_ImageNet_version_constrained/"
        model_file = f"ImageNet_{model_name}_num_atom_{self.n_atoms}_nepoch_{self.steps}_v_step_{steps_in}_d_step_{steps_in}" \
                     f"_sep_{len(data_train)}_{loss}_method_{method}.bin"
        self.model_file = os.path.join(path, model_file)

        # Learn dictionary
        if not os.path.exists(self.model_file):
            if is_distributed:
                IP = os.environ['SLURM_STEP_NODELIST']
                # world_size = int(os.environ['SLURM_NTASKS'])
                # world_size = torch.cuda.device_count()
                # print('world_size', world_size)
                # mp.spawn(self.learn_dictionary_distributed, args=(world_size, data_train), nprocs=world_size,
                #         join=True)
                self.learn_dictionary_distributed(data_train)
            else:
                if method == 'gd':
                    self.learn_dictionary_a(dataset=data_train, warm_start=True)
                elif method == 'alter':
                    self.learn_dictionary_b(dataset=data_train, warm_start=True)


    def f_loss(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1) # get the second largest logit
        j = torch.masked_select(outputs, one_hot_labels.bool()) # get the largest logit

        if self._targeted:
            return torch.clamp((i-j), min=0)
        else:
            return torch.clamp((j-i), min=0)

    def learn_dictionary_a(self, dataset, val, warm_start):
        """ Learn the adversarial dictionary """
        dataset.indexed = False
        # Shape parameters
        n_img = len(dataset)
        x, _ = next(iter(dataset))
        nc, nx, ny = x.shape

        # Other parameters
        batch_size = n_img if self.batch_size is None else self.batch_size
        coeff = 1. if self.targeted else -1.

        # Data loader
        dataset.indexed = True
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                                  num_workers=0)
        val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True, pin_memory=True,
                                                 num_workers=True)

        # Function
        criterion = nn.CrossEntropyLoss(reduction='mean')

        # Initialization of the dictionary D and coding vectors v
        if warm_start:
            path = f"dict_model_ImageNet_version_constrained/"
            warm_start_file = f"ImageNet_{self.model_name}_num_atom_{self.n_atoms}_nepoch_{self.steps}_AdamW" \
                         f"_{200}.bin"
            d, _, _, _ = torch.load(os.path.join(path, warm_start_file))
        else:
            if self.norm.lower() == 'l2':
                d = self.projection_d(torch.randn(nc, nx, ny, self.n_atoms, device=self.device))
            else:
                d = (-1 + 2*torch.rand(nc, nx, ny, self.n_atoms, device=self.device))

        v = self.projection_v(torch.rand(n_img, self.n_atoms, device=self.device))

        # initialized model
        adil_model = Attack_dict_model(d, v, self.eps-self.alpha).to(self.device)
        # optimise = torch.optim.AdamW([
        #     {'params': adil_model.d, 'lr': 0.02},
        #     {'params': adil_model.v}
        # ], lr=5e-3)
        optimise = torch.optim.AdamW(adil_model.parameters(), lr=self.step_size)

        # Initialization of intermediate variables
        loss_all = []
        fooling_rate_all = []
        # Algorithm
        adil_model.train()
        bar = trange(int(self.steps))
        for iteration in bar:
            # Gradients and loss computations
            loss_full = 0
            fooling_sample = 0
            for index, x, label in data_loader:
                # Load data
                x, label = x.to(device=self.device), label.to(device=self.device)

                # compute loss with model
                optimise.zero_grad()
                output = adil_model(x, index, self.model)
                fooling_sample += torch.sum(output.argmax(dim=-1) != label)

                if self.loss == 'ce':
                    loss = coeff*criterion(output, label)
                elif self.loss == 'logits':
                    loss = self.f_loss(output, label).sum()

                loss.backward()
                optimise.step()
                adil_model.update_v()
                adil_model.update_d()

                with torch.no_grad():
                    loss_full += loss

            loss_all.append(loss_full.item())
            fooling_rate_all.append(fooling_sample.item()/n_img)
            print(loss_all[-1], fooling_sample/n_img)

            fool_on_valset = 0
            for x, label in val_loader:
                fool_on_valset += self.forward_supervised_AdamW(x, label, adil_model.d.data, 'train')
            print(fool_on_valset/len(val))

            if iteration > 1 and abs(loss_all[iteration] - loss_all[iteration - 1]) < 1e-6:
                break

        torch.save([adil_model.d.data, adil_model.v.data, loss_all, fooling_rate_all, fool_on_valset/len(val)], self.model_file)

    def learn_dictionary_b(self, dataset, val, warm_start):
        """ Learn the adversarial dictionary """
        dataset.indexed=False
        # Shape parameters
        n_img = len(dataset)
        x, _ = next(iter(dataset))
        nc, nx, ny = x.shape

        # Other parameters
        batch_size = n_img if self.batch_size is None else self.batch_size
        coeff = 1. if self.targeted else -1.

        # Data loader
        dataset.indexed = True
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                                  num_workers=4)
        val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True, pin_memory=True,
                                                 num_workers=True)

        # Function
        criterion = nn.CrossEntropyLoss(reduction='mean')

        # Initialization of the dictionary D and coding vectors v
        if warm_start:
            path = f"dict_model_ImageNet_version_constrained/"
            warm_start_file = f"ImageNet_{self.model_name}_num_atom_{self.n_atoms}_nepoch_{self.steps}_AdamW" \
                              f"_{200}.bin"
            d, _, _, _ = torch.load(os.path.join(path, warm_start_file))
        else:
            if self.norm.lower() == 'l2':
                d = self.projection_d(torch.randn(nc, nx, ny, self.n_atoms, device=self.device))
            else:
                d = (-1 + 2 * torch.rand(nc, nx, ny, self.n_atoms, device=self.device))

        v = self.projection_v(torch.zeros(n_img, self.n_atoms, device=self.device))

        # initialized model
        adil_model = Attack_dict_model(d, v, self.eps).to(self.device)
        optimise_d = torch.optim.AdamW([adil_model.d], lr=2*self.step_size)
        optimise_v = torch.optim.AdamW([adil_model.v], lr=self.step_size)

        # Initialization of intermediate variables
        loss_all = []
        fooling_rate_all = []

        # Algorithm
        # bar = trange(int(self.steps))
        bar = range(int(self.steps//self.steps_inner))
        for iteration in bar:
            # Gradients and loss computations
            # v_step
            for _ in range(self.steps_inner):
                loss_full = 0
                fooling_sample = 0
                for index, x, label in data_loader:
                    # Load data
                    x, label = x.to(device=self.device), label.to(device=self.device)
                    adil_model.d.requires_grad = False
                    adil_model.v.requires_grad = True
                    output = adil_model(x, index, self.model)
                    fooling_sample += torch.sum(output.argmax(-1) != label)
                    optimise_v.zero_grad()
                    if self.loss == 'ce':
                        loss = coeff * criterion(output, label)
                    elif self.loss == 'logits':
                        loss = self.f_loss(output, label).sum()
                    loss.backward()

                    optimise_v.step()
                    adil_model.update_v()

                    with torch.no_grad():
                        loss_full += loss

                print(loss_full, fooling_sample / n_img)

            # d_step
            for _ in range(self.steps_inner):
                loss_full = 0
                fooling_sample = 0
                for index, x, label in data_loader:
                    # Load data
                    x, label = x.to(device=self.device), label.to(device=self.device)
                    adil_model.v.requires_grad = False
                    adil_model.d.requires_grad = True
                    output = adil_model(x, index, self.model)
                    fooling_sample += torch.sum(output.argmax(-1) != label)
                    optimise_d.zero_grad()
                    if self.loss == 'ce':
                        loss = coeff * criterion(output, label)
                    elif self.loss == 'logits':
                        loss = self.f_loss(output, label).sum()
                    loss.backward()

                    optimise_d.step()
                    adil_model.update_d()

                with torch.no_grad():
                    loss_full += loss

                    # print('d_step loss', loss_full)
            loss_all.append(loss_full.item())
            fooling_rate_all.append(fooling_sample.item() / n_img)

            print(loss_all[-1], fooling_sample/n_img)
            fool_on_valset = 0
            for x, label in val_loader:
                fool_on_valset += self.forward_supervised_AdamW(x, label, adil_model.d.data, 'train')
            print(fool_on_valset / len(val))

            if iteration > 1 and abs(loss_all[iteration] - loss_all[iteration - 1]) < 1e-6:
                break

        torch.save([adil_model.d.data, adil_model.v.data, loss_all, fooling_rate_all, fool_on_valset/len(val)], self.model_file)

    def learn_dictionary_distributed(self, dataset):
        """ Learn the adversarial dictionary by distributed data parallel"""

        # Shape parameters
        dataset.indexed = False
        n_img = len(dataset)
        x, _ = next(iter(dataset))
        nc, nx, ny = x.shape

        # # Line-search parameters
        # delta = .5
        # gamma = 1
        # beta = .5

        # Other parameters
        batch_size = n_img if self.batch_size is None else self.batch_size
        coeff = 1. if self.targeted else -1.

        #########################################
        # distributed process initialization
        #########################################
        print(rank, world_size, local_rank)
        dist_init(rank=rank, world_size=world_size)
        # local_rank = int(os.environ['SLURM_LOCALID'])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        print(device)
        print(dist.get_rank())

        ##########################################
        # distributed data loader
        ##########################################
        dataset.indexed = True  # Change dataset getitem to indexed mode by setting dataset.indexed to True
        data_sampler = DistributedSampler(dataset)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                                  num_workers=4, sampler=data_sampler)
        # val_sampler = DistributedSampler(validation)
        # val_loader = torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=False, pin_memory=True,
        #                                          num_workers=4, sampler=val_sampler)

        # Optimisation params
        # Initialization of the dictionary D and coding vectors v
        if self.norm.lower() == 'l2':
            d = self.projection_d(torch.randn(nc, nx, ny, self.n_atoms, device=device))
        else:
            d = (-1 + 2 * torch.rand(nc, nx, ny, self.n_atoms, device=device))

        v = self.projection_v(torch.randn(n_img, self.n_atoms, device=device))

        ###########################################
        # Create DDP model
        ###########################################
        criterion = nn.CrossEntropyLoss(reduction='mean')
        dict_model = Attack_dict_model(d, v, self.eps).cuda(local_rank)
        ddp_model = DDP(dict_model, device_ids=[local_rank])

        # Initialize Optimiser
        optimise = torch.optim.AdamW(ddp_model.parameters(), lr=0.01 * dist.get_world_size())

        # Initialization of intermediate variables
        loss_all = []
        fooling_rate_all = []

        if rank == 0:
            print('===============================begin to train==========================')
            # Algorithm
            bar = trange(int(self.steps))
            for _ in bar:
                # Gradients and loss computations
                loss_full = 0
                fooling_sample = 0
                optimise.zero_grad()
                for index, x, label in data_loader:
                    # Load data
                    # print(len(index))
                    x, label = x.to(device=device), label.to(device=device)

                    # compute loss with model
                    output = ddp_model(x, index, self.model.to(device))
                    fooling_sample_s = torch.sum(output.argmax(-1) != label)
                    if self.loss == 'ce':
                        loss = coeff * criterion(output, label)
                    elif self.loss == 'logits':
                        loss = self.f_loss(output, label, device).sum()

                    loss.backward()
                    optimise.step()

                dist.barrier()
                with torch.no_grad():
                    ddp_model.module.update_v()
                    ddp_model.module.update_d()
                    dist.reduce(loss, 0, op=dist.ReduceOp.SUM)
                    dist.reduce(fooling_sample_s, 0, op=dist.ReduceOp.SUM)
                    loss_full += loss
                    fooling_sample += fooling_sample_s

            if rank == 0:
                fooling_rate_all.append(fooling_sample.item() / n_img)
                loss_all.append(loss_full.item())

        if rank == 0:
            torch.save([ddp_model.module, loss_all, fooling_rate_all], self.model_file)

        cleanup()

        # Initialization of the dictionary D and coding vectors v
        if self.norm.lower() == 'l2':
            d = self.projection_d(torch.randn(nc, nx, ny, self.n_atoms, device=device))
        else:
            d = (-1 + 2*torch.rand(nc, nx, ny, self.n_atoms, device=device))

        v = self.projection_v(torch.randn(n_img, self.n_atoms, device=device))

        ###########################################
        # Create DDP model
        ###########################################
        criterion = nn.CrossEntropyLoss(reduction='mean')
        dict_model = Attack_dict_model(d, v, self.eps).cuda(rank)
        ddp_model = DDP(dict_model, device_ids=[rank])

        # Initialize Optimiser
        optimise = torch.optim.AdamW(ddp_model.parameters(), lr=0.01*dist.get_world_size())

        # Initialization of intermediate variables
        loss_all = []
        fooling_rate_all = []

        if rank == 0:
            print('===============================begin to train==========================')

        # Algorithm
        bar = trange(int(self.steps))
        for _ in bar:
            # Gradients and loss computations
            loss_full = 0
            fooling_sample = 0
            optimise.zero_grad()
            for index, x, label in data_loader:
                # Load data
                x, label = x.to(device=device), label.to(device=device)

                # compute loss with model
                output = ddp_model(x, index, self.model.to(device))
                fooling_sample_s = torch.sum(output.argmax(-1) != label)
                if self.loss == 'ce':
                    loss = coeff*criterion(output, label)
                elif self.loss == 'logits':
                    loss = self.f_loss(output, label).sum()

                loss.backward()
                optimise.step()

                dist.barrier()
                with torch.no_grad():
                    ddp_model.module.update_v()
                    ddp_model.module.update_d()
                    dist.reduce(loss, 0, op=dist.ReduceOp.SUM)
                    dist.all_reduce(fooling_sample_s, 0, op=dist.ReduceOp.SUM)
                    loss_full += loss
                    fooling_sample += fooling_sample_s

            if rank == 0:
                fooling_rate_all.append(fooling_sample.item() / n_img)
                loss_all.append(loss_full.item())

        if rank == 0:
            torch.save([ddp_model.module, loss_all, fooling_rate_all], self.model_file)

        cleanup()

    def forward(self, images, labels):

        images = images.to(self.device)
        labels = labels.to(self.device)

        # Check if the dictionary has been learned
        if not os.path.exists(self.model_file):
            print('The adversarial dictionary has not been learned.')
            print('It is now being learned on the given dataset')
            dataset = QuickAttackDataset(images=images, labels=labels)
            self.learn_dictionary(dataset=dataset, model=self.model)

        self.dictionary, _, _, _ = torch.load(self.model_file)

        if self.attack == 'supervised':
            ''' Supervised attack where the coding vectors are optimized '''
            # adv_img = self.forward_supervised_new(images, labels)
            adv_img = self.forward_supervised_AdamW(images, labels, self.dictionary, 'val')
            self.dictionary = None
            return adv_img
        else:
            ''' Unsupervised attack where the coding vectors are sampled '''
            adv_img = self.forward_unsupervised(images)
            self.dictionary = None
            return adv_img

    def forward_unsupervised(self, images):
        """ Unsupervised attack to unseen examples
        The method relies on sampling the coding vectors randomly according to some Laplace distribution
        """

        # Parameters
        n_samples = images.shape[0]

        # Variables
        fooling_flag = torch.zeros(n_samples, dtype=torch.bool).to(device=self.device)
        mse_best_do_fool = np.inf * torch.ones(n_samples)
        mse_best_no_fool = np.inf * torch.ones(n_samples)
        adv_images_best = images.clone()

        for n_a in range(int(self.trials)):

            # Sample adversarial images
            dv_norm_inf = []
            v = self.sample_sphere(n_samples).to(device=self.device)
            # v = torch.zeros(n_samples, self.n_atoms, device=self.device)
            # v[:, n_a//2] = self.eps if n_a%2 == 0 else -self.eps
            adv_images = torch.zeros_like(images)
            for ind in range(n_samples):
                dv = torch.tensordot(v[ind], self.dictionary.to(device=self.device), dims=([0], [3]))
                dv = torch.clamp(dv, min=-self.eps, max=self.eps)
                dv_norm_inf.append(torch.max(torch.abs(dv)).item())
                adv_images[ind, :, :, :] = clamp_image(images[ind, :, :, :] + dv)
            # print(dv_norm_inf)
            adv_labels = self.model(adv_images).argmax(dim=1)
            pre_labels = self.model(images).argmax(dim=1)

            # Evaluate their performance
            fooling = ~torch.eq(pre_labels, adv_labels)
            mse = torch.sum((images - adv_images) ** 2, dim=[1, 2, 3])

            # Keep the best ones
            for ind in range(n_samples):
                if not fooling_flag[ind] and fooling[ind]:
                    # the model has been fooled with this model and this adv is also fooling the model
                    fooling_flag[ind]=True
                    mse_best_do_fool[ind] = mse[ind]
                    adv_images_best[ind] = adv_images[ind]
                elif (fooling_flag[ind] and fooling[ind]) or (not fooling_flag[ind] and not fooling[ind]):
                    # the model has never been fooled with this sample
                    if mse[ind] < mse_best_no_fool[ind]:
                        mse_best_no_fool[ind] = mse[ind]
                        adv_images_best[ind] = adv_images[ind]

        return adv_images_best, dv_norm_inf

    def forward_supervised_new(self, images, labels):

        d = self.dictionary

        # Shape parameters
        n_img = len(labels)

        # Line-search parameters
        delta = .5
        beta = .5

        # Other parameters
        coeff = 1. if self.targeted else -1.  # Targeted vs. Untargeted attacks
        step_size = self.step_size * 0.01

        # Function
        criterion = nn.CrossEntropyLoss(reduction='none')

        # Initialization of the coding vectors v
        v = torch.zeros(n_img, self.n_atoms, device=self.device)

        # Initialization of intermediate variables
        v_old = torch.zeros_like(v)
        loss_all = np.nan * np.ones(int(self.steps))

        # Algorithm for image-wise code computing
        adversary = torch.empty(0, device=self.device)
        dv_norm_inf=[]

        for ind in range(n_img):
            img, label = images[ind, :, :, :].to(self.device), labels[ind].to(self.device)
            # print('{}_th image'.format(ind))
            vi = v[ind]
            for iteration in range(int(self.steps)):
                vi.detach()
                vi.requires_grad = True
                dv = torch.tensordot(vi, d, dims=([0], [3]))
                loss_i = coeff * criterion(self.model(img + dv), label.unsqueeze(0))

                loss_i.backward()

                # Forward-Backward step with line-search
                with torch.no_grad():
                    grad_v = vi.grad

                    vi_old = v_old[ind].copy_(vi)
                    loss_i_old = loss_i.item()
                    vi = self.projection_v((torch.clamp(vi - step_size * grad_v, min=0)).unsqueeze(0))[0]

                    # added distance
                    d_v = vi - vi_old

                    # First order approximation of the difference in loss
                    h = torch.sum((vi - vi_old) * grad_v) \
                        + .5 / self.step_size * torch.sum((vi - vi_old) ** 2)

                    dv = torch.tensordot(vi, d, dims=([0], [3]))
                    loss_i_cur = coeff * criterion(self.model(img + dv), label.unsqueeze(0)).item()
                    loss_i_new = loss_i_cur

                    # line-searching
                    index_i = 0
                    while loss_i_new > loss_i_old + beta*(delta**index_i)*h.item() and index_i<20:
                        index_i += 1
                        vi_new = vi_old + (delta ** index_i) * d_v
                        dv = torch.tensordot(vi_new, d, dims=([0], [3]))
                        loss_i_new = coeff * criterion(self.model(img + dv), label.unsqueeze(0)).item()

                    if loss_i_cur <= loss_i_new:
                        loss_i_new = loss_i_cur
                    else:
                        vi = vi_new
                    if torch.max(torch.abs(vi-vi_old)) < 1e-6:
                       break

                # Keep track of loss
                if np.isnan(loss_all[iteration]):
                    loss_all[iteration] = loss_i_new
                else:
                    loss_all[iteration] += loss_i_new
                print(loss_i_new)

            # Output
            dv = torch.tensordot(self.projection_v(torch.clamp(vi, min=0)), d, dims=([0], [3]))
            dv_norm_inf.append(torch.max(torch.abs(dv)).item())
            print(iteration, self.model(img+dv).argmax(-1), label)
            adversary = torch.cat([adversary, (img + dv).unsqueeze(0)])

        return torch.clamp(adversary, min=0, max=1), dv_norm_inf

    def forward_supervised(self, images, labels):

        dataset = QuickAttackDataset(images=images, labels=labels)
        d = self.dictionary

        # Shape parameters
        n_img = len(dataset)

        # Line-search parameters
        delta = .5
        gamma = 1
        beta = .5
        lipschitz = .9 / self.step_size

        # Other parameters
        batch_size = n_img if self.batch_size is None else self.batch_size
        coeff = 1. if self.targeted else -1.  # Targeted vs. Untargeted attacks
        indices = get_slices(n_img, batch_size)  # Slices of samples according to the batch-size

        # Data loader
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Function
        criterion = nn.CrossEntropyLoss(reduction='sum')
        criterion_batch = nn.CrossEntropyLoss(reduction='none')
        target = self.get_target(data_loader=data_loader)

        # Initialization of the coding vectors v
        v = torch.zeros(n_img, self.n_atoms, device=self.device)

        # Initialization of intermediate variables
        v_old = torch.zeros_like(v)
        grad_v_old = torch.zeros_like(v)
        loss_all = np.nan * np.ones(int(self.steps))

        # Line-search parameters
        flag_stop = [False] * n_img
        batch_mask = torch.ones(n_img, device=self.device)[:, None]
        min_index_i = torch.ones(n_img, device=self.device)
        loss_batch_save = torch.tensor([np.NAN] * n_img, device=self.device)

        # Algorithm
        bar = trange(int(self.steps))
        for iteration in bar:

            if sum(flag_stop) < n_img:

                # Gradients and loss computations
                loss_batch = torch.empty(0, device=self.device)
                for index, (x, _) in enumerate(data_loader):
                    # Prepare computation graph
                    v.detach()
                    v.requires_grad = True

                    # Load data
                    x = x.to(device=self.device)
                    ind = indices[index]
                    dv = torch.tensordot(v[ind], d, dims=([1], [3]))
                    loss_tmp = coeff * criterion_batch(self.model(x + dv), target[index])
                    loss = torch.sum(loss_tmp)
                    loss.backward()

                    with torch.no_grad():
                        loss_batch = torch.cat([loss_batch, loss_tmp])

                # Forward-Backward step with line-search
                with torch.no_grad():

                    grad_v = batch_mask * v.grad

                    # Guess the Lipschitz constant
                    if self.estimate_step_size:
                        if iteration <= 1:
                            lipschitz_old = lipschitz
                            lipschitz = torch.norm(grad_v - grad_v_old, 'fro') / torch.norm(v - v_old, 'fro')
                            lipschitz = lipschitz_old if torch.isinf(lipschitz) else lipschitz
                            self.step_size = .9 / lipschitz

                    # Memory
                    v_old.copy_(v)
                    grad_v_old.copy_(grad_v)

                    # Update
                    v = self.projection_v(v - self.step_size * grad_v)

                    # added distance
                    d_v = v - v_old

                    # First order approximation of the difference in loss
                    h = torch.sum((v - v_old) * grad_v, dim=[1]) \
                        + .5 * (gamma / self.step_size) * torch.sum((v - v_old) ** 2, dim=[1])

                    # Line-search
                    warm_restart = True
                    flag_batch = flag_stop.copy()  # No need to check samples for which it has converged
                    if warm_restart:
                        index_i = torch.clamp(min_index_i - 2, min=0)  # Warm-restart of line-search parameter
                    else:
                        index_i = torch.zeros(n_img, device=self.device)

                    while sum(flag_batch) < n_img:

                        v_new = v_old + (delta ** index_i)[:, None] * d_v

                        # Compute the loss (we could cut the running by 2 if we stored the computation graph ..)
                        loss_batch_new = torch.empty(0, device=self.device)
                        for index, (x, _) in enumerate(data_loader):
                            x = x.to(device=self.device)
                            ind = indices[index]
                            dv = torch.tensordot(v_new[ind], d, dims=([1], [3]))
                            loss_batch_new = torch.cat([loss_batch_new, (coeff * criterion_batch(self.model(x + dv),
                                                                                                 target[index]))])

                        # Check the sufficient decrease condition
                        criterion = loss_batch + beta * (delta ** index_i) * h
                        for ind_batch, (loss_val, criterion_val) in enumerate(zip(loss_batch_new, criterion)):

                            # only modify the epsilon for which convergence has not been reached
                            if flag_batch[ind_batch] is False:
                                if loss_val <= criterion_val:
                                    v[ind_batch] = v_new[ind_batch]
                                    flag_batch[ind_batch] = True
                                    loss_batch_save[ind_batch] = loss_batch_new[ind_batch]
                                    min_index_i[ind_batch].copy_(index_i[ind_batch])
                                else:
                                    if index_i[ind_batch] > 50:
                                        flag_batch[ind_batch] = True
                                        flag_stop[ind_batch] = True
                                        batch_mask[ind_batch] = 0

                        # Update the line-search index
                        index_i.add_(1)

                    v = copy.deepcopy(v).requires_grad_()
                    # Keep track of loss
                    loss_all[iteration] = torch.sum(loss_batch_save)

        # Output
        adversary = torch.empty(0, device=self.device)
        for index, (x, _) in enumerate(data_loader):
            x = x.to(device=self.device)
            ind = indices[index]
            dv = torch.tensordot(self.projection_v(v[ind]), d, dims=([1], [3]))
            adversary = torch.cat([adversary, x + dv])

        return torch.clamp(adversary, min=0, max=1)

    def forward_supervised_AdamW(self, images, labels, d, model='train'):
        """ Learn the adversarial dictionary by distributed data parallel"""
        # Shape parameters
        n_img = len(labels)

        # Other parameters
        coeff = 1. if self.targeted else -1.  # Targeted vs. Untargeted attacks

        # Function
        criterion = nn.CrossEntropyLoss(reduction='mean')

        # Initialization of the coding vectors v
        v = torch.zeros(n_img, self.n_atoms, device=self.device)

        # Initialization of intermediate variables
        loss_track = []

        adil_model = Attack_dict_model(d, v, eps=self.eps).to(self.device)
        adil_model.d.requires_grad = False
        optimise = torch.optim.AdamW([adil_model.v], lr=1e-2)

        # Algorithm for image-wise code computing
        # adversary = torch.empty(0, device=self.device)

        # Algorithm
        for iteration in range(50):
            # Gradients and loss computations
            loss_full = 0
            optimise.zero_grad()

            # Load data
            images, labels = images.to(device=self.device), labels.to(device=self.device)

            # compute loss with model
            output = adil_model(images, range(n_img), self.model.to(self.device))
            fooling_sample = torch.sum(output.argmax(-1) != labels)
            # if self.loss == 'ce':
            loss = coeff * criterion(output, labels)
            # elif self.loss == 'logits':
            #     loss = self.f_loss(output, labels).sum()

            loss.backward()
            v_old = adil_model.v.data.detach().clone()

            optimise.step()
            adil_model.update_v()

            loss_track.append(loss)

            if (adil_model.v.data-v_old).abs().max() < 1e-6:
                break

        if model == 'train':
            return fooling_sample
        else:
            dv = torch.tensordot(self.projection_v(adil_model.v.data), adil_model.d.data, dims=([1], [3]))
            adversary = images + dv
            return torch.clamp(adversary, min=0, max=1)

    def projection_v(self, var):
        if self.norm == 'l2':
            """ In order to respect l2 bound, v has to lie inside a l2-ball """
            v_norm = torch.norm(var, p='fro', dim=1, keepdim=True)
            return self.eps * torch.div(var, torch.maximum(v_norm, self.eps * torch.ones_like(v_norm)))

        elif self.norm == 'linf':
            """ In order to respect linf bound, v has to lie inside a l1-ball """
            return project_onto_l1_ball(var, eps=self.alpha)

    def projection_d(self, var):
        if self.norm == 'l2':
            """ In order to respect l2 bound, D has to lie inside a l2 unit ball """
            return constraint_dict(var, constr_set='l2ball')

        elif self.norm == 'linf':
            """ In order to respect l2 bound, D has to lie inside a linf unit ball """
            return torch.clamp(var, min=-1, max=1)

    def sample_sphere(self, n_samples):
        if self.norm == 'l2':
            """ In order to respect l2 bound, sample v on a l2-sphere """
            var = (2 * torch.rand(n_samples, self.n_atoms) - 1)
            # var = torch.randn(n_samples, self.n_atoms)
            v_norm = torch.norm(var, p='fro', dim=1, keepdim=True)
            return self.eps * torch.div(var, v_norm)
        elif self.norm == 'linf':
            """ Sample 'sparse' v on the l1-sphere """
            m = torch.distributions.uniform.Uniform(torch.tensor([self.alpha]), torch.tensor([2*self.alpha]))
            # p = torch.distributions.uniform.Uniform(torch.tensor([-1.0]), torch.tensor([1.0]))
            var_raw = m.sample(sample_shape=[n_samples, self.n_atoms])[:, :, 0]
            # var_sgn = p.sample(sample_shape=[n_samples, self.n_atoms])[:, :, 0]
            # var_sgn = np.random.choice([-1, 1], n_samples*self.n_atoms).reshape(n_samples, self.n_atoms)
            # var_sgn = torch.from_numpy(var_sgn)
            # var = torch.mul(var_sgn, var_raw)
            return self.projection_v(var_raw)

    def get_target(self, data_loader):
        """ Output the target or label to fool depending on self.targeted """
        target = []
        for index, (x, y) in enumerate(data_loader):
            target.append(get_target(x.to(device=self.device), y.to(device=self.device), self.targeted, self.model))
        return target
