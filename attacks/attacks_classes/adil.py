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
        # project v on l1 ball, i.e., ||v||_1 <= eps
        self.v.data.copy_(project_onto_l1_ball(self.v.data, self.eps))

    def update_d(self):
        # project D on l_inf with ||D||_inf <= 1 
        self.d.data.copy_(torch.clamp(self.d.data, min=-1, max=1))


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

    def __init__(self, model, eps=None, steps=5e2, norm='linf', targeted=False, n_atoms=100, batch_size=100,
                 data_train=None, data_val=None, trials=10, attack='supervised', model_name=None, step_size=0.01,
                 is_distributed=False, steps_in=None, loss='ce', method='gd', warm_start=False, kappa=50,
                 steps_inference=30):

        super().__init__("ADIL", model.eval())
        # Attack parameters
        self.norm = norm.lower()
        self.eps = eps
        self.n_atoms = n_atoms
        self.dictionary = None
        self.targeted = targeted
        self.attack = attack
        self.trials = trials
        self.step_size = step_size
        self.steps_inference = steps_inference

        # Algorithmic parameters
        self.steps = steps
        self.steps_inner = steps_in
        self.batch_size = batch_size
        self.loss = loss
        self.model_name = model_name
        self.method = method
        self.kappa = kappa

        path = f'trained_dicts/'
        model_file = f"ImageNet_{model_name}.bin"
        self.model_file = os.path.join(path, model_file)

        # Learn dictionary
        if not os.path.exists(self.model_file):
            if is_distributed:
                self.learn_dictionary_distributed(data_train)
            else:
                if method == 'gd': # simple gradient descent method
                    self.learn_dictionary_a(dataset=data_train, val=data_val, warm_start=warm_start)
                elif method == 'alter': # alternating method 
                    self.learn_dictionary_b(dataset=data_train, val=data_val, warm_start=warm_start)

    def f_loss(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1) # get the second largest logit
        j = torch.masked_select(outputs, one_hot_labels.bool()) # get the largest logit

        if self._targeted:
            return torch.clamp((i-j), min=-self.kappa)
        else:
            return torch.clamp((j-i), min=-self.kappa)

    def learn_dictionary_a(self, dataset, val, warm_start):
        """ Learn the adversarial dictionary """
        # dataset.indexed determine if output of index of data is necessairy
        dataset.indexed = False
        
        # Shape parameters
        n_img = len(dataset)
        x, _ = next(iter(dataset))
        nc, nx, ny = x.shape

        # Other parameters
        batch_size = n_img if self.batch_size is None else self.batch_size
        coeff = 1. if self.targeted else -1.

        # Data loader and make data.indexed True
        dataset.indexed = True
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                                  num_workers=0)
        val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True, pin_memory=True,
                                                 num_workers=0)

        # cross-entropy loss Function
        criterion = nn.CrossEntropyLoss(reduction='sum')

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

        # initialized model and optimiser
        adil_model = Attack_dict_model(d, v, self.eps).to(self.device)
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
                # get the output of the model
                label = self.model(x).argmax(dim=-1)

                # compute loss with model
                optimise.zero_grad()
                output = adil_model(x, index, self.model)
                fooling_sample += torch.sum(output.argmax(dim=-1) != label)

                if self.loss == 'ce':
                    loss = coeff*criterion(output, label)
                elif self.loss == 'logits':
                    loss = self.f_loss(output, label).sum()

                # constrained optimisation
                loss.backward()
                optimise.step()
                adil_model.update_v()
                adil_model.update_d()

                with torch.no_grad():
                    loss_full += loss

            # loss and fooling rate tracking
            loss_all.append(loss_full.item()/n_img)
            fooling_rate_all.append(fooling_sample.item()/n_img)
            print(loss_all[-1], fooling_sample/n_img)

            # evaluate on validation set
            fool_on_valset = 0
            adil_model.eval()
            for x, label in val_loader:
                fool_sample = self.forward_supervised_AdamW(x, label, adil_model.d.data, 'train')
                with torch.no_grad():
                    fool_on_valset += fool_sample
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
                                                  num_workers=0)
        val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True, pin_memory=True,
                                                 num_workers=0)

        # Function
        criterion = nn.CrossEntropyLoss(reduction='sum')

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
            adil_model.train()
            print('iteration', iteration)
            # Gradients and loss computations
            # v_step
            for _ in range(self.steps_inner):
                loss_full = 0
                fooling_sample = 0
                for index, x, label in data_loader:
                    # Load data
                    x, label = x.to(device=self.device), label.to(device=self.device)
                    label = self.model(x).argmax(dim=-1)
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

                print('v_step: ', loss_full.item()/n_img, fooling_sample.item() / n_img)

            # d_step
            for _ in range(self.steps_inner):
                loss_full = 0
                fooling_sample = 0
                for index, x, label in data_loader:
                    # Load data
                    x, label = x.to(device=self.device), label.to(device=self.device)
                    label = self.model(x).argmax(dim=-1)
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
            loss_all.append(loss_full.item()/n_img)
            fooling_rate_all.append(fooling_sample.item() / n_img)

            print('d_step: ', loss_all[-1], fooling_sample.item()/n_img)
            fool_on_valset = 0
            adil_model.eval()
            for x, label in val_loader:
                fool_sample = self.forward_supervised_AdamW(x, label, adil_model.d.data, 'train')
                with torch.no_grad():
                    fool_on_valset += fool_sample
            print('result on val set: ', fool_on_valset.item() / len(val))

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

    def forward(self, images, labels):

        images = images.to(self.device)
        labels = labels.to(self.device)

        # Check if the dictionary has been learned
        if not os.path.exists(self.model_file):
            print('The adversarial dictionary has not been learned.')
            print('It is now being learned on the given dataset')
            dataset = QuickAttackDataset(images=images, labels=labels)
            self.learn_dictionary(dataset=dataset, model=self.model)

        rlts = torch.load(self.model_file)
        self.dictionary = rlts[0]

        if self.attack == 'supervised':
            ''' Supervised attack where the coding vectors are optimized '''
            adv_img = self.forward_supervised_DDrague(images, labels, self.dictionary)
            # release memory
            self.dictionary = None
            return adv_img
        else:
            ''' Unsupervised attack where the coding vectors are sampled '''
            adv_img = self.forward_unsupervised(images)
            # release memory
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

    def forward_supervised_DDrague(self, images, labels, d):
        """ Learn z by supposing z = dv """
        # Shape parameters
        n_img = len(labels)

        # Other parameters
        coeff = 1. if self.targeted else -1.  # Targeted vs. Untargeted attacks

        # Function
        criterion = nn.CrossEntropyLoss(reduction='mean')

        # Initialization of the coding vectors v
        v = torch.zeros(n_img, self.n_atoms, device=self.device)

        # Pre_calculate ddrague
        dtd = torch.tensordot(d, d, dims=([0, 1, 2], [0, 1, 2]))
        dtd_inv = dtd.inverse()
        d_drg = torch.tensordot(dtd_inv, d, dims=([1], [3]))

        # Initialization of intermediate variables
        loss_track = []

        z = nn.Parameter(torch.zeros_like(images), requires_grad=True)
        optimise = torch.optim.AdamW([z], lr=1e-2)

        # Algorithm for image-wise code computing
        for iteration in range(self.steps_inference):
            optimise.zero_grad()

            # Load data
            images, labels = images.to(device=self.device), labels.to(device=self.device)
            labels = self.model(images).argmax(dim=-1)
            
            # compute loss with model
            v = torch.tensordot(z, d_drg, dims=([1, 2, 3], [1, 2, 3]))
            dv = torch.tensordot(v, d, dims=([1], [3]))
            output = self.model(images + dv)

            if self.loss == 'ce':
                loss = coeff * criterion(output, labels)
            elif self.loss == 'logits':
                loss = self.f_loss(output, labels).sum()

            loss.backward()
            z_old = z.detach().clone()

            optimise.step()
            z.data.copy_(torch.clamp(z.data, min=-self.eps, max=self.eps))

            loss_track.append(loss)

            if (z - z_old).abs().max() < 1e-6:
                break

        # calculate dv with z
        v = torch.tensordot(z, d_drg, dims=([1, 2, 3], [1, 2, 3]))
        dv = torch.tensordot(v, d, dims=([1], [3]))

        adversary = images + dv
        return torch.clamp(adversary, min=0, max=1)

    def forward_supervised_AdamW(self, images, labels, d, model='train'):
        """ learn v """
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
        for iteration in range(100):
            # Gradients and loss computations
            loss_full = 0
            optimise.zero_grad()

            # Load data
            images, labels = images.to(device=self.device), labels.to(device=self.device)
            labels = self.model(images).argmax(dim=-1)
            # compute loss with model
            output = adil_model(images, range(n_img), self.model.to(self.device))
            if self.loss == 'ce':
                loss = coeff * criterion(output, labels)
            elif self.loss == 'logits':
                loss = self.f_loss(output, labels).sum()

            loss.backward()
            v_old = adil_model.v.data.detach().clone()

            optimise.step()
            adil_model.update_v()

            loss_track.append(loss)

            if (adil_model.v.data-v_old).abs().max() < 1e-6:
                break

        dv = torch.tensordot(self.projection_v(adil_model.v.data), adil_model.d.data, dims=([1], [3]))

        if model == 'train':
            return torch.sum(self.model(images+dv).argmax(-1) != labels)
        else:
            adversary = images + dv
            return torch.clamp(adversary, min=0, max=1)

    def projection_v(self, var):
        if self.norm == 'l2':
            """ In order to respect l2 bound, v has to lie inside a l2-ball """
            v_norm = torch.norm(var, p='fro', dim=1, keepdim=True)
            return self.eps * torch.div(var, torch.maximum(v_norm, self.eps * torch.ones_like(v_norm)))

        elif self.norm == 'linf':
            """ In order to respect linf bound, v has to lie inside a l1-ball """
            return project_onto_l1_ball(var, eps=self.eps)

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
            m = torch.distributions.uniform.Uniform(torch.tensor([self.eps]), torch.tensor([2*self.eps]))
            var_raw = m.sample(sample_shape=[n_samples, self.n_atoms])[:, :, 0]
            return self.projection_v(var_raw)
