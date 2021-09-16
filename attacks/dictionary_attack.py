import os.path

import torch
import numpy as np
import torch.nn as nn
from tqdm import trange
import random
from scipy import stats
from torchattacks.attack import Attack
import time

from sda_logging import get_logger

logger = get_logger("Figs")

torch.random.manual_seed(123)
random.seed(123)
np.random.seed(123)


# ------------------------------------------------------------------------- #
# --------------------------------- TOOLS --------------------------------- #
# ------------------------------------------------------------------------- #


def dict_mult(dico, vec):
    [nchannel, x, y, z] = dico.shape
    # output = torch.zeros(3,x,y)
    output = torch.zeros(nchannel, x, y)
    for ii in range(z):
        output = output + dico[:, :, :, ii] * vec[ii]
    return output


def projection_dict_spectralBall(D):
    [ncolor, nx, ny, n_dict] = D.shape
    soft = nn.Softshrink(lambd=1)

    Dreshape = D.reshape(ncolor * nx * ny, n_dict)
    u, s, v = torch.svd(Dreshape)
    sproj = s - soft(s)
    Dproj = torch.mm(torch.mm(u, torch.diag(sproj)), v.t())

    return Dproj.reshape(ncolor, nx, ny, n_dict)


def clamp_image(image, max_val=1, min_val=0):
    return torch.clamp(image, max=max_val, min=min_val)


def projection_ortho(D):
    [ncolor, nx, ny, n_dict] = D.shape

    Dreshape = D.reshape(ncolor * nx * ny, n_dict)
    u, s, v = torch.svd(Dreshape)
    Dproj = torch.mm(u, v.t())

    return Dproj.reshape(ncolor, nx, ny, n_dict)


def project_onto_l1_ball(x, eps):
    """
    Compute Euclidean projection onto the L1 ball
    Pytorch version of Adrien Gaidon's work.
    Reference
    ----------
    [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
    """
    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1)
    arange = torch.arange(1, x.shape[1] + 1, device=x.device)
    rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
    theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - eps) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    x = mask * x + (1 - mask) * proj * torch.sign(x)
    return x.view(original_shape)


def constraint_dict(D, constr_set='l2ball', ortho=False):
    # Projection ||d||_2 = 1 (l2sphere) or ||d||_2 <= 1 (l2ball) or ||d||_1 <= (l1ball)
    n_atom = D.shape[-1]
    for ind in range(n_atom):
        Dnorm = torch.norm(D[:, :, :, ind], p='fro', keepdim=True)
        if constr_set == 'l2sphere':
            # Project onto the sphere
            D[:, :, :, ind] = torch.div(D[:, :, :, ind], Dnorm)
        elif constr_set == 'l2ball':
            # Project onto the ball
            D[:, :, :, ind] = torch.div(D[:, :, :, ind], torch.maximum(Dnorm, torch.ones_like(Dnorm)))
        else:
            D[:, :, :, ind] = project_onto_l1_ball(D[:, :, :, ind], eps=1)
    # Projection orthogonality
    if ortho:
        D = projection_ortho(D)
    return D


def fit_laplace(v):
    v_flat = v.detach().numpy().flatten()
    _, bg = stats.laplace.fit(v_flat)
    if bg == 0:
        return 1e-3
    else:
        return bg


def get_slices(N, step):
    """ Return the slices (of size step) of N values """
    l = range(N)
    return [list(l[ii:min(ii + step, N)]) for ii in range(0, N, step)]


def get_prox_l1(param):
    """ Soft thresholding operator """
    return torch.nn.Softshrink(lambd=param)


def get_target(img, label, targeted, classifier):
    """ if targeted=True, returns the second most probable target
    Otherwise, returns label
    """
    with torch.no_grad():
        if targeted:
            f_x = classifier(img)
            _, index = f_x.sort()
            return index[:, -2]
        else:
            return label


class QuickAttackDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.images[item], self.labels[item]


# ------------------------------------------------------------------------- #
# -------------------------- DICTIONARY LEARNING -------------------------- #
# ------------------------------------------------------------------------- #


def adil(dataset, model, targeted=True, niter=1e3, lambdaCoding=1., l2_fool=1., batchsize=None, step_size=.1, n_atom=10,
         dict_set='l2ball', device=torch.device("cpu"), dictionary=None):
    """
    ADiL: Adversarial Dictionary Learning
       - Full batch version
       - Backtracking implemented from [Bonettini et al., 2017]

     dataset: dataset to attack
     classifier: model to fool
     targeted: targeted attacks (2nd most probable) or untargeted attacks
     niter: number of iterations
     batchsize: to adjust depending on the memory available. It may speed up computations but has no impact on the learned attacks
     l2_fool: regularization parameter to penalize the l2 norm of attacks
     lambdaCoding: regularization parameter to penalize the l1 norm of coding vectors
     n_atom: number of dictionary elements
     dict_set: constraint on the dictionary elements (l2ball or l2sphere)
    """

    # Shape parameters
    if dictionary is None:
        n_img = len(dataset)
        x, _ = next(iter(dataset))
        nc, nx, ny = x.shape
    else:
        n_img = len(dataset)
        nc, nx, ny, _ = dictionary.shape
        print(n_img)

    # Line-search parameters
    delta = .5
    gamma = 1
    beta = .5
    lipschitz = .9 / step_size

    # Other parameters
    if batchsize is None: batchsize = n_img     # Batch-size
    coeff = 1. if targeted else -1.             # Targeted vs. Untargeted attacks
    indices = get_slices(n_img, batchsize)      # Slices of samples according to the batch-size

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=False)

    # Function
    criterion = nn.CrossEntropyLoss(reduction='sum')

    # Initialization of the dictionary D and coding vectors v
    if dictionary is None:
        d = torch.randn(3, nx, ny, n_atom, device=device)
        d = constraint_dict(d, constr_set=dict_set)
    else:
        d = dictionary
    v = torch.zeros(n_img, n_atom, device=device)

    # Initialization of intermediate variables
    d_old = torch.zeros_like(d)
    v_old = torch.zeros_like(v)
    grad_v_old = torch.zeros_like(v)
    grad_d_old = torch.zeros_like(d)
    loss_all = np.nan * np.ones(int(niter))
    loss_non_smooth_old = 0

    # Algorithm
    bar = trange(int(niter))
    flag_stop = False
    for iteration in bar:

        if not flag_stop:

            # Prepare computation graph
            v.detach()
            v.requires_grad = True
            if dictionary is None:
                d.detach()
                d.requires_grad = True

            # Loss computation
            loss_non_smooth = lambdaCoding * torch.sum(torch.abs(v))
            loss_smooth = 0
            for index, (x, y) in enumerate(data_loader):
                x, y = x.to(device=device), y.to(device=device)
                ind = indices[index]
                dv = torch.tensordot(v[ind], d, dims=([1], [3]))
                loss_smooth = loss_smooth + coeff * criterion(model(x + dv), get_target(x, y, targeted, model)) \
                              + .5 * l2_fool * torch.sum(dv ** 2)
            loss_full = loss_smooth + loss_non_smooth

            # Gradient computation
            loss_smooth.backward()
            grad_v = v.grad.data
            grad_d = d.grad.data if dictionary is None else torch.zeros(1)

            # Forward-Backward step with line-search
            with torch.no_grad():

                # Guess the Lipschitz constant
                if iteration > 1:
                    lipschitz = torch.sqrt(
                        torch.norm(grad_v - grad_v_old, 'fro') ** 2 + torch.norm(grad_d - grad_d_old, 'fro') ** 2)
                    lipschitz = lipschitz / torch.sqrt(
                        torch.norm(v - v_old, 'fro') ** 2 + torch.norm(d - d_old, 'fro') ** 2)

                # Memory
                d_old.copy_(d)
                v_old.copy_(v)
                grad_v_old.copy_(grad_v)
                grad_d_old.copy_(grad_d)
                loss_old = loss_full.data

                # Step-size
                step_size = .9 / lipschitz
                prox_l1 = get_prox_l1(param=step_size * lambdaCoding)

                # Update
                v = prox_l1(v - step_size * grad_v)
                if dictionary is None:
                    d = d - step_size * grad_d
                    d = constraint_dict(d, constr_set=dict_set)

                # added distance
                d_v = v - v_old
                d_d = d - d_old

                # First order approximation of the difference in loss
                h = torch.sum((d - d_old) * grad_d) + torch.sum((v - v_old) * grad_v) + .5 * (gamma / step_size) * (
                        torch.norm(d - d_old, 'fro') ** 2 + torch.norm(v - v_old, 'fro') ** 2) + loss_non_smooth \
                    - loss_non_smooth_old

                # Line-search
                flag = False
                index_i = 0
                while not flag:

                    new_v = v_old + (delta ** index_i) * d_v
                    new_d = d_old + (delta ** index_i) * d_d

                    # Compute the loss (we could cut the running by 2 if we stored the computation graph ..)
                    loss_non_smooth = lambdaCoding * torch.sum(torch.abs(new_v))    # replaced by new
                    loss_smooth = 0
                    for index, (x, y) in enumerate(data_loader):
                        x = x.to(device=device)
                        y = y.to(device=device)
                        ind = indices[index]
                        dv = torch.tensordot(new_v[ind], new_d, dims=([1], [3])) # replaced by new
                        loss_smooth = loss_smooth + coeff * criterion(model(x + dv), get_target(x, y, targeted, model)) \
                                      + .5 * l2_fool * torch.sum(dv ** 2)
                    loss_full = loss_smooth + loss_non_smooth

                    # Check the sufficient decrease condition
                    crit = loss_old + beta * (delta ** index_i) * h
                    if loss_full <= crit:
                        # Then its fine !
                        v = new_v
                        d = new_d
                        flag = True
                        loss_non_smooth_old = loss_non_smooth.data
                    else:
                        # Then we need to change index_i
                        index_i = index_i + 1
                        if index_i > 50:
                            # We have reached a stationary point
                            flag_stop = True
                            flag = True

                # Keep track of loss
                loss_all[iteration] = loss_full

    return d, v, loss_all


def sadil(dataset, model, targeted=True, nepochs=1e3, batchsize=1, lambdaCoding=1., l2_fool=1., stepsize=1.,
          n_atom=5, dict_set='l2ball', device=torch.device("cpu"), model_name=None):
    #
    # SADiL: Stochastic Adversarial Dictionary Learning
    #   - Stochastic version
    #   - Algorithm implemented based on SPRING [Driggs et al., 2021]
    #
    # dataset: dataset to attack
    # classifier: model to fool
    # nepochs: number of epochs
    # batchsize: number of images treated at each iteration

    # Parameters (number of images, dimension of images (3 channels))
    nimg = len(dataset)
    x, _ = next(iter(dataset))
    nc, nx, ny = x.shape

    # Dataloader
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=False)

    # Targeted vs. Untargeted attacks
    coeff = 1. if targeted else -1.

    def get_target(img, label, targeted, classifier):
        with torch.no_grad():
            if targeted:
                f_x = classifier(img)
                _, index = f_x.sort()
                return index[:, -2]
            else:
                return label

    # Slices of index
    indices = get_slices(nimg, batchsize)

    # Function
    criterion = nn.CrossEntropyLoss(reduction='sum')
    prox_l1 = get_prox_l1(param=stepsize * lambdaCoding)

    # Initialization
    D = torch.randn(3, nx, ny, n_atom, device=device)
    D = constraint_dict(D, constr_set=dict_set)
    v = torch.zeros(nimg, n_atom, device=device)

    def loss_all(loader, model, v):
        loss = 0
        for index, (x, y) in enumerate(loader):
            x, y = x.to(device=device), y.to(device=device)
            ind = indices[index]
            Dv = torch.tensordot(v[ind], D, dims=([1], [3]))
            loss += (coeff * criterion(model(x + Dv), get_target(x, y, targeted, model))
                     + .5 * l2_fool * torch.sum(Dv ** 2)).item()
        loss = loss + (lambdaCoding * torch.sum(torch.abs(v))).item()
        return loss

    loss = [loss_all(dataLoader, model, v)]

    # Algorithm (we can improve the visibility by defining an optimizer)
    bar = trange(int(nepochs))
    for _ in bar:

        for index, (x, y) in enumerate(dataLoader):
            x, y = x.to(device=device), y.to(device=device)
            ind = indices[index]

            # ---------- D-STEP ---------- #

            # Prepare computation graph
            v.detach()
            D.detach()
            D.requires_grad = True

            # Compute the loss
            Dv = torch.tensordot(v[ind], D, dims=([1], [3]))
            loss_smooth = coeff * criterion(model(x + Dv),
                                            get_target(x, y, targeted, model)) + .5 * l2_fool * torch.sum(Dv ** 2)

            # Gradient computation & memory
            loss_smooth.backward()
            grad_D = D.grad.data

            # Forward-Backward step
            with torch.no_grad():
                D = D - stepsize * grad_D
                D = constraint_dict(D, constr_set=dict_set)

            # ---------- V-STEP ---------- #

            # Prepare computation graph
            v.detach()
            D.detach()
            v.requires_grad = True

            # Compute the loss
            Dv = torch.tensordot(v[ind], D, dims=([1], [3]))
            loss_smooth = coeff * criterion(model(x + Dv),
                                            get_target(x, y, targeted, model)) + .5 * l2_fool * torch.sum(Dv ** 2)

            # Gradient computation
            loss_smooth.backward()
            grad_v = v.grad.data

            # Forward-Backward step
            with torch.no_grad():
                v[ind] = prox_l1(v[ind] - stepsize * grad_v[ind])

        loss.append(loss_all(dataLoader, model, v))
        if abs(loss[-1]-loss[-2]<1e-6):
            break

    path = f"/home/flavie2/Desktop/attack_learning/Sparse_Dictionary_Attack/dict_model_ImageNet/"
    model_file = f"ImageNet_{model_name}_lamCoding_{lambdaCoding}_lamFool_{l2_fool}_num_atom_" \
                 f"{n_atom}_nepoch_{nepochs}_{time.ctime(time.time())}.bin"

    torch.save([D, loss], os.path.join(path, model_file))

    return D, v, _

# ------------------------------------------------------------------------- #
# ------------------------- LEARN CODING VECTORS -------------------------- #
# ------------------------------------------------------------------------- #


def learn_coding_vectors(dataset, model, targeted=True, niter=1e3, lambdaCoding=1., l2_fool=1., batchsize=None, step_size=.1, n_atom=10,
         dict_set='l2ball', device=torch.device("cpu"), dictionary=None, verbose=False):

    # Shape parameters
    n_img = len(dataset)
    nc, nx, ny, _ = dictionary.shape

    # Line-search parameters
    delta = .5
    gamma = 1
    beta = .5
    lipschitz = .9 / step_size

    # Other parameters
    if batchsize is None: batchsize = n_img     # Batch-size
    coeff = 1. if targeted else -1.             # Targeted vs. Untargeted attacks
    indices = get_slices(n_img, batchsize)      # Slices of samples according to the batch-size

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=False)

    # Function
    criterion = nn.CrossEntropyLoss(reduction='sum')

    # Initialization of the dictionary D and coding vectors v
    d = dictionary
    v = torch.zeros(n_img, n_atom, device=device)

    # Initialization of intermediate variables
    v_old = torch.zeros_like(v)
    grad_v_old = torch.zeros_like(v)
    loss_all = np.nan * np.ones(int(niter))
    loss_non_smooth_old = 0

    # Algorithm
    bar = trange(int(niter)) if verbose else range(int(niter))
    flag_stop = False
    for iteration in bar:

        if not flag_stop:

            # Prepare computation graph
            v.detach()
            v.requires_grad = True

            # Loss computation
            loss_non_smooth = lambdaCoding * torch.sum(torch.abs(v))
            loss_smooth = 0
            for index, (x, y) in enumerate(data_loader):
                x, y = x.to(device=device), y.to(device=device)
                ind = indices[index]
                dv = torch.tensordot(v[ind], d, dims=([1], [3]))
                loss_smooth = loss_smooth + coeff * criterion(model(x + dv), get_target(x, y, targeted, model)) \
                              + .5 * l2_fool * torch.sum(dv ** 2)
            loss_full = loss_smooth + loss_non_smooth

            # Gradient computation
            loss_smooth.backward()
            grad_v = v.grad.data
            grad_d = d.grad.data if dictionary is None else torch.zeros(1)

            # Forward-Backward step with line-search
            with torch.no_grad():

                # Guess the Lipschitz constant
                if iteration > 1:
                    lipschitz = torch.norm(grad_v - grad_v_old, 'fro') / torch.norm(v - v_old, 'fro')

                # Memory
                v_old.copy_(v)
                grad_v_old.copy_(grad_v)
                loss_old = loss_full.data

                # Step-size
                step_size = .9 / lipschitz
                prox_l1 = get_prox_l1(param=step_size * lambdaCoding)

                # Update
                v = prox_l1(v - step_size * grad_v)
                if dictionary is None:
                    d = d - step_size * grad_d
                    d = constraint_dict(d, constr_set=dict_set)

                # added distance
                d_v = v - v_old

                # First order approximation of the difference in loss
                h = torch.sum((v - v_old) * grad_v) + .5 * (gamma / step_size) * (torch.norm(v - v_old, 'fro') ** 2) \
                    + loss_non_smooth - loss_non_smooth_old

                # Line-search
                flag = False
                index_i = 0
                while not flag:

                    new_v = v_old + (delta ** index_i) * d_v

                    # Compute the loss (we could cut the running by 2 if we stored the computation graph ..)
                    loss_non_smooth = lambdaCoding * torch.sum(torch.abs(new_v))    # replaced by new
                    loss_smooth = 0
                    for index, (x, y) in enumerate(data_loader):
                        x = x.to(device=device)
                        y = y.to(device=device)
                        ind = indices[index]
                        dv = torch.tensordot(new_v[ind], d, dims=([1], [3])) # replaced by new
                        loss_smooth = loss_smooth + coeff * criterion(model(x + dv), get_target(x, y, targeted, model)) \
                                      + .5 * l2_fool * torch.sum(dv ** 2)
                    loss_full = loss_smooth + loss_non_smooth

                    # Check the sufficient decrease condition
                    crit = loss_old + beta * (delta ** index_i) * h
                    if loss_full <= crit:
                        # Then its fine !
                        v = new_v
                        flag = True
                        loss_non_smooth_old = loss_non_smooth.data
                    else:
                        # Then we need to change index_i
                        index_i = index_i + 1
                        if index_i > 50:
                            # We have reached a stationary point
                            flag_stop = True
                            flag = True

                # Keep track of loss
                loss_all[iteration] = loss_full

    return v.detach()

# ------------------------------------------------------------------------- #
# ------------------------------ ADIL ATTACK ------------------------------ #
# ------------------------------------------------------------------------- #


class ADIL(Attack):
    """
    ADiL in the paper 'Adversarial Dictionary Learning'

    Arguments:
        model (nn.Module): model to attack.
        lambda_l1 (float): regularization parameter promoting sparsity of the coding vectors. (Default: 0.1)
        lambda_l2 (float): regularization parameter penalizing the l2-norm of the adversarial noise. (Default: 0.1)
        n_atoms (int): number of dictionary atoms. (Default: 10)
        version (string): 'deterministic' <--> ADiL. 'stochastic' <--> SADiL. (Default: 'deterministic')
        targeted (bool): if True, attacks target the second most probable label.
                         Otherwise it performs untargeted attacks. (Default: True)
        attack (string): behavior when attacking unseen examples. (Default: 'unsupervised')
                         'unsupervised' <--> sample the coding vectors
                         'supervised' <--> optimize over the coding vectors
        data_train (Dataset): dataset used to learn the dictionary.
        steps (float or int): number of steps to learn the dictionary
        trials (int): number of trials to find the best unsupervised attacks to unseen examples. (Default: 100)


    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    """

    def __init__(self, model, steps=1e2, lambda_l1=1e-1, lambda_l2=1e-1, version='deterministic', targeted=True,
                 attack='unsupervised', n_atoms=10, batch_size=1, data_train=None, step_size=.01, trials=100, device='cpu', model_name=None):
        super().__init__("ADIL", model)
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.n_atoms = n_atoms
        self.steps = steps
        self.batch_size = batch_size
        self.version = version
        self.targeted = targeted
        self.attack = attack
        self.dictionary = None
        self.scale = None
        self.trials = trials
        self.step_size = step_size
        self.device = device
        self.model_file = model_name

        if data_train is not None:
            self.learn_dictionary(dataset=data_train, model=model)

    def learn_dictionary(self, dataset, model):
        if self.version == 'deterministic':
            """ Theoretically grounded implementation """
            d, v, _ = adil(dataset=dataset, model=model.eval(), targeted=self.targeted, niter=self.steps,
                           lambdaCoding=self.lambda_l1, l2_fool=self.lambda_l2, batchsize=None,
                           step_size=self.step_size,
                           n_atom=self.n_atoms, dict_set='l2ball', device=self.device)
        else:
            """ Fast stochastic implementation """
            print('statistic attack learning, lambdaCoding={}, l2_fool={}, n_atoms={}'.format(self.lambda_l1, self.lambda_l2, self.n_atoms))
            d, v, _ = sadil(dataset=dataset, model=model.eval(), targeted=self.targeted, nepochs=self.steps,
                            lambdaCoding=self.lambda_l1, l2_fool=self.lambda_l2, batchsize=self.batch_size,
                            stepsize=self.step_size, n_atom=self.n_atoms, dict_set='l2ball', device=self.device, model_name=self.model_name)

        self.scale = fit_laplace(v.detach().cpu())
        self.dictionary = d.detach().to(torch.device("cpu"))

    def forward_unsupervised(self, images, labels):
        """ Unsupervised attack to unseen examples
        The method relies on sampling the coding vectors randomly according to some Laplace distribution
        """
        n_samples = images.shape[0]
        fooling_flag = torch.zeros(n_samples, dtype=torch.bool).to(device=self.device)
        mse_best_do_fool = np.inf * torch.ones(n_samples)
        mse_best_no_fool = np.inf * torch.ones(n_samples)
        adv_images_best = torch.zeros_like(images)

        laplace = torch.distributions.laplace.Laplace(loc=0, scale=self.scale)
        for _ in range(self.trials):

            # Sample adversarial images
            v = laplace.sample(sample_shape=[n_samples, self.n_atoms]).to(device=self.device)
            adv_images = torch.zeros_like(images)
            for ind in range(n_samples):
                adv_images[ind, :, :, :] = clamp_image(images[ind, :, :, :] + torch.tensordot(v[ind], self.dictionary.to(device=self.device),
                                                                                              dims=([0], [3])))
            adv_labels = self.model.eval()(adv_images).argmax(dim=1)
            pre_labels = self.model.eval()(images).argmax(dim=1)

            # Evaluate their performance
            fooling = ~torch.eq(pre_labels, adv_labels)
            mse = torch.sum((images - adv_images) ** 2, dim=[1, 2, 3])

            # Keep the best ones
            fooling_flag.copy_(torch.logical_or(fooling_flag, fooling))
            for ind in range(n_samples):
                if torch.logical_and(torch.all(fooling_flag[ind]), torch.all(fooling[ind])):
                    if mse[ind] < mse_best_do_fool[ind]:
                        mse_best_do_fool[ind] = mse[ind]
                        adv_images_best[ind] = adv_images[ind]
                elif torch.all(~fooling_flag[ind]):
                    if mse[ind] < mse_best_no_fool[ind]:
                        mse_best_no_fool[ind] = mse[ind]
                        adv_images_best[ind] = adv_images[ind]

        return adv_images_best

    def forward_supervised(self, images, labels):
        """ Unsupervised attack to unseen examples
            The method relies on the optimization over the coding vectors
            it might be speed up by sample-wise line-search
        """
        n_samples = images.shape[0]
        dataset = QuickAttackDataset(images=images, labels=labels)

        # Optimize over the coding vectors
        v = learn_coding_vectors(dataset=dataset, model=self.model.eval(), targeted=self.targeted, niter=self.steps,
                                 lambdaCoding=self.lambda_l1, l2_fool=self.lambda_l2, batchsize=None,
                                 step_size=self.step_size, n_atom=self.n_atoms, dict_set='l2ball', device=self.device,
                                 dictionary=self.dictionary.to(device=self.device))

        # Build the adversarial images
        adv_images = torch.zeros_like(images)
        for ind in range(n_samples):
            adv_images[ind, :, :, :] = clamp_image(images[ind, :, :, :]
                                                   + torch.tensordot(v[ind], self.dictionary.to(device=self.device),
                                                                     dims=([0], [3])))
        return adv_images.detach()

    def forward_supervised_slow(self, images, labels):
        """ Unsupervised attack to unseen examples
            The method relies on the optimization over the coding vectors
            slow version but working.
            To improve: provide a sample-wise line-search
        """
        n_samples = images.shape[0]
        adv_images = torch.zeros_like(images)
        for ind in range(n_samples):
            img = images[ind]
            img = img[None, :, :, :].to(self.device)
            label = torch.tensor([labels[ind]], device=self.device, dtype=torch.float)
            dataset = QuickAttackDataset(images=img, labels=label)
            v = learn_coding_vectors(dataset=dataset, model=self.model.eval(), targeted=self.targeted, niter=self.steps,
                                     lambdaCoding=self.lambda_l1, l2_fool=self.lambda_l2, batchsize=1,
                                     step_size=self.step_size, n_atom=self.n_atoms, dict_set='l2ball',
                                     device=self.device, dictionary=self.dictionary.to(device=self.device))
            adv = clamp_image(img + torch.tensordot(v, self.dictionary, dims=([1], [3]))).detach()
            adv_images[ind] = adv

        return adv_images

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # Check if the dictionary has been learned
        if self.dictionary is None:
            print('The adversarial dictionary has not been learned.')
            print('It is now being learned on the given dataset')
            dataset = QuickAttackDataset(images=images, labels=labels)
            self.learn_dictionary(dataset=dataset, model=self.model)

        if self.attack == 'supervised_slow':
            return self.forward_supervised_slow(images, labels)
        elif self.attack == 'supervised':
            return self.forward_supervised(images, labels)
        else:
            return self.forward_unsupervised(images, labels)


# ------------------------------------------------------------------------- #
# ---------------------------------- OLD ---------------------------------- #
# ------------------------------------------------------------------------- #


def make_adversarial_image(x, D, model, v=None, niter=1e3, l2_fool=1., lambdaCoding=1., stepsize=.1):
    # Learn the coding vectors 'v' given a dictionary 'D' in order to provide an adversary example 'x+Dv' which fool 'model'
    if v is None:
        v = torch.zeros(D.shape[3])
    # Function
    prox_l1 = torch.nn.Softshrink(lambd=stepsize * lambdaCoding)
    criterion = nn.CrossEntropyLoss()
    # Parameters
    delta = .5
    gamma = .9
    beta = .5
    # Initialization (to make sure it converges)
    # loss_all = np.nan*np.ones(int(niter))
    # Adversary targets
    f_x = model(x.unsqueeze(0))
    _, index = f_x.sort()
    c_a = index[0][-2]

    for _ in range(int(niter)):
        # Perform one step
        v = v.detach()
        v.requires_grad = True
        Dv = dict_mult(D, v)
        a = x + Dv

        # Computation of the gradient of the smooth part
        f_a = model(a)
        loss = criterion(f_a, c_a.unsqueeze(0)) + .5 * l2_fool * torch.sum(Dv ** 2)
        loss_nonsmooth_old = lambdaCoding * torch.sum(torch.abs(v))
        loss.backward()
        loss_old = loss.data + loss_nonsmooth_old
        grad = v.grad.clone().detach()

        # Forward-Backward step
        with torch.no_grad():
            v_old = v
            v = prox_l1(v - stepsize * grad)

            # added distance
            d_v = v - v_old

            # Linesearch
            flag = False
            index_i = 0

            while not flag:

                new_v = v_old + (delta ** (index_i)) * d_v

                # Compute the loss (we could cut the running by 2 if we stored the computation graph ..)
                loss_smooth = 0
                loss_nonsmooth = 0
                Dv = dict_mult(D, new_v)
                a = x + Dv
                loss_smooth = criterion(model(a), c_a.unsqueeze(0)) + .5 * l2_fool * torch.sum(Dv ** 2)
                loss_nonsmooth = lambdaCoding * torch.sum(torch.abs(new_v))
                loss_full = loss_smooth + loss_nonsmooth

                # Check the sufficient decrease condition
                h = torch.sum((v - v_old) * grad) + .5 * (gamma / stepsize) * (torch.norm(v - v_old, 'fro') ** 2)
                h = h + loss_nonsmooth - loss_nonsmooth_old
                crit = loss_old + beta * (delta ** (index_i)) * (h)

                if loss_full <= crit:
                    # Then its fine !
                    v = new_v
                    flag = True
                    # loss_all[iteration] = loss_full
                else:
                    # Then we need to change index_i
                    index_i = index_i + 1
                    if index_i > 100:
                        # We have reached a stationary point
                        flag_stop = True
                        flag = True
    return clamp_image(x + dict_mult(D, v)), dict_mult(D, v), v


def get_scale_coding_vector(x, D, model, lambdaCoding, l2_fool):
    _, _, v = make_adversarial_image(x, D, model, l2_fool=l2_fool, lambdaCoding=lambdaCoding)
    vflat = v.detach().numpy().flatten()
    _, scale = stats.laplace.fit(vflat)
    return scale


def sample_adversarial_image(x, D, model, attempt_before_rejection=10, noise_factor=1e0, scale=0.5):
    # Sample coding vector 'v' from a Laplacian distribution, so that 'x+Dv' provide an adversary example
    y_original = model(x).argmax()
    laplace_law = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([scale]))

    valid_adversary_rmse = []
    for _ in range(attempt_before_rejection):
        v_laplace = torch.stack([laplace_law.sample() for _ in range(D.shape[-1])])
        noise = dict_mult(D, v_laplace)
        x_fake = clamp_image(x + noise_factor * noise)
        y_fake = model(x_fake).argmax()
        if y_fake != y_original:
            rmse = torch.norm(noise) / torch.norm(x)
            valid_adversary_rmse.append((v_laplace, rmse))
    # No adversary found
    if len(valid_adversary_rmse) == 0:
        return x, None, torch.zeros(D.shape[-1])
    # else we look for the best adversary computed out of the attempt_before_rejection
    else:
        valid_adversary_rmse.sort(key=lambda x: x[1])
        (best_v, _) = valid_adversary_rmse[0]
    return clamp_image(x + dict_mult(D, best_v)), dict_mult(D, best_v), best_v


def get_ADiL_attack_unsupervised(D, v, samples=20):
    """ After running (S)ADiL, simply define :
    ADiL_attack_unsupervised = get_ADiL_attack_unsupervised(D,v)
    and then, you can attack any unseen example 'x' on the neural net 'classifier' as
    attack = ADiL_attack_unsupervised(x=x,classifier=classifier)
  """

    vflat = v.detach().numpy().flatten()
    ag, bg = stats.laplace.fit(vflat)
    K = D.shape[3]

    def attack(x, classifier=None, samples=None):
        v = np.random.laplace(loc=0, scale=bg, size=K)
        return clamp_image(x + dict_mult(D, v))

    def attack_loop(x, classifier, samples=samples):
        rmse = []
        fooling = []
        label = classifier(x).argmax().item()
        advs = []
        for _ in range(samples):
            adversary = attack(x)
            target = classifier(adversary).argmax().item()
            advs.append(adversary)
            fooling.append(label == target)
            rmse.append(torch.norm(x - adversary, 'fro').item())

        index = np.argsort(rmse)
        adv = advs[index[0]]
        for ii in index:
            if fooling[ii] == False:
                adv = advs[ii]
                break
        return adv

    return attack_loop if samples > 1 else attack
