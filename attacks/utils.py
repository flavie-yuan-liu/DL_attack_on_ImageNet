import torch
import numpy as np
from scipy import stats
from torchattacks.attack import Attack
from tqdm import trange
import torch.nn as nn


def dict_mult(dico, vec):
    [nchannel, x, y, z] = dico.shape
    output = torch.zeros(nchannel, x, y)
    for ii in range(z):
        output = output + dico[:, :, :, ii] * vec[ii]
    return output


def clamp_image(image, max_val=1, min_val=0):
    return torch.clamp(image, max=max_val, min=min_val)


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


def constraint_dict(d, constr_set='l2ball'):
    # Projection ||d||_2 = 1 (l2sphere) or ||d||_2 <= 1 (l2ball) or ||d||_1 <= (l1ball)
    n_atom = d.shape[-1]
    for ind in range(n_atom):
        d_norm = torch.norm(d[:, :, :, ind], p='fro', keepdim=True)
        if constr_set == 'l2sphere':
            # Project onto the sphere
            d[:, :, :, ind] = torch.div(d[:, :, :, ind], d_norm)
        elif constr_set == 'l2ball':
            # Project onto the ball
            d[:, :, :, ind] = torch.div(d[:, :, :, ind], torch.maximum(d_norm, torch.ones_like(d_norm)))
        else:
            d[:, :, :, ind] = project_onto_l1_ball(d[:, :, :, ind], eps=1)
    return d


def fit_laplace(v, dataset, model, device=torch.device('cpu')):
    mean = dict()
    scale = dict()
    versions = ['predictions_atoms', 'labels_atoms', 'atoms', 'none']
    for version in versions:
        mean_tmp, scale_tmp = fit_laplace_aux(v, model, dataset=dataset, conditioned=version, device=device)
        mean.update({version: mean_tmp})
        scale.update({version: scale_tmp})
    return mean, scale


def fit_laplace_aux(v, model, dataset=None, conditioned='labels_atoms', min_scale=1e-3, device=torch.device('cpu')):
    if conditioned == 'predictions_atoms':
        """ Fit a Laplace distribution Conditioned to the predictions & atoms """
        # get number of samples
        n_samples = len(dataset)

        # get number of classes (can be improved)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=n_samples)
        _, y = next(iter(data_loader))
        n_classes = len(torch.unique(y))

        v_stack = [np.empty(1) for _ in range(n_classes)]
        v = v.detach().numpy()
        for ind in range(n_samples):
            x, _ = dataset[ind]
            x = x.to(device=device)
            y = model(x[None, :, :, :]).detach().argmax().squeeze().cpu().data
            if len(v_stack[y]) == 1:
                v_stack[y] = v[ind]
            else:
                v_stack[y] = np.vstack((v_stack[y], v[ind].squeeze()))

        mean_all = [[] for _ in range(n_classes)]
        scale_all = [[] for _ in range(n_classes)]
        for ind in range(n_classes):
            mean, scale = fit_laplace_multivariate(v_stack[ind], min_scale=min_scale)
            mean_all[ind] = mean
            scale_all[ind] = scale

        return mean_all, scale_all
    if conditioned == 'labels_atoms':
        """ Fit a Laplace distribution Conditioned to the labels & atoms """
        # get number of samples
        n_samples = len(dataset)

        # get number of classes (can be improved)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=n_samples)
        _, y = next(iter(data_loader))
        n_classes = len(torch.unique(y))

        v_stack = [np.empty(1) for _ in range(n_classes)]
        v = v.detach().numpy()
        for ind in range(n_samples):
            _, y = dataset[ind]
            if len(v_stack[y]) == 1:
                v_stack[y] = v[ind]
            else:
                v_stack[y] = np.vstack((v_stack[y], v[ind]))

        mean_all = [[] for _ in range(n_classes)]
        scale_all = [[] for _ in range(n_classes)]
        for ind in range(n_classes):
            mean, scale = fit_laplace_multivariate(v_stack[ind], min_scale=min_scale)
            mean_all[ind] = mean
            scale_all[ind] = scale

        return mean_all, scale_all
    elif conditioned == 'atoms':
        """ Fit a Laplace distribution Conditioned to atoms """
        return fit_laplace_multivariate(v, min_scale=min_scale)
    else:
        v_flat = v.detach().numpy().flatten()
        ag, bg = stats.laplace.fit(v_flat)
        if bg < min_scale:
            return ag, min_scale
        else:
            return ag, bg


def fit_laplace_multivariate(v, min_scale):
    """ Fit a Laplace distribution Conditioned to the atoms """
    n_atoms = v.shape[1]
    mean = []
    scale = []
    for ind in range(n_atoms):
        ag, bg = stats.laplace.fit(v[:,ind])
        bg = min_scale if bg < min_scale else bg
        mean.append(ag)
        scale.append(bg)
    return mean, scale


def get_slices(n, step):
    """ Return the slices (of size step) of N values """
    n_range = range(n)
    return [list(n_range[ii:min(ii + step, n)]) for ii in range(0, n, step)]


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