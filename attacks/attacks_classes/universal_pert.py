import numpy as np
from deepfool import deepfool
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
from torch.autograd import Variable
from torchvision.transforms import transforms


def proj_lp(v, xi, p):

    # Project on the lp ball centered at 0 and of radius xi
    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v


def universal_perturbation(dataset,
                           valset,
                           f,
                           delta=0.02,
                           max_iter_uni=np.inf,
                           xi=20/255.0,
                           p=np.inf,
                           num_classes=10,
                           overshoot=0.02,
                           max_iter_df=10):
    """
    :param dataset: Images of size MxHxWxC (M: number of images)
    :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
    :param grads: gradient functions with respect to input (as many gradients as classes).
    :param delta: controls the desired fooling rate (default = 80% fooling rate)
    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)
    :param xi: controls the l_p magnitude of the perturbation (default = 10)
    :param p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)
    :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :param max_iter_df: maximum number of iterations for deepfool (default = 10)
    :return: the universal perturbation.
    """
    print('p =', p, xi)
    v = 0
    fooling_rate = 0.0
    best_fooling = 0.0
    num_images = 50000   # The length of testing data
    n_iter = 0

    while fooling_rate < 1-delta and n_iter < 100:
        # Shuffle the dataset
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)

        # Go through the data set and compute the perturbation increments sequentially
        k = 0
        f.cuda()
        for cur_img, _ in tqdm(data_loader):
            k += 1
            cur_img = cur_img.cuda()
            per = cur_img + torch.tensor(v).cuda()
            per = Variable(per, requires_grad=True)
            if int(f(cur_img).argmax()) == int(f(per).argmax()):
                # Compute adversarial perturbation
                f.zero_grad()
                dr, iter = deepfool(per, f, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)
                # print('dr = ', abs(dr).max())

                # Make sure it converged...
                if iter < max_iter_df-1:
                    v = v + dr
                    v = proj_lp(v, xi, p)



        # Perturb the dataset with computed perturbation
        # dataset_perturbed = dataset + v

        batch_size = 128
        est_labels_orig = torch.zeros(batch_size).cuda()
        est_labels_pert = torch.zeros(batch_size).cuda()

        # Compute the estimated labels in batches
        ii = 0
        with torch.no_grad():
            for img_batch, _ in tqdm(valset):
                # m = (ii * batch_size)
                # M = min((ii + 1) * batch_size, num_images)
                img_batch = img_batch.cuda()
                per_img_batch = img_batch + torch.tensor(v).cuda()
                # ii += 1
                # print(img_batch.shape)
                # print(m, M)
                est_labels_orig = torch.argmax(f(img_batch), dim=1)
                est_labels_pert = torch.argmax(f(per_img_batch), dim=1)
                break

            # Compute the fooling rate
            fooling_rate = torch.sum(est_labels_pert != est_labels_orig).float() / batch_size
            print(torch.sum(est_labels_pert != est_labels_orig).float())
            print('FOOLING RATE = ', fooling_rate)
            if fooling_rate > best_fooling:
                best_fooling = fooling_rate
            print('Best Fooling Rate = ', best_fooling)
            n_iter += 1
            print(n_iter)

    pertbation_name = 'Test-{:.2f}-{:.2f}.npy'.format(abs(v).max(), fooling_rate*100)
    np.save(pertbation_name, v)

    return v