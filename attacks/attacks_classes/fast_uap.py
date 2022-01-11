
from torchattacks.attack import Attack
from attacks.utils import *
import os
from copy import deepcopy
import torch
import torchattacks
import numpy as np
import copy
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from tqdm import trange, tqdm
torch.set_default_tensor_type(torch.DoubleTensor)


class FastUAP(Attack):
    """ Fast UAP: Fast Universal Adversarial Perturbation from  [Jiazhu Dai and Le Shu, 2021]
     Pytorch implementation aiming to reproduce the TensorFlow version developed in
     https://github.com/FallLeaf0914/fast-universal
     model: torch model to fool.
     data_train: dataset on which the attack is learned
     steps: number of maximum steps. (default: 100)
     fooling_rate: fooling rate to reach on data_train. (default: 1)
     norm: norm of the ball ('l2' or 'linf') constraining the adversarial noise. (default: 'linf')
     eps: radius of the ball on the adversarial noise. (default: inf)
     overshoot: overshoot parameter used in DeepFool. (default: 0.02)
     steps_deepfool: number of steps used in DeepFool. (default: 50)
     source: https://doi.org/10.1016/j.neucom.2020.09.052
     """

    def __init__(self, model, steps=10, fooling_rate=0.98, eps=np.inf, norm='linf', data_train=None, data_val=None,
                 overshoot=0.02, steps_deepfool=50, model_name=None):
        super().__init__('FastUAP', model)
        self.steps = steps
        self.fooling_rate = fooling_rate
        self.eps = eps
        self.norm = norm
        self.overshoot = overshoot
        self.steps_deepfool = steps_deepfool

        root = 'dict_model_ImageNet_version_constrained/{}_fast_uap/trained_dicts'.format(model_name)
        self.model_name = os.path.join(root, 'FastUAP_model')

        if not os.path.exists(self.model_name):
            self.learn_attack(dataset=data_train, val_data=data_val)

    def project(self, attack):
        if self.norm.lower() == 'l2':
            attack_norm = torch.norm(attack, p='fro', keepdim=False)
            if attack_norm > self.eps:
                return self.eps * attack / attack_norm
            else:
                return attack
        elif self.norm.lower() == 'linf':
            return torch.clamp(attack, min=-self.eps, max=self.eps)

    def learn_attack(self, dataset, val_data):

        # data loader
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        # solvers
        # deepfool = DeepFool(model=self.model, overshoot=self.overshoot, steps=self.steps_deepfool)
        #vdeepfoolcos = DeepFoolCosinus(model=self.model, overshoot=self.overshoot, steps=self.steps_deepfool)

        # variable
        x, _ = dataset[0]
        attack = torch.zeros_like(x, device=self.device)
        fooling_rate = []

        for iteration in trange(int(self.steps)):

            for x, _ in tqdm(data_loader):

                x = x[0].to(device=self.device)
                pert_image = x + attack

                # If the attack of 'x' does not fool the model ...
                if self.model(pert_image).argmax() == self.model(x).argmax():
                    iter, delta_attack = deepfool(pert_image, self.model)
                    # if torch.all(attack.eq(torch.zeros_like(attack))):
                    #     # if the attack is zero, compute the perturbation  that has the smallest magnitude and
                    #     # fools the model. The authors suggest to resort to DeepFool
                    #     delta_attack = deepfool(x, y) - x
                    # else:
                    #     # Otherwise, compute the perturbation that has similar orientation to the current
                    #     # perturbation and that fools the model.
                    #     delta_attack = deepfoolcos(x, y, attack) - x
                    if iter < self.steps_deepfool-1:
                        attack = self.project(attack + torch.from_numpy(delta_attack).to(self.device))

            fooling_rate.append(compute_fooling_rate(dataset=val_data, attack=attack,  model=self.model, device=self.device))
            print(fooling_rate[-1])
            if fooling_rate[-1] >= self.fooling_rate:
                break
        torch.save([attack, fooling_rate], self.model_name)

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)

        # Check if the Fast-UAP attack has been learned
        if not os.path.exists(self.model_name):
            print('The Fast-UAP attack has not been learned. It is now being learned on the given dataset.')
            dataset = QuickAttackDataset(images=images, labels=labels)
            self.learn_attack(dataset=dataset, model=self.model)
        else:
            attack, _ = torch.load(self.model_name)

        return torch.clamp(images + attack, min=0, max=1)


class DeepFoolCosinus(Attack):
    r"""
    'DeepFoolCosinus: variant of Deep Fool where, given some perturbation eps_old, find
            arg max_eps cosinus(eps, eps_old) s.t. image + eps + eps_old fools the classifier
    """

    def __init__(self, model, steps=50, overshoot=0.02):
        super().__init__("DeepFoolCosinus", model)
        self.steps = steps
        self.overshoot = overshoot
        self._supported_mode = ['default']

    def forward(self, images, labels, attack_init, return_target_labels=False):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        batch_size = len(images)
        correct = torch.tensor([True] * batch_size)
        target_labels = labels.clone().detach().to(self.device)
        curr_steps = 0

        adv_images = []
        for idx in range(batch_size):
            image = images[idx:idx + 1].clone().detach() + attack_init.clone().detach()
            adv_images.append(image)

        while (True in correct) and (curr_steps < self.steps):
            for idx in range(batch_size):
                if not correct[idx]: continue
                early_stop, pre, adv_image = self._forward_indiv(adv_images[idx], labels[idx], attack_init)
                adv_images[idx] = adv_image
                target_labels[idx] = pre
                if early_stop:
                    correct[idx] = False
            curr_steps += 1

        adv_images = torch.cat(adv_images).detach()

        if return_target_labels:
            return adv_images, target_labels

        return adv_images

    def _forward_indiv(self, image, label, attack_init):
        image.requires_grad = True
        fs = self.model(image)[0]
        _, pre = torch.sort(fs, dim=0, descending=True)
        # _, pre = torch.max(fs, dim=0)
        if pre[0] != label:
            return True, pre[0], image

        ws = self._construct_jacobian(fs[pre[:10]], image)
        image = image.detach()

        # f_0 = fs[label]
        # w_0 = ws[label]
        f_0 = fs[label]
        w_0 = ws[0]

        wrong_classes = [i for i in range(10) if pre[i] != label]
        f_k = fs[pre[wrong_classes]]
        w_k = ws[wrong_classes]
        #
        # wrong_classes = [i for i in range(len(fs)) if i != label]
        # f_k = fs[wrong_classes]
        # w_k = ws[wrong_classes]

        f_prime = f_k - f_0
        w_prime = w_k - w_0

        cosinus_best = - np.inf
        delta_best = 0
        target_label = 0

        for kk in range(len(wrong_classes)):

            delta = (torch.abs(f_prime[kk]) * w_prime[kk, :, :, :, :]\
                     / (torch.norm(w_prime[kk, :, :, :, :], p=2) ** 2))

            cosinus = torch.tensordot(nn.Flatten()(delta), nn.Flatten()(attack_init)) / \
                      (torch.norm(nn.Flatten()(delta), p=2) * torch.norm(nn.Flatten()(attack_init), p=2))

            if cosinus > cosinus_best:
                cosinus_best = cosinus
                delta_best = delta
                target_label = pre[kk]

        adv_image = image + (1 + self.overshoot) * delta_best
        adv_image = torch.clamp(adv_image, min=0, max=1).detach()
        return False, target_label, adv_image,

    def _construct_jacobian(self, y, x):
        x_grads = []
        for idx, y_element in enumerate(y):
            if x.grad is not None:
                x.grad.zero_()
            y_element.backward(retain_graph=(False or idx + 1 < len(y)))
            x_grads.append(x.grad.clone().detach())
        return torch.stack(x_grads).reshape(*y.shape, *x.shape)


# class DeepFool(Attack):
#     r"""
#     'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
#     [https://arxiv.org/abs/1511.04599]
#
#     Distance Measure : L2
#
#     Arguments:
#         model (nn.Module): model to attack.
#         steps (int): number of steps. (Default: 50)
#         overshoot (float): parameter for enhancing the noise. (Default: 0.02)
#
#     Shape:
#         - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
#         - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
#         - output: :math:`(N, C, H, W)`.
#
#     Examples::
#         >>> attack = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
#         >>> adv_images = attack(images, labels)
#
#     """
#     def __init__(self, model, steps=50, overshoot=0.02):
#         super().__init__("DeepFool", model)
#         self.steps = steps
#         self.overshoot = overshoot
#         self._supported_mode = ['default']
#
#     def forward(self, image, label, return_target_labels=False):
#         r"""
#         Overridden.
#         """
#         image = image.clone().detach().to(self.device)
#         label = label.clone().detach().to(self.device)
#         pert_image = image.detach().clone()
#
#         curr_steps = 0
#         perturbation = torch.zeros_like(image)
#
#         while (self.model(pert_image).argmax(dim=-1) == label) and (curr_steps < self.steps):
#             pert = self._forward_indiv(pert_image, label)
#             perturbation += pert
#             pert_image = pert_image+perturbation
#             curr_steps += 1
#
#         return curr_steps, perturbation
#
#     def _forward_indiv(self, image, label):
#         image = torch.autograd.Variable(image, requires_grad=True)
#         fs = self.model(image)[0]
#         _, pre = torch.sort(fs, dim=0, descending=True)
#         # _, pre = torch.max(fs, dim=0)
#         if pre[0] != label:
#             return (True, pre[0], image)
#
#         ws = self._construct_jacobian(fs[pre[:10]], image)
#         image = image.detach()
#
#         f_0 = fs[label]
#         w_0 = ws[0]
#
#         wrong_classes = [i for i in range(10) if pre[i] != label]
#         f_k = fs[pre[wrong_classes]]
#         w_k = ws[wrong_classes]
#
#         f_prime = f_k - f_0
#         w_prime = w_k - w_0
#         value = torch.abs(f_prime) \
#                 / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
#         _, hat_L = torch.min(value, 0)
#
#         delta = (torch.abs(f_prime[hat_L])*w_prime[hat_L] \
#                  / (torch.norm(w_prime[hat_L], p=2)**2))
#
#         pert = (1+self.overshoot)*delta
#         return pert
#
#     # https://stackoverflow.com/questions/63096122/pytorch-is-it-possible-to-differentiate-a-matrix
#     # torch.autograd.functional.jacobian is only for torch >= 1.5.1
#     def _construct_jacobian(self, y, x):
#         x_grads = []
#         for idx, y_element in enumerate(y):
#             if x.grad is not None:
#                 x.grad.zero_()
#             y_element.backward(retain_graph=(False or idx+1 < len(y)))
#             x_grads.append(x.grad.clone().detach())
#         return torch.stack(x_grads).reshape(*y.shape, *x.shape)


def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=10):
    """
    :param image: Image of size 3*H*W
    :param net: network (input: images, output: values of activation **BEFORE** softmax).
    :param num_class:
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :param max_iter:
    :return:minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    # net.zero_grad()
    f_image = net(Variable(image, requires_grad=True)).data.cpu().numpy().flatten()
    I = f_image.argsort()[::-1]
    I = I[0:num_classes] # get the top n possible class index
    label = I[0] # extract the top 1 class label

    input_shape = image.detach().cpu().numpy().shape # original image
    pert_image = copy.deepcopy(image) # initialize the perturbed image
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image, requires_grad=True)
    # net.zero_grad()
    fs = net(x)
    k_i = label

    while k_i == label and loop_i < max_iter:
        pert = np.inf
        fs[0,I[0]].backward(retain_graph=True)  #
        grad_orig = x.grad.data.cpu().numpy().copy()  # original grad
        for k in range(1, num_classes):
            zero_gradients(x)
            fs[0,I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()  # current grad

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0,I[k]] - fs[0,I[0]]).data.cpu().numpy()

            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)  #
        r_tot = np.float32(r_tot + r_i)  # r_total
        pert_image = image + (1+overshoot) * torch.from_numpy(r_tot).cuda()
        # pert_image = transform_Img(torch.clamp(transform_Img(pert_image), min=0, max=1))

        x = Variable(pert_image, requires_grad=True)
        fs = net(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())
        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    return loop_i, r_tot