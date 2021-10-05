import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients
from torchvision.transforms import transforms


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

    return r_tot, loop_i