import torch
import numpy as np
import torch.nn as nn
from tqdm import trange
from torchvision.transforms import transforms
import random

from tqdm import tqdm

device = torch.device('cuda')

torch.random.manual_seed(123)
random.seed(123)
np.random.seed(123)


def clamp_image(image, min=0, max=1):
    return torch.clamp(image, min, max)


def transform_Img(img):
    '''

    Parameters
    ----------
    img: input image

    Returns
    -------
    normalized image
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return normalize(img)


def transforms_inverse(img):
    '''

    Parameters
    ----------
    img: input image

    Returns
    -------
    do inverse normalization of the given image to transform image in the original space
    (with pixel value bound in [0, 255] or [0, 1])
    '''
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    img = inv_normalize(img)
    return img


def dict_mult(Dico, vec):
    [channel, x, y, z] = Dico.shape
    output = torch.zeros(channel, x, y, device=device)
    for ii in range(z):
        output = output + Dico[:, :, :, ii] * vec[ii]
    return output


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
    num_dict = D.shape[-1]
    for ind in range(num_dict):
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


def learn_dictionary_full_backtrack_vmilan(x, classifier, **kwargs):

    # SparseCoding attack which take into account the full batch at each iteration
    # The learning algorithm is a variant of "Direct Optimization of the Dictionary LearningProblem" by A. Rakotomamonjy
    # The backtracking VMILAN implemented comes from "On the convergence of a linesearch based proximal-gradient method for nonconvex optimization" by Bonettini et al.
    #
    # x: image to attack
    # classifier: model to fool
    niter = kwargs.get('niter', 1e3)               # number of iteration
    lambdaReg = kwargs.get('lambdaReg', 1.)         # coefficient of the regularization term
    lambdaCoding = kwargs.get('lambdaCoding', 1.)  # coefficient of the sparsity of v
    stepsize = kwargs.get('stepsize', 1.)          # step size for gradient descent
    num_dict = kwargs.get('num_dict', 8)           # range of dictionary
    dict_set = kwargs.get('dict_set', 'l2ball')    # D normalization method
    dict_ortho = kwargs.get('dict_ortho', False)   # if D is orthogonally constrained

    # Parameters
    nimg, channel, nx, ny = x.shape

    # initialization of D and v
    if 'D' in kwargs:
        D = kwargs['D']
    else:
        D = torch.randn(channel, nx, ny, num_dict, device=device)
        D = constraint_dict(D, constr_set=dict_set, ortho=dict_ortho)

    if 'v' in kwargs:
        v = kwargs['v']
    else:
        v = torch.zeros(nimg, num_dict, device=device)

    Lipschitz = 1/(stepsize)
    delta = .5
    gamma = 1
    beta = .5

    # Function
    criterion = nn.CrossEntropyLoss(reduction='sum')
    def get_prox_l1(param_stepsize):
      return torch.nn.Softshrink(lambd=param_stepsize*lambdaCoding)


    # Adversary targets
    c_a = classifier(x).sort().indices[:, -2]

    # Initialization
    loss_all = np.nan*np.ones(int(niter))
    grad_v = 0
    grad_D = 0
    v_old = v
    D_old = D
    loss_old = np.inf
    loss_smooth_old = np.inf

    # Algorithm (we can improve the visibility by defining an optimizer)
    # bar = trange(int(niter))
    flag_stop = False
    for iteration in range(int(niter)):
      if not flag_stop:

        # Prepare computation graph
        v.detach()
        D.detach()
        v.requires_grad=True
        D.requires_grad = True
        
        # Compute the loss
        def compute_loss(data, dict, s, target):
            Dv = torch.stack([dict_mult(dict, v_i) for v_i in s])
            a = data + Dv
            loss_smooth = criterion(classifier(a), target) + 0.5*lambdaReg*torch.sum(Dv**2)
            return loss_smooth

        loss_smooth = compute_loss(x, D, v, c_a)
        loss_nonsmooth = lambdaCoding*torch.sum(torch.abs(v)).item()
        loss_full = loss_smooth.item() + loss_nonsmooth

        # Gradient computation & memory
        loss_smooth.backward()
        grad_v_old = grad_v
        grad_D_old = grad_D
        grad_v = v.grad.data
        grad_D = D.grad.data
        loss_old = loss_full
        loss_nonsmooth_old = loss_nonsmooth

        loss_smooth.detach().cpu()
        del loss_smooth
        torch.cuda.empty_cache()

        # Forward-Backward step (with backtracking)
        with torch.no_grad():

          # Guess the Lipschitz constant
          if iteration > 1:
            Lipschitz = torch.sqrt(torch.norm(grad_v - grad_v_old,'fro')**2 + torch.norm(grad_D - grad_D_old,'fro')**2)
            Lipschitz = Lipschitz / torch.sqrt(torch.norm(v-v_old,'fro')**2 + torch.norm(D-D_old,'fro')**2)
            Lipschitz = np.minimum(Lipschitz.cpu().detach().numpy(), nimg) #Heuristic

          # Stepsize
          stepsize = .9/Lipschitz
          prox_l1 = get_prox_l1(param_stepsize=stepsize)

          # Update
          for id in range(nimg):
            v[id] = prox_l1(v[id] - stepsize*grad_v[id])
          D = D - stepsize*grad_D
          D = constraint_dict(D, constr_set=dict_set, ortho=dict_ortho)

          # added distance
          d_v = v - v_old
          d_D = D - D_old

          h = torch.sum((D - D_old) * grad_D) + torch.sum((v - v_old) * grad_v) + .5 * (gamma / stepsize) * (
                      torch.norm(D - D_old, 'fro') ** 2 + torch.norm(v - v_old, 'fro') ** 2)
          h = h.item()

          # Linesearch
          flag = False
          index_i = 0
          while not flag:

            new_v = v_old + (delta**(index_i))*d_v
            new_D = D_old + (delta**(index_i))*d_D

            # Compute the loss (we could cut the running by 2 if we stored the computation graph ..)
            loss_smooth = compute_loss(x, new_D, new_v, c_a)
            loss_smooth = loss_smooth.item()
            loss_nonsmooth = lambdaCoding*torch.sum(torch.abs(new_v)).item()
            loss_full = loss_smooth + loss_nonsmooth

            # Check the sufficient decrease condition
            h = h + loss_nonsmooth - loss_nonsmooth_old                                                                                                 
            crit = loss_old + beta*(delta**(index_i))*( h )
            if loss_full <= crit:
              # Then its fine !
              v = new_v
              D = new_D
              # Memory
              D_old = D
              v_old = v
              flag=True

            else:
              # Then we need to change index_i
              index_i = index_i + 1
              if index_i > 20:
                # We have reached a stationary point
                flag_stop = True
                flag = True

          # Keep track of loss
          loss_all[iteration] = loss_full
        

    return D, v, loss_all
        
        
# def sparseCoding_attack_spring(x, classifier, niter=1e3, batchsize=1, lambdaReg=1., lambdaCoding=1., stepsize=1, num_dict=5, dict_set='l2ball', dict_ortho=False, linesearch=True):
def sparseCoding_attack_spring(x, classifier, **kwargs):
    #
    # SparseCoding attack which take into account mini batch at each iteration
    # The algorithm implemented is SPRING (Driggs et al., 2021)
    # A heuristic Armijo-like linesearch strategy is used if linesearch=True
    #
    # x: image to attack
    # classifier: model to fool

    # get all the argments and if they are not given, setting them to default ones
    niter = kwargs.get('niter', 1e3)               # number of iteration
    batchsize = kwargs.get('batchsize', 1)         # specified batch size
    lambdaReg = kwargs.get('lambdaReg', 1.)         # coefficient of the regularization term
    lambdaCoding = kwargs.get('lambdaCoding', 1.)  # coefficient of the sparsity of v
    stepsize = kwargs.get('stepsize', 1.)          # step size for gradient descent
    num_dict = kwargs.get('num_dict', 8)           # range of dictionary
    dict_set = kwargs.get('dict_set', 'l2ball')    # D normalization method
    dict_ortho = kwargs.get('dict_ortho', False)   # if D is orthogonally constrained
    linesearch = kwargs.get('linesearch', True)    # if back tracking applied

    # get data size
    nimg, channel, nx, ny = x.shape

    # initialization of D and v
    if 'D' in kwargs:
        D = kwargs['D']
    else:
        D = torch.randn(channel, nx, ny, num_dict, device=device)
        D = constraint_dict(D, constr_set=dict_set, ortho=dict_ortho)

    if 'v' in kwargs:
        v = kwargs['v']
    else:
        v = torch.zeros(nimg, num_dict, device=device)

    delta = .5
    gamma = 1
    beta = .5

    # Error Function
    criterion = nn.CrossEntropyLoss()
    def get_prox_l1(param_stepsize):
      return torch.nn.Softshrink(lambd=param_stepsize*lambdaCoding)

    # Soft Thresholding Func
    prox_l1 = get_prox_l1(param_stepsize=stepsize)

    # Adversary targets
    c_a = classifier(x).sort().indices[:, -2]

    loss_all = np.nan*np.ones(int(niter))
    loss_full = np.zeros(int(nimg))
    for id in range(nimg):
      loss_full[id] = criterion(classifier(x[id].unsqueeze(0)), c_a[id].unsqueeze(0))
    index_i_old = 0

    # Algorithm (we can improve the visibility by defining an optimizer)
    bar = trange(int(niter))
    flag_stop = False
    for iteration in bar:

      if not flag_stop:

        # sample mini batch
        batch = np.random.randint(nimg, size=batchsize)

        flag = False
        D_old = D.detach()
        v_old = v.detach()
        index_i = index_i_old

        while not flag:

          ## D-STEP ##

          # Prepare computation graph
          v.detach()
          D.detach()
          D.requires_grad = True
          
          # Compute the loss
          def compute_loss(data, dict, s, obj, index):
              Dv = torch.stack([dict_mult(dict, s_i) for s_i in s[index]])
              a = data[index] + Dv
              return criterion(classifier(a), obj[index]) + 0.5*lambdaReg*torch.sum(Dv**2)

          loss_smooth = compute_loss(x, D, v, c_a, batch)

          # Gradient computation & memory
          loss_smooth.backward()
          grad_D = D.grad.data

          # Forward-Backward step
          with torch.no_grad():
            D = D - stepsize*grad_D
            D = constraint_dict(D, constr_set=dict_set, ortho=dict_ortho)


          ## V-STEP ##

          # Prepare computation graph
          v.detach()
          D.detach()
          v.requires_grad=True
          
          # Compute the loss
          loss_smooth = compute_loss(x, D, v, c_a, batch)

          # Gradient computation
          loss_smooth.backward()
          grad_v = v.grad.data

          # Forward-Backward step (with backtracking)
          with torch.no_grad():
            for id in batch:
              v[id] = prox_l1(v[id] - stepsize*grad_v[id])
        

          ## END STEP ##
          with torch.no_grad():

            ## Loss ##
            # Here loss_prev and loss_now are used to compare the loss improvement on the selected batch

            loss_tmp = loss_full
            loss_prev = 0
            loss_now = 0
            for id in batch:
               Dv_id = dict_mult(D,v[id])
               a = x[id] + Dv_id
               loss_tmp[id] = criterion(classifier(a), c_a[id].unsqueeze(0)) + .5*lambdaReg*torch.sum(Dv_id**2) + lambdaCoding*torch.sum(torch.abs(v[id]))
               loss_prev = loss_prev + loss_full[id]
               loss_now  = loss_now + loss_tmp[id]


            if linesearch:
          
              # Check the sufficient decrease condition
              h = torch.sum((D-D_old)*grad_D) + .5*(gamma/stepsize)*(torch.norm(D-D_old,'fro'))**2
              crit = loss_prev + beta*(delta**(index_i))*( h )

              if loss_now <= crit:
                loss_full = loss_tmp
                loss_all[iteration] = loss_full.mean()
                index_i_old = np.maximum(0,index_i-2)
                flag = True
              else:
                index_i = index_i + 1
                if index_i > 50:
                  flag_stop = True
                D = D_old
                v = v_old
            else:
              flag = True
              loss_all[iteration] = loss_full.mean()

    return D, v, loss_all


def dict_training(dataloader, classify, filename, **kwargs):
    '''
    # large scale training by updating D and v batch after batch
    Problem for solving: argmin_{D, v} sum( Loss(classifier(x_i), classifier(x_i+Dv_i))+.5*lambdaReg|Dv_i|_F^2 + lambdaCoding|v_i|_1, (1) )
    Parameters
    ----------
    dataloader: dataloader of train dataset
    classify: model for attacking
    filename: full path of the result file saving the learned D and losses
    kwargs:

    Returns
    -------
    D for adversarial attack generation and losses of training, and save them in the given file
    '''

    method = kwargs.get('method', None)  # method for updating D and v, statistic or full version(default)
    niter = kwargs.get('niter', 1e3)  # max iteration number
    lambdaReg = kwargs.get('lambdaReg', 1.)
    lambdaCoding = kwargs.get('lambdaCoding', 1.)
    stepsize = kwargs.get('stepsize', 1.)
    num_dict = kwargs.get('num_dict', 8)  # number of atoms of dictionary
    dict_set = kwargs.get('dict_set', 'l2ball')
    dict_ortho = kwargs.get('dict_ortho', False)




    # get data size
    ndata = len(dataloader.dataset)
    channel, nx, ny = 3, 224, 224

    # initialization
    D = torch.randn(channel, nx, ny, num_dict, device=device)
    D = constraint_dict(D, constr_set=dict_set, ortho=dict_ortho)
    v = torch.zeros(ndata, num_dict, device=device)
    losses = []

    for i in tqdm(range(int(niter))):
        losses_i = []
        start = 0
        for x, _ in dataloader:
            end = start+x.shape[0]
            x = x.to(device)
            if method == 'statistics':
                D, v[start:end], losses_x = sparseCoding_attack_spring(x, classify, niter=1, D=D, v=v[start:end],
                                                                   lamdaReg=lambdaReg, lambdaCoding=lambdaCoding, stepsize=stepsize)
            else:
                D, v[start:end], losses_x = learn_dictionary_full_backtrack_vmilan(x, classify, niter=100, D=D, v=v[start:end], lambdaReg=lambdaReg,
                                                       lambdaCoding=lambdaCoding, stepsize=stepsize, num_dict=8)
            start += x.shape[0]
            losses_i.append(losses_x[-1])
        losses.append(losses_i)

    torch.save([D, losses], filename)


def make_adversarial_image(x, D, classifier, v=None, niter=1e3, l2_fool=1., l1_sparse=1., stepsize=.1):
    # Learn the coding vectors 'v' given a dictionary 'D' in order to provide an adversary example 'x+Dv' which fool 'classifier'
    if v is None:
        v = torch.zeros(D.shape[3])

    # Function
    prox_l1 = torch.nn.Softshrink(lambd=stepsize * l1_sparse)
    criterion = nn.CrossEntropyLoss()

    # Parameters
    delta = .5
    gamma = .9
    beta = .5

    # Initialization (to make sure it converges)
    # Adversary targets
    f_x = classifier(x.unsqueeze(0))
    _, index = f_x.sort()
    c_a = index[0][-2]

    for i in range(int(niter)):
        # Perform one step
        v = v.detach()
        v.requires_grad = True
        Dv = dict_mult(D, v)
        a = x + Dv

        # Computation of the gradient of the smooth part
        f_a = classifier(a.unsqueeze(0))
        loss = criterion(f_a, c_a.unsqueeze(0)) + .5 * l2_fool * torch.sum(Dv ** 2)
        loss_nonsmooth_old = l1_sparse*torch.sum(torch.abs(v)).item()
        loss.backward()

        loss_old = loss.item() + loss_nonsmooth_old
        # print(i, criterion(f_a, c_a.unsqueeze(0)).item(), loss_old)
        v_old = v
        grad = v.grad.clone().detach()


        # Forward-Backward step
        with torch.no_grad():
            v = prox_l1(v - stepsize*grad)

            # added distance
            d_v = v - v_old

            # Linesearch
            flag = False
            index_i = 0

            while not flag:

                new_v = v_old + (delta**(index_i))*d_v

                # Compute the loss (we could cut the running by 2 if we stored the computation graph ..)

                Dv= dict_mult(D,new_v)
                a = x + Dv
                loss_smooth = criterion(classifier(a.unsqueeze(0)), c_a.unsqueeze(0)) + .5*l2_fool*torch.sum(Dv**2)
                loss_nonsmooth = l1_sparse*torch.sum(torch.abs(new_v))
                loss_smooth, loss_nonsmooth = loss_smooth.item(), loss_nonsmooth.item()
                loss = loss_smooth + loss_nonsmooth

                # Check the sufficient decrease condition
                h = torch.sum((v-v_old)*grad) + .5*(gamma/stepsize)*(torch.norm(v-v_old,'fro')**2)
                h = h.item() + loss_nonsmooth - loss_nonsmooth_old
                crit = loss_old + beta*(delta**(index_i))*( h )

                if loss <= crit:
                    # Then its fine !
                    v = new_v
                    flag = True
                else:
                    # Then we need to change index_i
                    index_i = index_i + 1
                    if index_i > 10:
                        # We have reached a stationary point
                        # flag_stop = True
                        return x + dict_mult(D, v_old), dict_mult(D, v_old), v_old

    return x + dict_mult(D, v), dict_mult(D, v), v
