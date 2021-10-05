from attacks.utils import *
import torch
import os

# ------------------------------------------------------------------------- #
# -------------------------- DICTIONARY LEARNING -------------------------- #
# ------------------------------------------------------------------------- #


def adil(dataset, model, targeted=True, niter=1e3, lambdaCoding=1., l2_fool=1., batchsize=None, step_size=.1, n_atom=10,
         dict_set='l2ball', device="cpu", dictionary=None):
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
          n_atom=5, dict_set='l2ball', device=torch.device("cpu"), model_file=None):
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
        if abs(loss[-1]-loss[-2]) < 1e-6:
            break

    torch.save([D, loss], model_file)

    return D, v, _


def sadil_updated(dataset, model, targeted=True, nepochs=1e3, batchsize=1, lambdaCoding=1., l2_fool=1., stepsize=1.,
          n_atom=5, dict_set='l2ball', device="cpu", model_file=None):
    # an updated version of sadil aiming at reducing time and accelerating convergence
    #   - Version for large scale dataset, e.g., ImageNet
    #
    # dataset: dataset to attack
    # classifier: model to fool
    # nepochs: number of epochs
    # batchsize: number of images treated at each iteration

    # Parameters (number of images, dimension of images (3 channels))
    nimg = len(dataset)
    x, _ = next(iter(dataset))
    nc, nx, ny = x.shape

    delta = 0.5
    beta = 0.5

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
    stepsize_D = stepsize
    stepsize_v = stepsize

    # Function
    criterion = nn.CrossEntropyLoss(reduction='sum')
    # prox_l1 = get_prox_l1(param=stepsize * lambdaCoding)

    # Initialization
    D = torch.randn(3, nx, ny, n_atom, device=device)
    D = constraint_dict(D, constr_set=dict_set)
    v = torch.zeros(nimg, n_atom, device=device)

    def loss_all(loader, classifier, vec, Dict):
        loss = 0
        for index, (x, y) in enumerate(loader):
            x, y = x.to(device=device), y.to(device=device)
            vec.detach()
            Dict.detach()
            ind = indices[index]
            Dv = torch.tensordot(vec[ind], Dict, dims=([1], [3]))
            loss += (coeff * criterion(model(x + Dv), get_target(x, y, targeted, classifier)) \
                   + .5 * l2_fool * torch.sum(Dv ** 2)).item()
        loss = loss + (lambdaCoding * torch.sum(torch.abs(vec))).item()
        return loss

    loss = [loss_all(dataLoader, model, v, D)]
    label = []
    pred = []

    # Algorithm (we can improve the visibility by defining an optimizer)
    bar = trange(int(nepochs))
    for i_bar in bar:

        grad_D = torch.zeros_like(D)
        i_max = 0

        for index, (x, y) in enumerate(dataLoader):
            x, y = x.to(device=device), y.to(device=device)
            ind = indices[index]
            if i_bar == 0:
                label = label + y.tolist()
                pred = pred + model(x).sort().indices[:,-1].tolist()

            # Prepare computation graph
            v.detach()
            D.detach()
            v.requires_grad = True

            # Compute the loss
            Dv = torch.tensordot(v[ind], D, dims=([1], [3]))
            loss_smooth = coeff * criterion(model(x + Dv),
                                            get_target(x, y, targeted, model)) + .5 * l2_fool * torch.sum(Dv ** 2)

            # ---------- V-STEP ---------- #
            # Gradient computation & memory
            loss_smooth.backward()
            grad_v = v.grad.data

            # print('grad_v', torch.max(torch.abs(grad_v)).item())

            v_old = v[ind].detach().clone()
            loss_batch_old = (loss_smooth+lambdaCoding * torch.sum(torch.abs(v[ind]))).item()

            # Forward-Backward step
            prox_l1 = get_prox_l1(param=stepsize_v * lambdaCoding)
            with torch.no_grad():
                v[ind] = prox_l1(v[ind] - stepsize_v * grad_v[ind])

            # backtracking
            with torch.no_grad():

                # Compute the loss
                Dv = torch.tensordot(v[ind], D, dims=([1], [3]))
                loss_smooth = coeff * criterion(model(x + Dv),
                                                get_target(x, y, targeted, model)) + .5 * l2_fool * torch.sum(Dv ** 2)
                v_cur = v[ind].detach().clone()
                loss_batch_cur = (loss_smooth+lambdaCoding * torch.sum(torch.abs(v[ind]))).item()
                loss_batch_cur_0 = loss_batch_cur
                delta_h = (torch.sum(torch.mul(grad_v[ind], (v_cur-v_old)))+1/2/stepsize_v * torch.norm(v_cur-v_old)**2\
                          +(torch.sum(torch.abs(v_cur))-torch.sum(torch.abs(v[ind])))).item()

                i = 0
                while loss_batch_cur > loss_batch_old + delta_h*beta and i < 5:
                    i += 1
                    v[ind] = (delta**i)*v_cur + (1-delta**i)*v_old

                    Dv = torch.tensordot(v[ind], D, dims=([1], [3]))
                    loss_smooth = coeff * criterion(model(x + Dv),
                                                    get_target(x, y, targeted, model)) + .5 * l2_fool * torch.sum(Dv ** 2)
                    loss_batch_cur = (loss_smooth+torch.sum(torch.abs(v[ind]))).item()
                    delta_h = delta_h * delta

                if loss_batch_cur_0 <= loss_batch_cur:
                    v[ind] = v_cur
                else:
                    v[ind] = v_cur
                    i_max = max(i, i_max)

            v.detach()
            D.detach()
            D.requires_grad = True

            # Compute the loss
            Dv = torch.tensordot(v[ind], D, dims=([1], [3]))
            loss_smooth = coeff * criterion(model(x + Dv),
                                            get_target(x, y, targeted, model)) + .5 * l2_fool * torch.sum(Dv ** 2)

            # Gradient computation
            loss_smooth.backward()
            grad_D = grad_D + D.grad.data

        stepsize_v = max(stepsize_v*(delta**i_max), 1e-5)

        if torch.max(torch.abs(grad_D)).item() < 1e-4:
            continue

        D_old = D.detach().clone()
        loss_i_old = loss_all(dataLoader, model, v, D_old)

        with torch.no_grad():
            D = D - stepsize_D * grad_D
            D = constraint_dict(D, constr_set=dict_set)

        # determine stepsize around critical point by line search
        with torch.no_grad():
            D_cur = D.detach().clone()
            loss_i_cur = loss_all(dataLoader, model, v, D_cur)
            loss_i_cur_0 = loss_i_cur
            delta_h_D = torch.sum(torch.mul(grad_D, (D_cur-D_old)))+1/2/stepsize_D*torch.norm(D_cur-D_old)**2

            i = 0
            while loss_i_cur > loss_i_old + delta_h_D * beta and i < 5:
                i += 1
                D = (delta ** i) * D_cur + (1 - delta ** i) * D_old
                loss_i_cur = loss_all(dataLoader, model, v, D)
                delta_h_D = delta_h_D * delta

            if loss_i_cur_0 <= loss_i_cur:
                D = D_cur
                loss.append(loss_i_cur_0)
            else:
                D = D_cur
                stepsize_D = max(stepsize_D*delta**i, 1e-6)
                loss.append(loss_i_cur)
            # print(loss[-1])

        if abs(loss[-1] - loss[-2]) < 1e-6:
            break

    torch.save([D, label, pred, v, loss], model_file)

    return D, v

# ------------------------------------------------------------------------- #
# ------------------------- LEARN CODING VECTORS -------------------------- #
# ------------------------------------------------------------------------- #


def learn_coding_vectors(dataset, model, targeted=True, niter=1e2, lambda_l1=1., lambda_l2=1., batch_size=None,
                         step_size=torch.tensor(.1), n_atom=10, dict_set='l2ball', device=torch.device("cpu"),
                         dictionary=None, verbose=False):
    # Shape parameters
    n_img = len(dataset)
    nc, nx, ny, _ = dictionary.shape

    # Line-search parameters
    delta = .9
    gamma = 1
    beta = .5
    lipschitz = .9 / step_size

    # Other parameters
    batch_size = n_img if batch_size is None else batch_size
    coeff = 1. if targeted else -1.  # Targeted vs. Untargeted attacks
    indices = get_slices(n_img, batch_size)  # Slices of samples according to the batch-size

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Function
    criterion = nn.CrossEntropyLoss(reduction='sum')

    # Initialization of the dictionary D and coding vectors v
    d = dictionary
    v = torch.zeros(n_img, n_atom, device=device)

    # Initialization of intermediate variables
    v_old = torch.zeros_like(v)
    loss_all = [np.nan]

    # Algorithm
    bar = trange(int(niter)) if verbose else range(int(niter))
    flag_stop = False
    for iteration in bar:
        # Prepare computation graph
        v.detach()
        v.requires_grad = True

        # Loss computation
        loss_smooth = 0
        for index, (x, y) in enumerate(data_loader):
            x, y = x.to(device=device), y.to(device=device)
            ind = indices[index]
            dv = torch.tensordot(v[ind], d, dims=([1], [3]))
            loss_smooth = loss_smooth + coeff * criterion(model(x + dv), get_target(x, y, targeted, model)) \
                          + .5 * lambda_l2 * torch.sum(dv ** 2)

        loss_old = (loss_smooth + lambda_l1 * torch.sum(torch.abs(v))).item()

        # Gradient computation
        loss_smooth.backward()
        grad_v = v.grad.data

        # Forward-Backward step with line-search
        with torch.no_grad():

            # Memory
            v_old.copy_(v)

            # Step-size
            prox_l1 = get_prox_l1(param=step_size * lambda_l1)

            # Update
            v = prox_l1(v - step_size * grad_v)

            # added distance
            d_v = v - v_old

            # First order approximation of the difference in loss
            h = torch.sum((v - v_old) * grad_v) + .5 * (gamma / step_size) * (torch.norm(v - v_old, 'fro') ** 2) \
                + lambda_l1 * torch.sum(torch.abs(v)) - lambda_l1 * torch.sum(torch.abs(v_old))

            # Line-search
            flag = False
            index_i = 0
            while not flag:
                new_v = v_old + (delta ** index_i) * d_v

                # Compute the loss (we could cut the running by 2 if we stored the computation graph ..)
                loss_smooth = 0
                for index, (x, y) in enumerate(data_loader):
                    x = x.to(device=device)
                    y = y.to(device=device)
                    ind = indices[index]
                    dv = torch.tensordot(new_v[ind], d, dims=([1], [3]))  # replaced by new
                    loss_smooth = loss_smooth + coeff * criterion(model(x + dv), get_target(x, y, targeted, model)) \
                                  + .5 * lambda_l2 * torch.sum(dv ** 2)
                loss_full = (loss_smooth + lambda_l1 * torch.sum(torch.abs(new_v))).item()

                if index_i == 0:
                    loss_cur = loss_full

                # Check the sufficient decrease condition
                crit = loss_old + beta * (delta ** index_i) * h
                if loss_full <= crit:
                    # Then its fine !
                    if loss_cur > loss_full:
                        v = new_v
                        step_size=step_size*delta**index_i
                        loss_all.append(loss_full)
                    else:
                        loss_all.append(loss_cur)
                    flag = True
                else:
                    # Then we need to change index_i
                    index_i = index_i + 1
                    if index_i > 10:
                        # We have reached a stationary point
                        v = new_v
                        loss_all.append(loss_full)
                        flag = True
                dv = torch.tensordot(v, d, dims=([1], [3]))
            # print(iteration, step_size, loss_all[-1], criterion(model(x + dv),
            #                                                    get_target(x, y, targeted, model)).item(),
            #      lambda_l1 * torch.sum(torch.abs(v)).item())
        if loss_all[-2]-loss_all[-1] < 1e-6:
            break

    return v.detach()

# ------------------------------------------------------------------------- #
# ------------------------------ ADIL ATTACK ------------------------------ #
# ------------------------------------------------------------------------- #


class ADILR(Attack):
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
        attack_conditioned: the method for prediction, 'predictions_atoms', 'labels_atoms', 'atoms', 'none'
        trials (int): number of trials to find the best unsupervised attacks to unseen examples. (Default: 100)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    """

    def __init__(self, model, steps=1e2, lambda_l1=1e-1, lambda_l2=1e-1, version='deterministic', targeted=True,
                 attack='supervised', n_atoms=10, batch_size=1, data_train=None, step_size=.01, trials=100, budget=10/255,
                 device='cpu', model_name=None, param_or_train='param_selecting', attack_conditioned='labels_atoms'):
        super().__init__("ADIL", model)
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.n_atoms = n_atoms
        self.steps = steps
        self.batch_size = batch_size
        self.version = version
        self.targeted = targeted
        self.attack = attack
        self.coding_vectors = None
        self.attack_conditioned = attack_conditioned
        self.dictionary = None
        self.scale = None
        self.mean = None
        self.trials = trials
        self.step_size = step_size
        self.device = device
        self.budget = budget

        path = f"dict_model_ImageNet/"
        model_file = f"ImageNet_{model_name}_lamCoding_{self.lambda_l1}_lamFool_{self.lambda_l2}_num_atom_" \
                     f"{self.n_atoms}_nepoch_{self.steps}_{param_or_train}.bin"
        self.model_file = os.path.join(path, model_file)

        if not os.path.exists(self.model_file):
            self.learn_dictionary(dataset=data_train, model=model)
        elif attack == 'unsupervised' and (self.scale is None or self.mean is None):
            _, v, _ = torch.load(self.model_file)
            self.mean, self.scale = fit_laplace(v.detach().cpu())

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

            _, v = sadil_updated(dataset=dataset, model=model.eval(), targeted=self.targeted, nepochs=self.steps,
                                 lambdaCoding=self.lambda_l1, l2_fool=self.lambda_l2, batchsize=self.batch_size,
                                 stepsize=self.step_size, n_atom=self.n_atoms, dict_set='l2ball', device=self.device,
                                 model_file=self.model_file)

        self.mean, self.scale = fit_laplace(v.detach().cpu())

    def forward_unsupervised_conditioned_atoms(self, images):
        """ Unsupervised attack to unseen examples
        The method relies on sampling the coding vectors randomly according to some Laplace distribution
        """

        # Parameters
        n_samples = images.shape[0]
        mean = self.mean['atoms']
        scale = self.scale['atoms']

        # Variables
        fooling_flag = torch.zeros(n_samples, dtype=torch.bool).to(device=self.device)
        mse_best_do_fool = np.inf * torch.ones(n_samples)
        mse_best_no_fool = np.inf * torch.ones(n_samples)
        adv_images_best = torch.zeros_like(images)

        laplace_all = []
        for ind in range(self.n_atoms):
            laplace_all.append(torch.distributions.laplace.Laplace(loc=mean[ind], scale=scale[ind]))
        for _ in range(self.trials):

            # Sample adversarial images
            v = torch.zeros([n_samples, self.n_atoms]).to(device=self.device)
            for ind in range(self.n_atoms):
                v[:, ind] = laplace_all[ind].sample(sample_shape=[n_samples]).to(device=self.device)
            adv_images = torch.zeros_like(images)
            for ind in range(n_samples):
                adv_images[ind, :, :, :] = clamp_image(images[ind, :, :, :] +
                                                       torch.tensordot(v[ind], self.dictionary.to(device=self.device),
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
                    # the model has been fooled with this model and this adv is also fooling the model
                    if mse[ind] < mse_best_do_fool[ind]:
                        mse_best_do_fool[ind] = mse[ind]
                        adv_images_best[ind] = adv_images[ind]
                elif torch.all(~fooling_flag[ind]):
                    # the model has never been fooled with this sample
                    if mse[ind] < mse_best_no_fool[ind]:
                        mse_best_no_fool[ind] = mse[ind]
                        adv_images_best[ind] = adv_images[ind]

        return adv_images_best

    def forward_unsupervised_conditioned_targetAtoms(self, images, labels, version='labels'):
        """ Unsupervised attack to unseen examples
        The method relies on sampling the coding vectors randomly according to some Laplace distribution
        """

        # Parameters
        n_samples = images.shape[0]
        if version.lower() == 'labels':
            mean = self.mean['labels_atoms']
            scale = self.scale['labels_atoms']
            target = labels
        else:
            mean = self.mean['predictions_atoms']
            scale = self.scale['predictions_atoms']
            target = self.model.eval()(images).argmax(dim=1)

        # Variables
        fooling_flag = torch.zeros(n_samples, dtype=torch.bool).to(device=self.device)
        mse_best_do_fool = np.inf * torch.ones(n_samples)
        mse_best_no_fool = np.inf * torch.ones(n_samples)
        adv_images_best = torch.zeros_like(images)

        for _ in range(self.trials):

            # Sample adversarial images
            v = torch.zeros([n_samples, self.n_atoms]).to(device=self.device)
            adv_images = torch.zeros_like(images)
            for ind in range(n_samples):
                mean_sample = mean[target[ind]]
                scale_sample = scale[target[ind]]
                for ind_atom in range(self.n_atoms):
                    laplace = torch.distributions.laplace.Laplace(loc=mean_sample[ind_atom],
                                                                  scale=scale_sample[ind_atom])
                    v[ind, ind_atom] = laplace.sample(sample_shape=[1]).to(device=self.device)

                adv_images[ind, :, :, :] = clamp_image(images[ind, :, :, :] +
                                                       torch.tensordot(v[ind], self.dictionary.to(device=self.device),
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
                    # the model has been fooled with this model and this adv is also fooling the model
                    if mse[ind] < mse_best_do_fool[ind]:
                        mse_best_do_fool[ind] = mse[ind]
                        adv_images_best[ind] = adv_images[ind]
                elif torch.all(~fooling_flag[ind]):
                    # the model has never been fooled with this sample
                    if mse[ind] < mse_best_no_fool[ind]:
                        mse_best_no_fool[ind] = mse[ind]
                        adv_images_best[ind] = adv_images[ind]

        return adv_images_best

    def forward_unsupervised(self, images):
        """ Unsupervised attack to unseen examples
        The method relies on sampling the coding vectors randomly according to some Laplace distribution
        """

        # Parameters
        n_samples = images.shape[0]
        mean = self.mean['none']
        scale = self.scale['none']

        # Variables
        fooling_flag = torch.zeros(n_samples, dtype=torch.bool).to(device=self.device)
        mse_best_do_fool = np.inf * torch.ones(n_samples)
        mse_best_no_fool = np.inf * torch.ones(n_samples)
        adv_images_best = torch.zeros_like(images)

        laplace = torch.distributions.laplace.Laplace(loc=mean, scale=scale)
        for _ in range(self.trials):

            # Sample adversarial images
            v = laplace.sample(sample_shape=[n_samples, self.n_atoms]).to(device=self.device)
            adv_images = torch.zeros_like(images)
            for ind in range(n_samples):
                adv_images[ind, :, :, :] = clamp_image(images[ind, :, :, :] +
                                                       torch.tensordot(v[ind], self.dictionary.to(device=self.device),
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
                    # the model has been fooled with this model and this adv is also fooling the model
                    if mse[ind] < mse_best_do_fool[ind]:
                        mse_best_do_fool[ind] = mse[ind]
                        adv_images_best[ind] = adv_images[ind]
                elif torch.all(~fooling_flag[ind]):
                    # the model has never been fooled with this sample
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
        # time1 = time.time()
        v = learn_coding_vectors(dataset=dataset, model=self.model.eval(), targeted=self.targeted, device=self.device,
                                 lambda_l1=self.lambda_l1, lambda_l2=self.lambda_l2, batch_size=None,
                                 step_size=torch.tensor(100), n_atom=self.n_atoms, dict_set='l2ball',
                                 dictionary=self.dictionary.to(device=self.device))

        # print(time.time()-time1)
        # Build the adversarial images
        adv_images = torch.zeros_like(images)
        for ind in range(n_samples):
            Dv = torch.tensordot(v[ind], self.dictionary.to(device=self.device), dims=([0], [3]))
            Dv = torch.clamp(Dv, min=-self.budget, max=self.budget)
            adv_images[ind, :, :, :] = clamp_image(images[ind, :, :, :] + Dv)
        return adv_images.detach()

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # Check if the dictionary has been learned
        if not os.path.exists(self.model_file):
            print('The adversarial dictionary has not been learned.')
            print('It is now being learned on the given dataset')
            dataset = QuickAttackDataset(images=images, labels=labels)
            self.learn_dictionary(dataset=dataset, model=self.model)

        self.dictionary, _, _ = torch.load(self.model_file)

        if self.attack == 'supervised':
            ''' Supervised attack where the coding vectors are optimized '''
            adv_img = self.forward_supervised(images, labels)
        else:
            ''' Unsupervised attack where the coding vectors are sampled '''
            if self.attack_conditioned == 'labels_atoms':
                adv_img = self.forward_unsupervised_conditioned_targetAtoms(images, labels, version='labels')
            if self.attack_conditioned == 'predictions_atoms':
                adv_img = self.forward_unsupervised_conditioned_targetAtoms(images, labels, version='predictions')
            elif self.attack_conditioned == 'atoms':
                adv_img = self.forward_unsupervised_conditioned_atoms(images)
            else:
                adv_img = self.forward_unsupervised(images)
        self.dictionary = None
        return adv_img

