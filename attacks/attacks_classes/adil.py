from attacks.utils import *
import torch
import os


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

    def __init__(self, model, eps=None, steps=1e2, norm='L2', targeted=False, n_atoms=10, batch_size=None,
                 data_train=None, step_size=None, trials=10, attack='unsupervised', model_name=None,
                 estimate_step_size=False):
        super().__init__("ADIL", model.eval())
        # Attack parameters
        self.norm = norm.lower()
        self.eps = eps
        self.n_atoms = n_atoms
        self.dictionary = None
        self.targeted = targeted
        self.attack = attack
        self.trials = trials

        # Algorithmic parameters
        self.steps = steps
        self.estimate_step_size = estimate_step_size
        self.step_size = n_atoms if step_size is None else step_size
        self.batch_size = batch_size
        self.loss = None

        path = f"dict_model_ImageNet_version_constrained/"
        model_file = f"ImageNet_{model_name}_num_atom_{self.n_atoms}_nepoch_{self.steps}.bin"
        self.model_file = os.path.join(path, model_file)

        # Learn dictionary
        if data_train is not None:
            self.learn_dictionary(dataset=data_train, model=model)

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
            """ In order to respect linf bound, sample v on a l1-sphere """
            var = torch.randn(n_samples, self.n_atoms)
            v_norm = torch.norm(var, p=1, dim=1, keepdim=True)
            return self.eps * torch.div(var, v_norm)

    def get_target(self, data_loader):
        """ Output the target or label to fool depending on self.targeted """
        target = []
        for index, (x, y) in enumerate(data_loader):
            target.append(get_target(x.to(device=self.device), y.to(device=self.device), self.targeted, self.model))
        return target

    def learn_dictionary(self, dataset, model):
        """ Learn the adversarial dictionary """

        # Shape parameters
        n_img = len(dataset)
        x, _ = next(iter(dataset))
        nc, nx, ny = x.shape

        # Line-search parameters
        delta = .5
        gamma = 1
        beta = .5
        lipschitz = .9 / self.step_size

        # Other parameters
        batch_size = n_img if self.batch_size is None else self.batch_size
        coeff = 1. if self.targeted else -1.
        indices = get_slices(n_img, batch_size)  # Slices of samples according to the batch-size

        # Data loader
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        target = self.get_target(data_loader=data_loader)

        # Function
        criterion = nn.CrossEntropyLoss(reduction='sum')

        # Initialization of the dictionary D and coding vectors v
        if self.norm.lower() == 'l2':
            d = self.projection_d(torch.randn(nc, nx, ny, self.n_atoms, device=self.device))
        else:
            d = (-1 + 2*torch.rand(nc, nx, ny, self.n_atoms, device=self.device))
        v = torch.zeros(n_img, self.n_atoms, device=self.device)

        # Initialization of intermediate variables
        d_old = torch.zeros_like(d)
        v_old = torch.zeros_like(v)
        grad_v_old = torch.zeros_like(v)
        grad_d_old = torch.zeros_like(d)
        loss_all = np.nan * np.ones(int(self.steps))
        index_i = 0

        # Algorithm
        bar = trange(int(self.steps))
        flag_stop = False
        for iteration in bar:

            if not flag_stop:

                # Gradients and loss computations
                grad_v = torch.zeros_like(v)
                grad_d = torch.zeros_like(d)
                loss_full = 0
                for index, (x, _) in enumerate(data_loader):
                    # Prepare computation graph
                    v.detach()
                    v.requires_grad = True
                    d.detach()
                    d.requires_grad = True

                    # Load data
                    x = x.to(device=self.device)  # , y.to(device=self.device)
                    ind = indices[index]
                    dv = torch.tensordot(v[ind], d, dims=([1], [3]))
                    loss = coeff * criterion(model(x + dv), target[index])
                    loss.backward()

                    with torch.no_grad():
                        grad_v += v.grad
                        grad_d += d.grad
                        loss_full += loss.item()

                # Forward-Backward step with line-search
                with torch.no_grad():

                    # # Cheap guess of the Lipschitz constant
                    # if self.estimate_step_size:
                    #     if iteration <= 1:
                    #         lipschitz_old = lipschitz
                    #         lipschitz = torch.sqrt(
                    #             torch.norm(grad_v - grad_v_old, p='fro') ** 2 + torch.norm(grad_d - grad_d_old,
                    #                                                                        p='fro') ** 2)
                    #         lipschitz = lipschitz / torch.sqrt(
                    #             torch.norm(v - v_old, p='fro') ** 2 + torch.norm(d - d_old, p='fro') ** 2)
                    #         lipschitz = lipschitz_old if torch.isinf(lipschitz) else lipschitz
                    #         self.step_size = .9 / lipschitz

                    # Memory
                    d_old.copy_(d)
                    v_old.copy_(v)
                    # grad_v_old.copy_(grad_v)
                    # grad_d_old.copy_(grad_d)
                    loss_old = loss_full

                    # Update
                    v = self.projection_v(v - self.step_size * grad_v)
                    d = self.projection_d(d - self.step_size * grad_d)

                    # print(torch.norm(v, p='fro', dim=1))

                    # added distance
                    d_v = v - v_old
                    d_d = d - d_old

                    # First order approximation of the difference in loss
                    h = torch.sum((d - d_old) * grad_d) + torch.sum((v - v_old) * grad_v) \
                        + .5 * (gamma / self.step_size) * (torch.norm(d - d_old, 'fro') ** 2
                                                           + torch.norm(v - v_old, 'fro') ** 2)

                    # Line-search
                    flag = False
                    index_i = np.maximum(index_i - 1, 0)
                    while not flag:

                        v_new = v_old + (delta ** index_i) * d_v
                        d_new = d_old + (delta ** index_i) * d_d

                        # Compute the loss (we could cut the running by 2 if we stored the computation graph ..)
                        loss_new = 0
                        for index, (x, _) in enumerate(data_loader):
                            x = x.to(device=self.device)
                            ind = indices[index]
                            dv = torch.tensordot(v_new[ind], d_new, dims=([1], [3]))
                            loss_new += (coeff * criterion(model(x + dv), target[index])).item()

                        # Check the sufficient decrease condition
                        if loss_new <= loss_old + beta * (delta ** index_i) * h.item():
                            # Then its fine !
                            v = v_new
                            d = d_new
                            flag = True
                        else:
                            # Then we need to change index_i
                            index_i = index_i + 1
                            if index_i > 50:
                                # We have reached a stationary point
                                flag_stop = True
                                flag = True

                    # Keep track of loss
                    loss_all[iteration] = loss_new
        torch.save([d, loss_all], self.model_file)

    def forward(self, images, labels):

        images = images.to(self.device)
        labels = labels.to(self.device)

        # Check if the dictionary has been learned
        if not os.path.exists(self.model_file):
            print('The adversarial dictionary has not been learned.')
            print('It is now being learned on the given dataset')
            dataset = QuickAttackDataset(images=images, labels=labels)
            self.learn_dictionary(dataset=dataset, model=self.model)

        self.dictionary, _ = torch.load(self.model_file)

        if self.attack == 'supervised':
            ''' Supervised attack where the coding vectors are optimized '''
            adv_img = self.forward_supervised(images, labels)
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

        for _ in range(int(self.trials)):

            # Sample adversarial images
            v = self.sample_sphere(n_samples).to(device=self.device)
            adv_images = torch.zeros_like(images)
            for ind in range(n_samples):
                adv_images[ind, :, :, :] = clamp_image(images[ind, :, :, :] +
                                                       torch.tensordot(v[ind], self.dictionary.to(device=self.device),
                                                                       dims=([0], [3])))
            adv_labels = self.model(adv_images).argmax(dim=1)
            pre_labels = self.model(images).argmax(dim=1)

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
        target = self.get_target(data_loader=data_loader)

        # Initialization of the coding vectors v
        v = torch.zeros(n_img, self.n_atoms, device=self.device)

        # Initialization of intermediate variables
        v_old = torch.zeros_like(v)
        grad_v_old = torch.zeros_like(v)
        loss_all = np.nan * np.ones(int(self.steps))
        index_i = 0

        # Algorithm
        bar = trange(int(self.steps))
        flag_stop = False
        for iteration in bar:

            if not flag_stop:

                # Gradients and loss computations
                grad_v = 0
                loss_full = 0
                for index, (x, _) in enumerate(data_loader):
                    # Prepare computation graph
                    v.detach()
                    v.requires_grad = True

                    # Load data
                    x = x.to(device=self.device)
                    ind = indices[index]
                    dv = torch.tensordot(v[ind], d, dims=([1], [3]))
                    loss = coeff * criterion(self.model(x + dv), target[index])
                    loss.backward()

                    with torch.no_grad():
                        grad_v += v.grad
                        loss_full += loss

                # Forward-Backward step with line-search
                with torch.no_grad():

                    # Guess the Lipschitz constant
                    if self.estimate_step_size:
                        if iteration <= 1:
                            lipschitz_old = lipschitz
                            lipschitz = torch.norm(grad_v - grad_v_old, 'fro') / torch.norm(v - v_old, 'fro') ** 2
                            lipschitz = lipschitz_old if torch.isinf(lipschitz) else lipschitz
                            self.step_size = .9 / lipschitz

                    # Memory
                    v_old.copy_(v)
                    grad_v_old.copy_(grad_v)
                    loss_old = loss_full

                    # Update
                    v = self.projection_v(v - self.step_size * grad_v)

                    # added distance
                    d_v = v - v_old

                    # First order approximation of the difference in loss
                    h = torch.sum((v - v_old) * grad_v) \
                        + .5 * (gamma / self.step_size) * torch.norm(v - v_old, 'fro') ** 2

                    # Line-search
                    flag = False
                    index_i = np.maximum(index_i - 2, 0)
                    while not flag:

                        v_new = v_old + (delta ** index_i) * d_v

                        # Compute the loss (we could cut the running by 2 if we stored the computation graph ..)
                        loss_new = 0
                        for index, (x, _) in enumerate(data_loader):
                            x = x.to(device=self.device)
                            ind = indices[index]
                            dv = torch.tensordot(v_new[ind], d, dims=([1], [3]))
                            loss_new += coeff * criterion(self.model(x + dv), target[index])

                        # Check the sufficient decrease condition
                        if loss_new <= loss_old + beta * (delta ** index_i) * h:
                            # Then its fine !
                            v = v_new
                            flag = True
                        else:
                            # Then we need to change index_i
                            index_i = index_i + 1
                            if index_i > 50:
                                # We have reached a stationary point
                                flag_stop = True
                                flag = True

                    # Keep track of loss
                    loss_all[iteration] = loss_new

        return torch.clamp(images + dv, min=0, max=1)