import argparse
import os
import torch
import torchvision.models as models
from torch.utils.data import random_split
from attacks import ADILR, ADIL
import numpy as np
import performance as perf
from DS_ImageNet import DS_ImageNet
from imagenet_loading import load_ImageNet, dataset_split_by_class
import torchattacks
import random
import model_accuracy



class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input-mean)/std


def main(args):

    if not torch.cuda.is_available():
        print('Check cuda setting for model training on ImageNet')
        return

    if not args.distributed:
        torch.cuda.set_device(0)
        device = torch.device(0)
    torch.backends.cudnn.benchmark = True

    # ------------------------------------------------------------------------
    # loading model (densenet, googlenet, inception, mobilenetv2, resnet, vgg)
    # ------------------------------------------------------------------------

    model_name = args.model.lower()
    if model_name == 'resnet':
        model = models.resnet18(pretrained=True, progress=False)
    elif model_name == 'densenet':
        model = models.densenet121(pretrained=True, progress=False)
    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=True, progress=False)
    elif model_name == 'inception':
        model = models.inception_v3(pretrained=True, progress=False)
    elif model_name == 'mobilenet':
        model = models.mobilenet_v2(pretrained=True, progress=False)
    elif model_name == 'vgg':
        model = models.vgg11(pretrained=True, progress=False)

    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = torch.nn.Sequential(
        norm_layer,
        model
    )
    model.eval()

    # ----------------------------------------------------------------------
    # loading imagenet data
    # ----------------------------------------------------------------------

    dataset, classes = load_ImageNet()
    # if args.distributed:
    #     acc = model_accuracy.run_accuracy_computing(args, dataset, model)
    # else:
    #     acc = model_accuracy(args, dataset, model, device=device)
    # print("accuracy of the the model {} is {}".format(model_name, acc*100))

    # Set the number of samples for training
    num_train_per_class = 2  # set the number of samples for training 10 x number of classes
    num_val_per_class = 2
    num_test_per_class = 5

    # prepare the class-balanced dataset
    # default setting: 10 per class for training, 2 per class for validation, 5 per class for testing

    train_dataset, val_dataset, test_dataset \
        = dataset_split_by_class(dataset, [num_train_per_class, num_val_per_class, num_test_per_class],
                                 number_of_classes=args.trained_classes)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=20, shuffle=False) #, pin_memory=True, num_workers=1)

    # ----------------------------------------------------------------------
    # hyper-parameter selecting
    # ----------------------------------------------------------------------

    # lambda_grid_l1 = np.logspace(start=-4, stop=-4, num=1)
    # lambda_grid_l2 = np.logspace(start=-4, stop=-4, num=1)
    n_atoms_grid = np.array([1, 10, 50, 100])
    log_grid_small = np.logspace(start=-1, stop=4, num=5)
    # log_grid_step_size = np.logspace(start=-3, stop=-1, num=3)
    eps = 8/255
    norm = 'linf'
    num_trials_grid = [10]
    steps_in_grid = 100

    # eps = [0.5]
    # norm = 'l2'

    '''
    FGSM(model, eps=8/255),
    FFGSM(model, eps=8/255, alpha=10/255),
    CW(model, c=1, lr=0.01, steps=100, kappa=0),
    DeepFool(model, steps=100),
    PGD(model, eps=8/255, alpha=2/225, steps=100, random_start=True),
    PGDL2(model, eps=8/255, alpha=0.2, steps=100),
    
    BIM(model, eps=8/255, alpha=2/255, steps=100),
    RFGSM(model, eps=8/255, alpha=2/255, steps=100),
    EOTPGD(model, eps=8/255, alpha=2/255, steps=100, eot_iter=2),
    TPGD(model, eps=8/255, alpha=2/255, steps=100),
    MIFGSM(model, eps=8/255, alpha=2/255, steps=100, decay=0.1),
    VANILA(model),
    GN(model, sigma=0.1),
    APGD(model, eps=8/255, steps=100, eot_iter=1, n_restarts=1, loss='ce'),
    APGD(model, eps=8/255, steps=100, eot_iter=1, n_restarts=1, loss='dlr'),
    APGDT(model, eps=8/255, steps=100, eot_iter=1, n_restarts=1),
    FAB(model, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=False),
    FAB(model, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=True),
    Square(model, eps=8/255, n_queries=5000, n_restarts=1, loss='ce'),
    AutoAttack(model, eps=8/255, n_classes=10, version='standard'),
    OnePixel(model, pixels=5, inf_batch=50),
    DIFGSM(model, eps=8/255, alpha=2/255, steps=100, diversity_prob=0.5, resize_rate=0.9)
    '''

    attacks_hyper = {
        # 'ADiLR': perf.get_atks(model.to(device), ADILR, 'lambda_l1', lambda_grid_l1, 'lambda_l2', lambda_grid_l2,
        #                       'n_atoms', n_atoms_grid, version='stochastic', data_train=train_dataset, device=device,
        #                       batch_size=100, model_name=model_name, steps=150, attack_conditioned='atoms'),Text(0.5, 124.8322222222222, 'number of trials with computing time 30-38s, 122-132s, 524-552s, 1023-1072s, 5014-5222s, 10033-10420s')

        # 'adil': perf.get_atks(model.to(device), ADIL, 'n_atoms', n_atoms_grid, 'trials', num_trials_grid, steps_in=100,
        #                       data_train=train_dataset, data_val=val_dataset, norm=norm, attack='unsupervised',
        #                       eps=eps, steps=1000, targeted=False, step_size=1, batch_size=50, model_name=model_name),
        # --------------------------------------- Other attacks --------------------------------------------- #
        # 'DeepFool': perf.get_atks(model.to(device), torchattacks.DeepFool, steps=100),
        # 'CW': perf.get_atks(model.to(device), torchattacks.CW, 'c', log_grid_small, steps=100, lr=0.001),
        # 'FGSM': perf.get_atks(model.to(device), torchattacks.FGSM, eps=eps),
        # 'FFGSM': perf.get_atks(model.to(device), torchattacks.FFGSM, alpha=10/255, eps=eps),
        # 'MIFGSM': perf.get_atks(model.to(device), torchattacks.MIFGSM, alpha=2/255, eps=eps, steps=100, decay=0.1),
        # 'PGD': perf.get_atks(model.to(device), torchattacks.PGD, eps=eps, alpha=2 / 255, steps=100, random_start=True),
        # --------------------------------- Attacks with l2-ball constraint --------------------------------- #
        # Optimal since eps = radius of ball ** 2
        # 'PGDL2': perf.get_atks(model.to(device), torchattacks.PGDL2, alpha=0.2, eps=eps, steps=100),
        'APGD': perf.get_atks(model.to(device), torchattacks.APGD, loss='ce', norm='Linf', eps=eps, steps=100),
        'AutoAttack': perf.get_atks(model.to(device), torchattacks.AutoAttack, norm='Linf', eps=eps, n_classes=1000),

    }

    # print('Evaluation process')
    # val_perf = perf.get_performance(attacks_hyper, model, val_loader, device=device)
    # param_selection_file = 'dict_model_ImageNet_version_constrained/model_adil_resultat_for_param_selecting.bin'
    # torch.save(val_perf, param_selection_file)

    print('Test process')
    test_perf = perf.get_performance(attacks_hyper, model, test_loader, device=device)
    param_selection_file = 'dict_model_ImageNet_version_constrained/model_baseline_resultat_test.bin'
    torch.save(test_perf, param_selection_file)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--model', '-m',
        metavar='M',
        default='vgg',
    )
    argparser.add_argument(
        '--seed', '-s',
        metavar='S',
        type=int,
        default=3,
        help='change seed to carry out the exp'
    )
    argparser.add_argument(
        '--trained-classes',
        metavar='TC',
        type=int,
        default=1000,
        help='number of class for training'
    )
    argparser.add_argument(
        '--distributed',
        metavar='D',
        type=bool,
        default=False,
        help='If distributed data parallel used, default value is False'
    )

    args = argparser.parse_args()

    seed = args.seed  # Do from 1 to 5
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    main(args)



