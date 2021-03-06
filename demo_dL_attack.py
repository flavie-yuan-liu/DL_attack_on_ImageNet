import argparse
import os
import torch
import torchvision.models as models
from torch.utils.data import random_split
from attacks import ADILR, ADIL, UAPPGD, FastUAP
import numpy as np
import performance as perf
from DS_ImageNet import DS_ImageNet
from imagenet_loading import load_ImageNet, dataset_split_by_class
import torchattacks
import random
from model_accuracy import model_accuracy


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

    # ----------------------------------------------------------------------
    # loading imagenet data
    # ----------------------------------------------------------------------
    dataset, classes = load_ImageNet()
    acc = model_accuracy(dataset, model, device=device)
    print("accuracy of the the model {} is {}".format(model_name, acc*100))

    # Set the number of samples for training
    num_train_per_class = args.num_train_per_class  # set the number of samples for training 10 x number of classes
    num_val_per_class = 2
    num_test_per_class = 5

    # prepare the class-balanced dataset
    # default setting: 10 per class for training, 2 per class for validation, 5 per class for testing

    train_dataset, val_dataset, test_dataset \
        = dataset_split_by_class(dataset, [num_train_per_class, num_val_per_class, num_test_per_class],
                                 number_of_classes=args.trained_classes)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=20, shuffle=False) #, pin_memory=True, num_workers=1)

    # ----------------------------------------------------------------------
    # hyper-parameter selecting
    # ----------------------------------------------------------------------

    # lambda_grid_l1 = np.logspace(start=-4, stop=-4, num=1)  # params for regularized adil
    # lambda_grid_l2 = np.logspace(start=-4, stop=-4, num=1)
    n_atoms_grid = np.array([100])
    # log_grid_small = np.logspace(start=-1, stop=4, num=5)
    # log_grid_step_size = np.logspace(start=-3, stop=-1, num=3)
    eps = 8/255
    norm = 'linf'
    alpha_grid = [0/255]
    kappa_grid = [50]

    '''
    BIM(model, eps=8/255, alpha=2/255, steps=100),
    RFGSM(model, eps=8/255, alpha=2/255, steps=100),
    EOTPGD(model, eps=8/255, alpha=2/255, steps=100, eot_iter=2),
    TPGD(model, eps=8/255, alpha=2/255, steps=100),
    VANILA(model),
    GN(model, sigma=0.1),
    FAB(model, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=False),
    FAB(model, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=True),
    Square(model, eps=8/255, n_queries=5000, n_restarts=1, loss='ce'),
    OnePixel(model, pixels=5, inf_batch=50),
    DIFGSM(model, eps=8/255, alpha=2/255, steps=100, diversity_prob=0.5, resize_rate=0.9)
    '''

    attacks_hyper = {
        # 'ADiLR': perf.get_atks(model, ADILR, 'lambda_l1', lambda_grid_l1, 'lambda_l2', lambda_grid_l2,
        #                       'n_atoms', n_atoms_grid, version='stochastic', data_train=train_dataset, device=device,
        #                       batch_size=100, model_name=model_name, steps=150, attack_conditioned='atoms'),
        'adil': perf.get_atks(model.to(device), ADIL, 'n_atoms', n_atoms_grid, 'kappa', kappa_grid, alpha=0/255,
                              data_train=train_dataset, norm=norm, attack='supervised', eps=eps, steps=500,
                              targeted=False, step_size=0.01, batch_size=100,
                              model_name=model_name, is_distributed=args.distributed, steps_in=1, loss='logits',
                              method='gd', data_val=val_dataset, warm_start=False, steps_inference=args.steps_inference),
        # method='gd' or 'alter'; loss='ce' or 'logits
        # 'adil': perf.get_atks(model.to(device), ADIL, 'n_atoms', n_atoms_grid, data_train=train_dataset, norm=norm,
        #                       attack='supervised', eps=eps, steps=150, targeted=False, step_size=0.01, batch_size=128,
        #                       model_name=model_name, is_distributed=args.distributed, steps_in=1, loss='logits',
        #                       method='alter', data_val=val_dataset, warm_start=False),
        # method='gd' or 'alter'; loss='ce' or 'logits'
        #######################################################################################################
        # --------------------------------------- Other attacks --------------------------------------------- #
        #######################################################################################################
        # 'DeepFool': perf.get_atks(model.to(device), DeepFool, steps=100),
        # 'CW': perf.get_atks(model.to(device), torchattacks.CW, 'c', log_grid_small, steps=100, lr=0.001),
        # 'FGSM': perf.get_atks(model.to(device), torchattacks.FGSM, eps=eps),
        # 'FFGSM': perf.get_atks(model.to(device), torchattacks.FFGSM, alpha=10/255, eps=eps),
        # 'MIFGSM': perf.get_atks(model.to(device), torchattacks.MIFGSM, alpha=2/255, eps=eps, steps=100, decay=0.1),
        # 'PGD': perf.get_atks(model.to(device), torchattacks.PGD, eps=eps, alpha=2 / 255, steps=100, random_start=True),
        # --------------------------------- Attacks with l2-ball constraint --------------------------------- #
        # Optimal since eps = radius of ball ** 2
        # 'APGD': perf.get_atks(model.to(device), torchattacks.APGD, loss='ce', norm='Linf', eps=eps, steps=100),
        # 'AutoAttack': perf.get_atks(model.to(device), torchattacks.AutoAttack, norm='Linf', eps=eps, n_classes=1000),
        # ------------------------------------------Universal Attack---------------------------------------------------
        # 'UAP_PGD': perf.get_atks(model.to(device), UAPPGD, eps=eps, data_train=train_dataset, data_val=val_dataset,
        #                          norm=norm, steps=100, model_name=model_name),
        # 'FastUAP': perf.get_atks(model.to(device), FastUAP, eps=eps, steps_deepfool=50, data_train=train_dataset,
        #                         data_val=val_dataset, norm=norm, steps=10, model_name=model_name)

    }

    print('Evaluation process')
    val_perf = perf.get_performance(attacks_hyper, model, val_loader, device=device)
    param_selection_file = f'dict_model_ImageNet_version_constrained/model_sampling_adil_inference_rlts_sampling_' \
                           f'{num_train_per_class*args.trained_classes}_{args.steps_inference}_' \
                           f'{args.seed}_ce.bin'
    torch.save(val_perf, param_selection_file)

    print('Test process')
    test_perf = perf.get_performance(attacks_hyper, model, test_loader, device=device)
    param_selection_file = 'dict_model_ImageNet_version_constrained/model_adil_resultat_test_ce.bin'
    torch.save(test_perf, param_selection_file)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--model', '-m',
        metavar='M',
        default='mobilenet',
    )
    argparser.add_argument(
        '--seed', '-s',
        metavar='S',
        type=int,
        default=3,
        help='change seed to carry out the exp'
    )
    argparser.add_argument(
        '--num-train-per-class',
        type=int,
        default=1,
        help='number per class for training'
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
    argparser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='select number of class for training, default is 0'
    )
    argparser.add_argument(
        '--steps-inference',
        type=int,
        default=100,
        help='select number of steps for inference, default is 100'
    )

    args = argparser.parse_args()
    seed = args.seed  # Do from 1 to 5
    print(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    main(args)



