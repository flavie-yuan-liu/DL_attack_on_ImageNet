import argparse
import os
import torch
import torchvision.models as models
from torch.utils.data import random_split
from attacks.dictionary_attack import ADIL
import numpy as np
import simulations.performance as perf
from DS_ImageNet import DS_ImageNet
import torchattacks
import random
from torch.utils.data import Subset
import torchmetrics


class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input-mean)/std


def dataset_split_by_class(dataset, number_per_class, number_of_classes=1000):

    # number_of_class is list containing two elements,
    # i.e., number_of_class = [number_per_class_train, number_per_class_val, number_per_class_test]

    samples = dataset.samples
    labels = [l for (_, l) in samples]
    sorted_idx = np.argsort(labels)
    num_classes = len(dataset.classes)

    # this only works for the imagenet validation set with 50 samples per class
    matrix_sorted_idx = sorted_idx.reshape([num_classes, 50])

    split1 = number_per_class[0]+number_per_class[1]
    split2 = sum(number_per_class)

    for i in range(matrix_sorted_idx.shape[0]):
        random.shuffle(matrix_sorted_idx[i, :])

    indices_train = matrix_sorted_idx[:number_of_classes, 0:number_per_class[0]].flatten()
    indices_val = matrix_sorted_idx[:number_of_classes, number_per_class[0]:split1].flatten()
    indices_test = matrix_sorted_idx[number_of_classes:, split1:split2].flatten()

    return Subset(dataset, indices_train), Subset(dataset, indices_val), Subset(dataset, indices_test)


def load_ImageNet():

    # Set data path for ILSVRC2012 validation set
    imagenet_file = './data/ImageNet/ImageNet1000_unnormalized.bin'

    # load ImageNet dataset, the validation set is used here for dictionary training
    dataset = torch.load(imagenet_file)
    num_data = len(dataset)
    clsses = dataset.classes

    return dataset, clsses


def model_accuracy(dataset, model, device='cpu'):
    metric = torchmetrics.Accuracy()
    metric.to(device)
    model.eval()
    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=256)
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            model = model.to(device)
            pred = model(x)
            acc = metric(pred, y)
        acc = metric.compute()
    metric.reset()
    return acc


def main(args):

    if not torch.cuda.is_available():
        print('Check cuda setting for model training on ImageNet')
        return
    device = 'cuda'

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
    # acc = model_accuracy(dataset, model, device=device)
    # print("accuracy of the the model {} is {}".format(model_name, acc*100))

    # Set the number of samples for training
    num_train_per_class = 1  # set the number of samples for training 10 x number of classes
    num_val_per_class = 2
    num_test_per_class = 5

    # prepare the class-balanced dataset
    # default setting: 10 per class for training, 2 per class for validation, 5 per class for testing

    train_dataset, val_dataset, test_dataset \
        = dataset_split_by_class(dataset, [num_train_per_class, num_val_per_class, num_test_per_class],
                                 number_of_classes=1000)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False)

    # ----------------------------------------------------------------------
    # hyper-parameter selecting
    # ----------------------------------------------------------------------

    lambda_grid_l1 = np.logspace(start=-4, stop=-4, num=1)
    lambda_grid_l2 = np.logspace(start=-4, stop=-4, num=1)
    n_atoms_grid = np.array([100])
    log_grid_small = np.logspace(start=-3, stop=-1, num=4)
    log_grid_step_size = np.logspace(start=-3, stop=-1, num=3)
    eps = 10/255

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
        # 'ADiL': perf.get_atks(model.to(device), ADIL, 'lambda_l1', lambda_grid_l1, 'lambda_l2', lambda_grid_l2,
        #                       'n_atoms', n_atoms_grid, version='stochastic', data_train=train_dataset, device=device,
        #                       batch_size=100, model_name=model_name, steps=150, attack_conditioned='atoms'),
        # --------------------------------------- Other attacks --------------------------------------------- #
        # 'DeepFool': perf.get_atks(model.to(device), torchattacks.DeepFool, steps=100),
        'CW': perf.get_atks(model.to(device), torchattacks.CW, 'c', log_grid_small, steps=100),
        'FGSM': perf.get_atks(model.to(device), torchattacks.FGSM, eps=eps),
        'FFGSM': perf.get_atks(model.to(device), torchattacks.FFGSM, alpha=12/255, eps=eps),
        # --------------------------------- Attacks with l2-ball constraint --------------------------------- #
        # Optimal since eps = radius of ball ** 2
        'PGDL2': perf.get_atks(model.to(device), torchattacks.PGDL2, alpha=0.2, eps=eps, steps=100),
        'APGD-L2-ce': perf.get_atks(model.to(device), torchattacks.APGD, loss='ce', norm='L2', eps=eps),
        'AutoAttack-L2': perf.get_atks(model.to(device), torchattacks.AutoAttack, norm='L2', eps=eps),
    }

    print('Evaluation process')
    budget = [4 / 255, 8 / 255, 16 / 255]
    attacks, validation_perf, validation_perf_tmp = perf.select_hyperparameter(attacks_hyper, model=model,
                                                                               data=val_loader,
                                                                               budget=budget,
                                                                               criterion='mse_limit',
                                                                               device=device)

 #   print(validation_perf_tmp['fooling_rate'], validation_perf_tmp['mse'], validation_perf_tmp['rmse'])

    param_selection_file = 'dict_model_ImageNet/model_comparaison.bin'
    torch.save([attacks, validation_perf, validation_perf_tmp], param_selection_file)


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
        default=1,
        help='change seed to carry out the exp'
    )
    args = argparser.parse_args()

    seed = args.seed  # Do from 1 to 5
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    main(args)