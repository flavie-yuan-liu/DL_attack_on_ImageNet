import argparse
import os
import torch
import torchvision.models as models
from torch.utils.data import random_split
from attacks.dictionary_attack import ADIL
import numpy as np
import simulations.performance as perf
from DS_ImageNet import DS_ImageNet
import random
from torch.utils.data import Subset
import torchmetrics


def dataset_split_by_class(dataset, number_per_class):

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

    indices_train = matrix_sorted_idx[:, 0:number_per_class[0]].flatten()
    indices_val = matrix_sorted_idx[:, number_per_class[0]:split1].flatten()
    indices_test = matrix_sorted_idx[:, split1:split2].flatten()

    return Subset(dataset, indices_train), Subset(dataset, indices_val), Subset(dataset, indices_test)


def load_ImageNet():

    # Set data path for ILSVRC2012 validation set
    imagenet_file = './data/ImageNet/ImageNet1000.bin'

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

    # ----------------------------------------------------------------------
    # loading imagenet data
    # ----------------------------------------------------------------------

    dataset, classes = load_ImageNet()
    acc = model_accuracy(dataset, model, device=device)
    print("accuracy of the the model {} is {}".format(model_name, acc*100))

    # Set the number of samples for training
    num_train_per_class = 10  # set the number of samples for training 10 x number of classes
    num_val_per_class = 2
    num_test_per_class = 5

    # train_dataset, test1_dataset = random_split(dataset, [num_train_set, num_data-num_train_set])
    # val_dataset, test_dataset = random_split(test1_dataset, [num_val_set, num_data-num_train_set-num_val_set])

    train_dataset, val_dataset, test_dataset \
        = dataset_split_by_class(dataset, [num_train_per_class, num_val_per_class, num_test_per_class])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    # ----------------------------------------------------------------------
    # hyper-parameter selecting
    # ----------------------------------------------------------------------
    lambda_grid = np.logspace(start=-5, stop=-1, num=4)
    n_atoms_grid = np.array([20, 50])

    attacks_hyper = {
        'ADiL': perf.get_atks(model.to(device), ADIL, 'lambda_l1', lambda_grid, 'lambda_l2', lambda_grid,
                              'n_atoms', n_atoms_grid, version='stochastic', data_train=train_dataset,
                              device=device, batch_size=128, model_name=model_name, steps=5e2)
    }


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--model', '-m',
        metavar='M',
        default='resnet',
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