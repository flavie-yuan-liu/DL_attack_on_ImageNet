import numpy as np
import simulations.performance as perf
import torch
import torchvision
import torchvision.transforms as transforms
import torchattacks
from models.PyTorch_CIFAR10.cifar10_models.resnet import resnet50
from models.PyTorch_CIFAR10.cifar10_models.densenet import densenet121
from models.PyTorch_CIFAR10.cifar10_models.vgg import vgg11_bn
from models.PyTorch_CIFAR10.cifar10_models.inception import inception_v3
from tabulate import tabulate
from attacks.dictionary_attack import ADIL
import random
import pickle
import os

seed = 1    # Do from 1 to 5
torch.random.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# ------ INSTRUCTIONS ------- #
# To make it work, you first need to download the weights of the neural networks
# Go to models/Pytorch_Cifar10 and run 'python train.py --download_weights 1'
#
# ------ INFORMATION -------- #
# This file is intended as a template for Cifar10 experiments
# Here we show how to select the hyper-parameters according to some criterion and budget
# Then, we evaluate the learned attacks on various architectures
#
# ------ DISCLAIMER -------- #
# This code is not optimal for large scale experiments since everything is kept in memory.
# Instead, one could perform sequential experiments with savings/loadings
# All attacks are objects so they can easily be saved/loaded with pickle


def split_dataset(raw_dataset, n_class, n_val, n_test):
    # Split every example according to its label
    example_per_class = {}
    for (x, y) in raw_dataset:
        if y not in example_per_class.keys():
            example_per_class[y] = [[x, torch.tensor(y)]]
        else:
            y_list = example_per_class[y]
            y_list.append([x, torch.tensor(y)])

    # Randomize the examples
    for y in example_per_class.keys():
        random.shuffle(example_per_class[y])

    # Select the right number of examples for Validation and Test
    validation_set_list = []
    test_set_list = []
    for y in example_per_class.keys():
        n_ex_val = n_val // n_class
        n_ex_test = n_test // n_class
        all_y_example = example_per_class[y]
        n_example = len(all_y_example)

        val_y_example = all_y_example[:n_ex_val]
        test_y_example = all_y_example[n_ex_val:n_ex_test]

        validation_set_list.extend(val_y_example)
        test_set_list.extend(test_y_example)

    return validation_set_list, test_set_list


if __name__ == '__main__':

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    pin_memory = True if use_cuda else False
    torch.backends.cudnn.benchmark = False if use_cuda else False

    ###########################################
    #           Dataset: CIFAR10              #
    ###########################################
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10
    num_validation = 120     # Make sure to have enough samples (> 10*num_labels) and to equally represent all labels
    num_test = 1200          # We should use all the remaining samples
    batch_size = 12

    # Split into validation and test set
    raw_test_data = dataset(root='../data', train=False, download=True, transform=transform)
    validation_set, test_set = split_dataset(raw_dataset=raw_test_data, n_class=10, n_val=num_validation,
                                             n_test=num_test)

    validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, pin_memory=True)

    ###########################################
    #                 Models                  #
    ###########################################

    # Model used to learn the attacks
    model = densenet121(pretrained=True).to(device=device)

    ###########################################
    #                Attacks                  #
    ###########################################

    # Hyper-parameter ranges (examples)
    lambda_grid_1 = np.logspace(start=-3, stop=0, num=10)
    lambda_grid_2 = np.logspace(start=-1, stop=1.5, num=10)
    log_grid_small = np.logspace(start=-4, stop=-1, num=10)                     # Used for very small values
    log_grid_step_size = np.logspace(start=-3, stop=-1, num=5)                  # Used for learning rates
    mse_grid_fine = (np.sqrt(8 / 255)) * np.logspace(start=-2, stop=1, num=10)  # Used for mse-like param
    mse_grid_coarse = [np.sqrt(4 / 255), np.sqrt(8 / 255), np.sqrt(16/255)]     # Used for mse param

    # All attacks with all possible hyper-parameters to check from
    # Torchattacks can be found at: https://github.com/Harry24k/adversarial-attacks-pytorch
    attacks_hyper = {
        'ADiL': perf.get_atks(model, ADIL, 'lambda_l1', lambda_grid_1, 'lambda_l2', lambda_grid_2,
                              data_train=validation_set, steps=1e3),
        # --------------------------------------- Other attacks --------------------------------------------- #
        'DeepFool': perf.get_atks(model, torchattacks.DeepFool, 'overshoot', log_grid_small),
        'CW': perf.get_atks(model, torchattacks.CW, 'lr', log_grid_step_size, 'c', log_grid_small),
        'FGSM': perf.get_atks(model, torchattacks.FGSM, 'eps', log_grid_step_size),
        'FFGSM': perf.get_atks(model, torchattacks.FFGSM, 'eps', mse_grid_fine, 'alpha', log_grid_step_size),
        # --------------------------------- Attacks with l2-ball constraint --------------------------------- #
        # Optimal since eps = radius of ball ** 2
        'PGDL2': perf.get_atks(model, torchattacks.PGDL2, 'eps', mse_grid_coarse, 'alpha', log_grid_step_size),
        'APGD-L2-ce': perf.get_atks(model, torchattacks.APGD, 'eps', mse_grid_coarse, loss='ce', norm='L2'),
        'AutoAttack-L2': perf.get_atks(model, torchattacks.AutoAttack, 'eps', mse_grid_coarse, norm='L2'),
    }

    # Select the hyper-parameters on the validation set according to the given budgets and criterion. For instance
    # criterion='mse_limit' selects the hyper-parameters for which the attacks have the highest fooling rate while not
    # exceeding the mse budget
    print('-' * 7 + ' VALIDATION ' + '-' * 7)
    budget = [4/255, 8/255, 16/255]
    attacks, validation_perf, validation_perf_tmp = perf.select_hyperparameter(attacks_hyper, model=model,
                                                                               data=validation_loader,
                                                                               budget=budget,
                                                                               criterion='mse_limit',
                                                                               device=device)

    ###########################################
    #       Evaluate Attacks Transfer         #
    ###########################################

    # Model to transfer to
    # (In order to save memory we could call them only when needed)
    model_transfer = {
        'DenseNet121': densenet121(pretrained=True),
        'ResNet50': resnet50(pretrained=True),
        'VGG11': vgg11_bn(pretrained=True),
        'INCEPTIONV3': inception_v3(pretrained=True)
    }

    # Evaluate transfer performance on multiple architectures
    print('-' * 7 + ' TEST ' + '-' * 7)
    perf_transfer = []
    for ind, _ in enumerate(budget):
        perf_transfer.append(perf.get_transfer_performance(attacks[ind], model_transfer, data=test_loader,
                                                           device=device))
        print(str(ind+1) + '/' + str(len(budget)))

    ###########################################
    #             Print and Save              #
    ###########################################

    # Save results for all budgets separately
    for ind, budget_val in enumerate(budget):

        # Compute performance table
        result_fooling_rate = []
        for attack_name in perf_transfer[ind].keys():
            result_fooling_rate_tmp = [attack_name]
            header = ['Attack']
            for model_name in perf_transfer[ind][attack_name].keys():
                header.append(model_name)
                result_fooling_rate_tmp.append(perf_transfer[ind][attack_name][model_name]['fooling_rate'])
            result_fooling_rate.append(result_fooling_rate_tmp)
        table = tabulate(result_fooling_rate, headers=header)

        # Print performance
        print(table)
        print('\n')

        # Save performance
        folder_path = os.path.join(os.getcwd(), 'results')
        filename = 'result_' + model._get_name() + '_seed' + str(seed) + '_budget' + "{:.3f}".format(
            budget_val)
        filename = os.path.join(folder_path, filename)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        with open(filename + '.txt', 'w') as f:
            print(table, file=f)
        with open(filename + '.pickle', 'wb') as handle:
            object_to_save = [attacks[ind], validation_perf[ind], validation_perf_tmp]
            pickle.dump(object_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)