import numpy as np
import simulations.performance as perf
import torch
import torchvision
import torchvision.transforms as transforms
from models.PyTorch_CIFAR10.cifar10_models.densenet import densenet121
from attacks.dictionary_attack import ADIL
import random

seed = 123
torch.random.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# ------ INSTRUCTIONS ------- #
# To make it work, you first need to download the weights of the neural networks
# Go to models/Pytorch_Cifar10 and run 'python train.py --download_weights 1'

# ------ INFORMATION -------- #
# This file is intended to show how the newest version of ADiL works


if __name__ == '__main__':

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    pin_memory = True if use_cuda else False
    torch.backends.cudnn.benchmark = True if use_cuda else False

    ###########################################
    #           Dataset: CIFAR10              #
    ###########################################
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10
    num_validation = 120     # Make sure to have enough samples (> 10*num_labels) and to equally represent all labels
    num_test = 1200          # We should use all the remaining samples
    batch_size = 12
    n_class = 10

    # Split into validation and test set
    raw_test_data = dataset(root='../data', train=False, download=True, transform=transform)


    # Split every example according to its label
    example_per_class = {}
    for (x,y) in raw_test_data:
        if y not in example_per_class.keys():
            example_per_class[y] = [[x,torch.tensor(y)]]
        else:
            y_list = example_per_class[y]
            y_list.append([x,torch.tensor(y)])

    # Randomize the examples
    for y in example_per_class.keys():
        random.shuffle(example_per_class[y])


    # Select the right number of examples for Validation and Test
    validation_set_list = []
    test_set_list = []
    for y in example_per_class.keys():
        n_ex_val = num_validation // n_class
        n_ex_test = num_test // n_class
        all_y_example = example_per_class[y]
        n_example = len(all_y_example)

        val_y_example = all_y_example[:n_ex_val]
        test_y_example = all_y_example[n_ex_val:n_ex_test]

        validation_set_list.extend(val_y_example)
        test_set_list.extend(test_y_example)

    validation_loader = torch.utils.data.DataLoader(dataset=validation_set_list, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set_list, batch_size=batch_size, shuffle=True, pin_memory=True)

    # validation_set, test_set = torch.utils.data.random_split(raw_test_data, [num_validation, len(raw_test_data)-num_validation],
    #                                                          generator=torch.Generator().manual_seed(seed))
    # validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=batch_size,
    #                                                 pin_memory=pin_memory)
    # # Take a smaller subset of the test set
    # test_subset = torch.utils.data.Subset(test_set, list(range(num_test)))
    # test_loader = torch.utils.data.DataLoader(dataset=test_subset, batch_size=batch_size, pin_memory=pin_memory)
    # # Further split the validation into two parts (for ADiL and UAP)
    # # Here I split in half but whatever
    # validation_p1, validation_p2 = torch.utils.data.random_split(validation_set, [num_validation//2, num_validation//2],
    #                                                              generator=torch.Generator().manual_seed(seed))
    validation_p1 = validation_set_list[:num_validation//2]
    validation_p2 = validation_set_list[num_validation//2:]

    ###########################################
    #                 Models                  #
    ###########################################

    # Model used to learn the attacks
    model = densenet121(pretrained=True).to(device=device)

    # Define ADiL
    adil = ADIL(model=model, data_train=validation_p1, lambda_l1=.1, lambda_l2=10, n_atoms=10, version='deterministic')

    # Evaluate perf
    adil.attack = 'unsupervised' #Sample v
    perf_unsupervised = perf.performance(adil, model=model.eval(), data=test_loader)
    print(perf_unsupervised)

    adil.attack = 'supervised'  #Optimize v
    perf_supervised = perf.performance(adil, model=model.eval(), data=test_loader)
    print(perf_supervised)
