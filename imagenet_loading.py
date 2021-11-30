from DS_ImageNet import DS_ImageNet
import torch
import random
from torch.utils.data import Subset
import numpy as np


class Subset_I(Subset):
    def __init__(self, dataset, indices, indexed=False):
        super(Subset_I, self).__init__(dataset=dataset, indices=indices)
        self.indexed = indexed

    def __getitem__(self, item):
        x, y = super().__getitem__(item)
        if self.indexed:
            return item, x, y
        else:
            return x, y


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
    indices_test = matrix_sorted_idx[:number_of_classes, split1:split2].flatten()

    return Subset_I(dataset, indices_train), Subset_I(dataset, indices_val), Subset_I(dataset, indices_test)


def load_ImageNet():

    # Set data path for ILSVRC2012 validation set
    imagenet_file = './data/ImageNet/ImageNet1000_unnormalized.bin'

    # load ImageNet dataset, the validation set is used here for dictionary training
    dataset = torch.load(imagenet_file)
    num_data = len(dataset)
    clsses = dataset.classes

    return dataset, clsses