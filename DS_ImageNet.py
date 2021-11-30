import os
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import argparse
import torch


ROOT_PATH = 'ILSVRC'
TRAIN_PATH = os.path.join(ROOT_PATH, 'Data/train')
VALID_PATH = os.path.join(ROOT_PATH, 'Data/val')
LABLE_PATH = os.path.join(ROOT_PATH, 'LOC_synset_mapping.txt')


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


def read_label(path):

    classes_to_wnids = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            t = line.rstrip().split(' ', 1)
            classes_to_wnids[t[0]] = t[1]

    return classes_to_wnids


class DS_ImageNet(ImageFolder):

    def __init__(self, filepath, split='train', transform=None, target_transform=None):
        self.split = split
        datapath = TRAIN_PATH if self.split == 'train' else VALID_PATH
        fullpath = os.path.join(filepath, datapath)
        super(DS_ImageNet, self).__init__(fullpath, transform=transform, target_transform=target_transform)
        self.idx_to_class = self.dict_item_rev_order()
        self.classes_to_wnids = read_label(os.path.join(filepath, LABLE_PATH))
        self.classes = [self.classes_to_wnids[self.idx_to_class[i]].split(',', 1)[0] for i in range(len(self.classes))]

    def dict_item_rev_order(self):
        idx_to_class = {}
        for key, value in self.class_to_idx.items():
            idx_to_class[value] = idx_to_class.get(value, key)
        return idx_to_class


def main(args):
    root = args.root
    valid_data = DS_ImageNet(root, split=args.split, transform=transform)
    # torch.save(valid_data, os.path.join(root, args.file_samples_dataset))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('ImageNet management')
    argparser.add_argument(
        '--root', '-r',
        metavar='R',
        default='./data/ImageNet',
        help='ImageNet root file path from current project (default "./")'
    )
    argparser.add_argument(
        '--split',
        metavar='S',
        default='val',
        help='Train data or validation data (default train)'
    )
    argparser.add_argument(
        '--file-samples-dataset',
        metavar='P',
        default='ImageNet1000_unnormalized.bin'
    )
    args = argparser.parse_args()

    main(args)




