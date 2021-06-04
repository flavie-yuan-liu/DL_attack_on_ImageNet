import os
import torch
from torchvision.transforms import transforms
from MyImageNet import MyImageNet
from torch.utils.data import random_split
from attacks.DL_attack import dict_training
from evaluation import evl


def load_ImageNet():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Attention: this transformation should be consistent with that used for model training
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    # Set root file path for ILSVRC2012 validation set
    imagenet_root = './data/ImageNet/ILSVRC/Data'

    # load ImageNet dataset, the validation set is used here for dictionary training
    dataset = MyImageNet.MyImageNet(root=imagenet_root, split='val', transform=trans)
    num_data = len(dataset)

    # Set the number of samples for training
    train_size = 128# int(0.1 * num_data)
    train_dataset, test_dataset = random_split(dataset,
                                             [train_size, num_data - train_size])

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, shuffle=True, batch_size=128  # , num_workers=0, pin_memory=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=128, shuffle=False  # , num_workers=0, pin_memory=True
    )

    return train_loader, test_loader


if __name__ == '__main__':

    if torch.cuda.is_available():

        dataset_name = 'ImageNet'
        model_name = 'ResNet'

        # load data
        train_loader, test_loader = load_ImageNet()

        # load model (By default, we load the trained ResNet18 on ImageNet)
        model = torch.load('./models/ImageNet_ResNet.model')

        l2_fool, l1_sparse = 1e-4, 1e-4
        folder_name = f"./models"
        filename_D = f"Dict_{dataset_name}_{model_name}_LamReg_{l2_fool}_LamCoding_{l1_sparse}.bin"
        fullpath = os.path.join(folder_name, filename_D)
        # train dictionary for attack if DL attack model not exist
        if not os.path.exists(fullpath):
            dict_training(train_loader, model, fullpath, lambdaCoding=l1_sparse, lambdaReg=l2_fool, stepsize=.1, niter=20)

        # load D
        D, _ = torch.load(fullpath)

        # test DL attack model's fooling rate
        test_images, label = next(iter(test_loader))
        point, param = evl(test_images, label, D, model, l2_fool, l1_sparse)


    else:
        print("Error: this project need to be run on GPU")




