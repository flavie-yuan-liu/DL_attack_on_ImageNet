import argparse

import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from torch.utils.data import random_split
from imagenet_loading import load_ImageNet
from DS_ImageNet import DS_ImageNet
from attacks import ADIL
from PIL import Image
from torchvision.transforms import transforms
import torchattacks
import random


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
    # check if gpu available
    if not torch.cuda.is_available():
        print('Check cuda setting for model training on ImageNet')
        return

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
    # loading image
    # ----------------------------------------------------------------------
    data, classes = load_ImageNet()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    im_path = 'data/ImageNet/ILSVRC/Data/val/n01484850/ILSVRC2012_val_00002752.JPEG'
    with open(im_path, 'rb') as f:
        im = Image.open(f)
        im = im.convert("RGB")

    im = transform(im)

    # ----------------------------------------------------------------------
    # hyper-parameter selecting
    # ----------------------------------------------------------------------
    eps = 8/255
    attack = ADIL(model.to(device), eps=eps, model_name=model_name)

    im = im.to(device=device)
    y = 2
    label = model(im).argmax(dim=-1)
    adversary = attack(im.unsqueeze(0), label.unsqueeze(0))
    attack_label = model(adversary).argmax(dim=-1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # plt.axis('off')
    axes[0].imshow(im.detach().cpu().numpy().transpose((1, 2, 0)), cmap=plt.jet())
    axes[0].set_title(f'orginal image: {classes[2]}', fontsize=24)
    axes[0].set_axis_off()
    # plt.axis('off')
    scaled_pert = (adversary[0, :, :, :]-im+eps)/torch.max(adversary[0, :, :, :]-im+eps)
    axes[1].imshow(scaled_pert.detach().cpu().numpy().transpose((1, 2, 0)), cmap=plt.jet())
    axes[1].set_title(f'pertubation', fontsize=24)
    axes[1].set_axis_off()
    axes[2].imshow(adversary[0, :, :, :].detach().cpu().numpy().transpose((1, 2, 0)), cmap=plt.jet())
    axes[2].set_title(f'attack image: {classes[109]}', fontsize=24)
    axes[2].set_axis_off()
    fig.tight_layout(pad=0.5)
    plt.savefig('attack_samples.png')
    plt.show()

    return


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--model', '-m',
        metavar='M',
        default='mobilenet',
    )
    args = argparser.parse_args()
    main(args)
