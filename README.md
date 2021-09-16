# DL_attack_on_ImageNet

A Pytorch implementation of attack learning based on dictionary leanring, specially for the Imagenet Dataset. The implementation for CIFAR10 can be found in https://github.com/lucasanquetil/Sparse-Dictionary-Attack.git

## Training

### data loading
You can create training data of imagenet by running DS_ImageNet.py. Then, a file named imagenet1000.bin will be created. Or you can special your data name by setting --file-samples-dataset and data path --root. Here we use only 10000 samples of the data in validation set for attack dictionary training). You can also create data with train data by setting --split. In the script, we assume the data is save in ILSVC/Data/train and ILSVC/Data/val, you should change it if your path is different.

### Start the training:
To train the dictionary for attack, we assume that you have already a trained deep network. Change the model load path to your own and begin the training.

By default, we attacked the resenet18 and trained a dictionary on 128 images, you can test with your own image by using the function attacks.DL_attack.make_adversarial_image

## Evaluation


## Reference:
__________________________________
Adversarial Dictionary Learning




