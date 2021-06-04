# DL_attack_on_ImageNet

A Pytorch implementation of attack learning based on dictionary leanring, specially for the Imagenet Dataset. The implementation for CIFAR10 can be found in https://github.com/lucasanquetil/Sparse-Dictionary-Attack.git


## Start the training:
To train the dictionary for attack, we assume that you have already a trained deep network. Change the model load path to your own and begin the training.

By defaulte, we attack the resenet18 and trained a dictionary on 128 images, you can test with your own image by using the function attacks.DL_attack.make_adversarial_image


## Reference:
__________________________________
Adversarial Dictionary Learning




