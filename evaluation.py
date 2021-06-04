import torch
from attacks.DL_attack import make_adversarial_image
from tqdm import tqdm
from attacks.DL_attack import transform_Img, transforms_inverse, clamp_image


def evl(minibatch, label, D, classifier, lambdaReg, lambdaCoding):
    '''

    Parameters
    ----------
    Problem for solving: argmin Loss(classifier(x), classifier(x+Dv))+.5*lambdaReg|Dv|_F^2 + lambdaCoding|v|_1, (1)
    minibatch: samples for attack creating
    label: the real label of the input samples
    D: learned dictionary for adversarial attack
    classifier: model for data classification, i.e. ResNet
    lambdaReg, lambdaCoding: params in problem (1)

    Returns
    -------
    return the fooling rate and rmse of the model
    '''
    classifier.eval()
    r_diff = 0
    n = 0
    n_img = minibatch.shape[0]
    for (x,y) in zip(tqdm(minibatch), label):
        # Sparse Dico Attack
        x_attack_dico, noise_dico, _ = make_adversarial_image(x, D, classifier, v=None, l2_fool=lambdaReg,
                                                              l1_sparse=lambdaCoding, stepsize=5e-1, niter=5e3)
        # clamp image to have pixel valued well defined
        x_attack_dico = transform_Img(clamp_image(transforms_inverse(x_attack_dico)))
        r_diff = r_diff + torch.norm(x-x_attack_dico)/torch.norm(x)
        if classifier(x.unsqueeze(0)).argmax().item() != classifier(x_attack_dico.unsqueeze(0)).argmax().item():
            n = n+1
            print(n)

    #DICO METRICS
    rmse_dico = r_diff/n_img
    fooling_rate_dico = n/n_img

    param = 'Lam_Coding: '+str(lambdaCoding) + ', Lam_Reg: ' + str(lambdaReg)
    print('NUM DICO ', D.shape[3])
    print('Attack RMSE DICO ', rmse_dico.item())
    print('FOOLING RATE DICO ', fooling_rate_dico)

    return [rmse_dico.item(), fooling_rate_dico], param
