import time

import numpy as np
import torch

def get_args(args):
    """ Output all possible combination of arguments
    nb: try to write it recursively
    """
    num_args = len(args)
    num_var = num_args // 2

    pass_args = []
    if num_var > 0:
        val1 = args[0]
        p1 = args[1]
        for ii in p1:

            if num_var > 1:
                val2 = args[2]
                p2 = args[3]
                for jj in p2:

                    if num_var > 2:
                        val3 = args[4]
                        p3 = args[5]
                        for kk in p3:
                            pass_args.append({val1: ii, val2: jj, val3: kk})

                    else:
                        pass_args.append({val1: ii, val2: jj})
            else:
                pass_args.append({val1: ii})
    else:
        pass_args.append(dict())

    return pass_args


def get_atks(model, atk, *args, **kwargs):
    """ Gather all attacks with all possible hyper-parameters values passed as *args """

    args_list = get_args(args)
    atks = []
    for arg_val in args_list:
        kwargs.update(arg_val)
        atks.append(atk(model, **kwargs))
    return atks


def select_hyperparameter(atks_hyper, model, data, budget, criterion='mse_limit', device=torch.device('cpu')):
    """ Select the hyper-parameters of attacks according to some budget """

    # Compute the performance
    validation_perf = get_performance(atks_hyper, model=model, data=data, verbose=True, device=device)
    mse = validation_perf['mse']
    rmse = validation_perf['rmse']
    fooling_rate = validation_perf['fooling_rate']
    atks_name = rmse.keys()

    # Select according to the budget and the criterion
    atks_selected = []
    perf = []
    for budget_val in budget:
        res_atks = dict()
        res_fooling = dict()
        res_rmse = dict()
        res_mse = dict()
        for key in atks_name:
            if criterion == 'rmse':
                ind = np.argmin(np.abs(np.array(rmse[key]) - budget_val))
            elif criterion == 'mse':
                ind = np.argmin(np.abs(np.array(mse[key]) - budget_val))
            elif criterion == 'fooling_rate':
                # ind = np.argmin(np.abs(np.array(fooling_rate[key]) - budget_val))
                vmin = np.abs(np.array(fooling_rate[key]) - budget_val)
                ind_min = np.where(vmin == vmin.min())
                ind_min_mse = np.argmax(np.array(rmse[key])[ind_min[0]])  # try the one with largest mse
                ind = ind_min[0][ind_min_mse]
            elif criterion == 'mse_limit':
                # Pick the hyper-parameters for which the mse budget does not exceed
                vmse = np.array(mse[key]) - budget_val
                ind_mse_admissible = np.where(vmse <= 0)[0]
                # From those, pick the ones maximizing the fooling rate
                if len(ind_mse_admissible) > 0:
                    vfr = np.array(fooling_rate[key])[ind_mse_admissible]
                    ind_maxfool = np.where(vfr == vfr.max())[0]
                    # If there is an ambiguity pick the ones maximizing the mse
                    ind_tmp = ind_mse_admissible[ind_maxfool]
                    ind_maxmse = np.argmax(np.array(mse[key])[ind_tmp])
                    # Then we get the index
                    ind = ind_tmp[ind_maxmse]
                else:
                    ind = np.nan
            else:
                raise ValueError
            if np.isnan(ind):
                res_fooling.update({key: np.nan})
                res_rmse.update({key: np.nan})
                res_mse.update({key: np.nan})
                res_atks.update({key: []})
            else:
                res_fooling.update({key: fooling_rate[key][ind]})
                res_rmse.update({key: rmse[key][ind]})
                res_mse.update({key: mse[key][ind]})
                res_atks.update({key: [atks_hyper[key][ind]]})
        perf.append({'fooling_rate': res_fooling, 'rmse': res_rmse, 'mse': res_mse})
        atks_selected.append(res_atks)

    return atks_selected, perf, validation_perf


# ---------- COMPUTE PERFORMANCE ----------- #


def get_performance(atks, model, data, verbose=False, device=torch.device('cpu')):
    fooling_rate = dict()
    rmse = dict()
    mse = dict()

    # Get through all types of attacks
    for name in atks.keys():
        if verbose:
            print(name, '...')

        fooling_rate_tmp = []
        rmse_tmp = []
        mse_tmp = []

        for atk in atks[name]:
            print('Attack Image learning with {}'.format(atk))
            start = time.time()
            perf_tmp = performance(attack=atk, model=model.to(device=device), data=data, device=device)
            end = time.time()
            print('time costing {}s'.format(end - start))
            print('performance: {}'.format(perf_tmp))
            fooling_rate_tmp.append(perf_tmp['fooling_rate'])
            rmse_tmp.append(perf_tmp['rmse'])
            mse_tmp.append(perf_tmp['mse'])

        fooling_rate.update({name: fooling_rate_tmp})
        rmse.update({name: rmse_tmp})
        mse.update({name: mse_tmp})

    return {'fooling_rate': fooling_rate, 'rmse': rmse, 'mse': mse}


def performance(attack, model, data, device=torch.device('cpu')):
    num_samples = data.dataset.__len__()
    fooling = 0
    rmse = 0
    mse = 0
    device = attack.device
    for x, y in data:
        # Assign to the device
        x, y = x.to(device=device), y.to(device=device)
        # start = time.time()
        adversary = attack(x, y)
        # print('single batch adversarial image processing time: {}'.format(time.time()-start))
        fooling += compute_fooling_rate(model=model.eval(), adversary=adversary, clean=x)
        rmse += compute_rmse(adversary=adversary, clean=x)
        mse += compute_mse(adversary=adversary, clean=x)
    perf = {
        "fooling_rate": fooling / num_samples,
        "rmse": rmse / num_samples,
        "mse": mse / num_samples
    }
    return perf


# ---------- COMPUTE TRANSFER PERFORMANCE ----------- #


def get_transfer_performance(atks, models, data, device=torch.device('cpu')):
    """ Given a dictionary of attacks, it returns all performance on various models
    -> auxiliary function: get_transfer_performance_aux
    """
    perf_transfer = dict()
    for name in atks.keys():
        if len(atks[name]) > 0:
            perf_tmp = get_transfer_performance_aux(atks[name][0], models, data=data, device=device)
        else:
            perf_tmp = empty_transfer_performance(models)
        perf_transfer.update({name: perf_tmp})

    return perf_transfer


def empty_transfer_performance(model_transfer):
    perf = dict()
    for name_tmp in model_transfer.keys():
        perf[name_tmp] = dict({'fooling_rate': np.nan, 'rmse': np.nan, 'mse': np.nan})
    return perf


def get_transfer_performance_aux(attack, model_transfer, data, device=torch.device('cpu')):
    """ Given a specific attack type, return the transfer performance on various models """

    # Parameters
    num_samples = data.dataset.__len__()

    # Initialize performance
    perf = dict()
    for name_tmp in model_transfer.keys():
        perf[name_tmp] = dict({'fooling_rate': 0., 'rmse': 0., 'mse': 0.})

    # Go through all samples
    for x, y in data:
        x, y = x.to(device=device), y.to(device=device)
        adversary = attack(x, y)

        # Compute performance on all models
        for model_name in model_transfer.keys():
            # Loading model
            model = model_transfer[model_name].to(device=device)

            # Compute performance
            perf[model_name]['fooling_rate'] += compute_fooling_rate(model=model, adversary=adversary,
                                                                     clean=x) / num_samples
            perf[model_name]['rmse'] += compute_rmse(adversary=adversary, clean=x) / num_samples
            perf[model_name]['mse'] += compute_mse(adversary=adversary, clean=x) / num_samples

    return perf


# ---------- PERFORMANCE CRITERIA ----------- #


def compute_fooling_rate(model, adversary, clean, reduction='sum'):
    """ Compute the fooling rate """
    label_clean = model.eval()(clean).argmax(dim=1)
    label_adversary = model.eval()(adversary).argmax(dim=1)
    label_different = (label_clean != label_adversary)
    if reduction == 'sum':
        return label_different.float().sum().item()
    elif reduction == 'mean':
        return label_different.float().mean().item()


def compute_rmse(adversary, clean, reduction='sum'):
    """ Compute the relative mean squared error """
    upper = torch.sum((adversary - clean) ** 2, dim=[1, 2, 3])
    lower = torch.sum(clean ** 2, dim=[1, 2, 3])
    ratio = upper / lower
    if reduction == 'sum':
        return torch.sum(ratio).item()
    elif reduction == 'mean':
        return torch.mean(ratio).item()


def compute_mse(adversary, clean, reduction='sum'):
    """ Compute the mean squared error """
    upper = torch.sum((adversary - clean) ** 2, dim=[1, 2, 3])
    if reduction == 'sum':
        return torch.sum(upper).item()
    elif reduction == 'mean':
        return torch.mean(upper).item()
