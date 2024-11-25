import torch
import torch.nn.functional as F
from utils.experiment import make_nograd_func
from torch.autograd import Variable
from torch import Tensor
from ptflops import get_model_complexity_info
import wandb
import os
from datetime import datetime
import time


def prepare_input(resolution):
    x1 = torch.FloatTensor(1, *resolution)
    x2 = torch.FloatTensor(1, *resolution)
    return {"left": x1, "right": x2}

def get_model_size(mdl):
    # now = datetime.now()
    # file_name = now.strftime("%H%M%S")
    file_name= round(time.time() * 1000)
    torch.save(mdl.state_dict(), f"{file_name}.pt")
    #print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    model_size = os.path.getsize(f"{file_name}.pt")/1e6
    os.remove(f'{file_name}.pt')

    return model_size

def log_sizes(model):
    ckpt_size = get_model_size(model)
    feat_ckpt_size = get_model_size(model.feature_extraction)
    print("Model Size: %.2f MB" %ckpt_size)
    print("Feature Extraction Size: %.2f MB" %feat_ckpt_size)
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (3, 256, 512), input_constructor= prepare_input, as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        print('{:<50}  {:<8}'.format('Computational complexity (size= 3 x 256x 512): ', macs))
        print('{:<50}  {:<8}'.format('Number of parameters: ', params))

    wandb.log({'Feature_ext_size_MB': feat_ckpt_size,
               'Model_size_MB': ckpt_size,
               "Operations_GMac": float(macs.split()[0]),
               "Parameters_M": float(params.split()[0])
               })


def check_shape_for_metric_computation(*vars):
    assert isinstance(vars, tuple)
    for var in vars:
        assert len(var.size()) == 3
        assert var.size() == vars[0].size()

# a wrapper to compute metrics for each image individually
def compute_metric_for_each_image(metric_func):
    def wrapper(D_ests, D_gts, masks, *nargs):
        check_shape_for_metric_computation(D_ests, D_gts, masks)
        bn = D_gts.shape[0]  # batch size
        results = []  # a list to store results for each image
        # compute result one by one
        for idx in range(bn):
            # if tensor, then pick idx, else pass the same value
            cur_nargs = [x[idx] if isinstance(x, (Tensor, Variable)) else x for x in nargs]
            if masks[idx].float().mean() / (D_gts[idx] > 0).float().mean() < 0.1:
                print("masks[idx].float().mean() too small, skip")
            else:
                ret = metric_func(D_ests[idx], D_gts[idx], masks[idx], *cur_nargs)
                results.append(ret)
        if len(results) == 0:
            print("masks[idx].float().mean() too small for all images in this batch, return 0")
            return torch.tensor(0, dtype=torch.float32, device=D_gts.device)
        else:
            return torch.stack(results).mean()
    return wrapper

@make_nograd_func
@compute_metric_for_each_image
def D1_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = (E > 3) & (E / D_gt.abs() > 0.05)
    return torch.mean(err_mask.float())

@make_nograd_func
@compute_metric_for_each_image
def Thres_metric(D_est, D_gt, mask, thres):
    assert isinstance(thres, (int, float))
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = E > thres
    return torch.mean(err_mask.float())

# NOTE: please do not use this to build up training loss
@make_nograd_func
@compute_metric_for_each_image
def EPE_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    return F.l1_loss(D_est, D_gt, size_average=True)


def calc_error(est_disp=None, gt_disp=None, lb=None, ub=None):
    """
    Args:
        est_disp (Tensor): in [BatchSize, Channel, Height, Width] or
            [BatchSize, Height, Width] or [Height, Width] layout
        gt_disp (Tensor): in [BatchSize, Channel, Height, Width] or
            [BatchSize, Height, Width] or [Height, Width] layout
        lb (scalar): the lower bound of disparity you want to mask out
        ub (scalar): the upper bound of disparity you want to mask out
    Output:
        dict: the error of 1px, 2px, 3px, 5px, in percent,
            range [0,100] and average error epe
    """
    error1 = torch.Tensor([0.])
    error2 = torch.Tensor([0.])
    error3 = torch.Tensor([0.])
    error5 = torch.Tensor([0.])
    epe = torch.Tensor([0.])


    if (not torch.is_tensor(est_disp)) or (not torch.is_tensor(gt_disp)):
        return {
            '1px': error1 * 100,
            '2px': error2 * 100,
            '3px': error3 * 100,
            '5px': error5 * 100,
            'epe': epe
        }

    assert torch.is_tensor(est_disp) and torch.is_tensor(gt_disp)
    assert est_disp.shape == gt_disp.shape

    est_disp = est_disp.clone().cpu()
    gt_disp = gt_disp.clone().cpu()

    mask = torch.ones(gt_disp.shape, dtype=torch.uint8)
    if lb is not None:
        mask = mask & torch.tensor(gt_disp > lb, dtype=torch.uint8)
    if ub is not None:
        mask = mask & torch.tensor(gt_disp < ub, dtype=torch.uint8)
    mask.detach_()
    if abs(mask.sum()) < 1.0:
        return {
            '1px': error1 * 100,
            '2px': error2 * 100,
            '3px': error3 * 100,
            '5px': error5 * 100,
            'epe': epe
        }

    gt_disp = gt_disp[mask]
    est_disp = est_disp[mask]

    abs_error = torch.abs(gt_disp - est_disp)
    total_num = mask.float().sum()

    error1 = torch.sum(torch.gt(abs_error, 1).float()) / total_num
    error2 = torch.sum(torch.gt(abs_error, 2).float()) / total_num
    error3 = torch.sum(torch.gt(abs_error, 3).float()) / total_num
    error5 = torch.sum(torch.gt(abs_error, 5).float()) / total_num
    epe = abs_error.float().mean()

    return {
        'psm_1px': error1 * 100,
        'psm_2px': error2 * 100,
        'psm_3px': error3 * 100,
        'psm_5px': error5 * 100,
        'psm_epe': epe
    }
