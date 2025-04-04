from __future__ import print_function, division
import argparse
import os
from builtins import int

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__, model_loss, __loss_type__
from utils import *
from torch.utils.data import DataLoader
import gc

import wandb
from utils.metrics import log_sizes
# from torchinfo import summary

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='LeanStereoNet')
parser.add_argument('--model', default='leanstereo', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--trainlist', required=True, help='training list')
parser.add_argument('--testlist', required=True, help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=8, help='testing batch size')
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

parser.add_argument('--wandb_project_name', type=str, default="test", help='name of the weights and baises project')
parser.add_argument('--wandb_run_name', type=str, default="test",
                    help='name of the run inside weights and baises project')

parser.add_argument('--loss_type', default='smoothL1', help='select which function to use as loss',
                    choices=__loss_type__.keys())
parser.add_argument('--aux_mode', default='train', help='select a model mode')
parser.add_argument('--error_optimization_criteria', type=str, default="EPE",
                    help='optimize the model for EPE or D1', choices={"EPE","D1"})
__optimizer__ ={
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD
}
parser.add_argument('--optimizer', default='adam', help='optimizer to use', choices=__optimizer__.keys())


# parse arguments, set seeds
args = parser.parse_args()
# print(args)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True)
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=18, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](args)

# Calculate computational complexity of the model
wandb.init(project=args.wandb_project_name, config=args,
                    settings=wandb.Settings(start_method='fork'))
log_sizes(model)

model = nn.DataParallel(model)
model.cuda()

if args.optimizer == "sgd":
    kwargs= {"lr":args.lr, "momentum":0.9}
else:
    kwargs={"lr":args.lr, "betas":(0.9, 0.999)}
optimizer = __optimizer__[args.optimizer](model.parameters(), **kwargs)
# print(model)

loss_type = args.loss_type
criteria= args.error_optimization_criteria

# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'],strict=False)

print("start at epoch {}".format(start_epoch))
# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

def train():
    wandb.watch(model, log='gradients', log_freq=args.summary_freq)
    error = 100
    best_epoch = 1

    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, compute_metrics=do_summary)
            if do_summary:
                wandb.log({'train_loss': scalar_outputs['loss'],
                           'train_EPE': scalar_outputs['EPE'][-1],
                           'train_D1': scalar_outputs['D1'][-1],
                           'train_Thres1': scalar_outputs['Thres1'][-1],
                           'train_Thres2': scalar_outputs['Thres2'][-1],
                           'train_Thres3': scalar_outputs['Thres3'][-1],
                           'train_step': global_step,
                           'epoch': epoch_idx
                           })

                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss,
                                                                                       time.time() - start_time))
        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))

            # torch.onnx.export(model, sample ,"checkpoint_{:0>6}.onnx".format(epoch_idx))
            wandb.save("checkpoint_{:0>6}.onnx".format(epoch_idx))
        gc.collect()

        # testing
        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TestImgLoader):
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            val_loss, scalar_outputs, image_outputs = test_sample(sample, compute_metrics=do_summary)
            avg_test_scalars.update(scalar_outputs)

            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)

            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                     batch_idx,
                                                                                     len(TestImgLoader), val_loss,
                                                                                     time.time() - start_time))
        avg_test_scalars = avg_test_scalars.mean()
        save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        print("avg_test_scalars", avg_test_scalars)

        wandb.log({'val_loss': avg_test_scalars['loss'],
                   'val_EPE': avg_test_scalars['EPE'][-1],
                   'val_D1': avg_test_scalars['D1'][-1],
                   'val_Thres1': avg_test_scalars['Thres1'][-1],
                   'val_Thres2': avg_test_scalars['Thres2'][-1],
                   'val_Thres3': avg_test_scalars['Thres3'][-1],
                   'val_step': global_step,
                   'val_epoch': epoch_idx
                   })

        if avg_test_scalars[criteria][-1] < error:
            error = avg_test_scalars[criteria][-1]
            best_epoch = epoch_idx
            print("New best checkpoint found!!")

            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_best.ckpt".format(args.logdir))

        wandb.log({'Best ckpt epoch': best_epoch,
                   'epoch': epoch_idx
                   })
        gc.collect()


# train one sample
def train_sample(sample, compute_metrics=False):
    model.train()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    optimizer.zero_grad()

    disp_ests = model(imgL, imgR)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)

    loss = model_loss(disp_ests, disp_gt, mask, __loss_type__[loss_type])

    # if loss_type == "ohemCE":
    #     disp_ests = [torch.argmax(d, axis=1).type(torch.float32) for d in disp_ests]

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            image_outputs["errormap"] = [disp_error_image_func.apply(disp_est, disp_gt) for disp_est in disp_ests]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    loss.backward()
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    disp_ests = model(imgL, imgR)

    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)

    loss = model_loss(disp_ests, disp_gt, mask, __loss_type__[loss_type])

    # if loss_type == "ohemCE":
    #     disp_ests = [torch.argmax(d, axis=1).type(torch.float32) for d in disp_ests]

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    if compute_metrics:
        image_outputs["errormap"] = [disp_error_image_func.apply(disp_est, disp_gt) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs

if __name__ == '__main__':

    args = wandb.config  # just to ensure same params are logged in wandb and also used same in our model
    wandb.run.name = args.wandb_run_name

    train()

    wandb.save("{}/checkpoint_best.ckpt".format(args.logdir))
