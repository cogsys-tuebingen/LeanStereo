from __future__ import print_function, division
import argparse
import torch.backends.cudnn as cudnn
from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader
import wandb
from test_function import test, test_sample
import os
import warnings
warnings.filterwarnings("ignore")
from ptflops.flops_counter import get_model_complexity_info, params_to_string

from models import model_loss, __loss_type__
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='LeanStereoNet')
parser.add_argument('--model', default='leanstereo', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--testlist', required=True, help='testing list')
parser.add_argument('--loadckpt', required=True, help='load the weights from a specific checkpoint')
parser.add_argument('--example_img_freq', default=20, type=int, help='how often error maps and disparitys are created')

parser.add_argument('--wandb_project_name', type=str, default="test_test", help='name of the weights and baises project')
parser.add_argument('--wandb_run_name', type=str, default="test", help='name of the run inside weights and baises project')
parser.add_argument('--loss_type', default='smoothL1', help='select which function to use as loss',
                    choices=__loss_type__.keys())
parser.add_argument('--aux_mode', default='test', help='select a model mode')

# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    feat_size=os.path.getsize("tmp.pt")/1e6
    print("%.2f MB" %(feat_size))
    os.remove('tmp.pt')
    return feat_size

# model, optimizer
model = __models__[args.model](args)
print("Feature extraction size")
feat_size= print_model_size(model.feature_extraction)
model = nn.DataParallel(model)
model.cuda()

# load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])


print("Model size")
model_size= print_model_size(model)

img_freq = args.example_img_freq

with torch.cuda.device(0):
    input_size = (3, 256, 512)
    macs, params = get_model_complexity_info(model, input_size, as_strings=True,
                                             print_per_layer_stat=False, verbose=False,
                                             input_constructor=prepare_input)

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

if __name__ == '__main__':
    with wandb.init(project=args.wandb_project_name, config= args, settings=wandb.Settings(start_method='fork')):
        args= wandb.config #just to ensure same params are logged in wandb and also used same in our model
        wandb.run.name =args.wandb_run_name

        wandb.log({'Feature_ext_size_MB': feat_size,
                   'Model_size_MB': model_size,
                   "Operations_GMac": float(macs.split()[0]),
                   "Parameters_M": float(params.split()[0])
                   })

        test(model, img_freq, TestImgLoader, args.dataset, args.maxdisp, logfilename=args.wandb_run_name)
