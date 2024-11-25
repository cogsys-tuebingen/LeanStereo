from models.leanstereonet import LeanStereo
from models.loss import model_loss, LogL1Loss, LogL1Loss_v2

from functools import partial
import torch.nn.functional as F

__models__ = {
    "leanstereo": LeanStereo
}

__loss_type__ = {
    "smoothL1": partial(F.smooth_l1_loss, reduction="mean"),
    "huber": partial(F.huber_loss),
    "l2": partial(F.mse_loss),
    "LogL1Loss": LogL1Loss(),
    "LogL1Loss_v2": LogL1Loss_v2()
    # "ohemCE": OhemCELoss()
}




