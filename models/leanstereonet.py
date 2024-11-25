from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
import math
from models.backbone import FeatureExtractionNet



class feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature
        self.features = FeatureExtractionNet()
        if self.concat_feature:
            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

    def forward(self, x):

        features = self.features(x)
        if not self.concat_feature:
            return {"features": features}
        else:
            concat_feature = self.lastconv(features)
            return {"features": features, "concat_feature": concat_feature}
        # return features


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6


class LeanStereoNet(nn.Module):
    def __init__(self, args, use_concat_volume=False):
        super(LeanStereoNet, self).__init__()
        self.maxdisp = args.maxdisp
        self.use_concat_volume = use_concat_volume
        self.aux_mode = args.aux_mode

        # nclasses = args.nclasses
        self.num_groups = int( self.maxdisp/ 8)  # BGA layer maps to 128 feat maps

        self.apply_regression= False if args.loss_type == "ohemCE" else True

        self.concat_channels = 32
        self.concatconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(128, self.concat_channels, kernel_size=1, padding=0, stride=1,
                                                  bias=False))
        # if self.aux_mode == "train":

        self.patch = nn.Conv3d(40, 40, kernel_size=(1, 3, 3), stride=1, dilation=1, groups=40, padding=(0, 1, 1),
                               bias=False)
        self.patch_l1 = nn.Conv3d(8, 8, kernel_size=(1, 3, 3), stride=1, dilation=1, groups=8, padding=(0, 1, 1),
                                  bias=False)
        self.patch_l2 = nn.Conv3d(16, 16, kernel_size=(1, 3, 3), stride=1, dilation=2, groups=16, padding=(0, 2, 2),
                                  bias=False)
        self.patch_l3 = nn.Conv3d(16, 16, kernel_size=(1, 3, 3), stride=1, dilation=3, groups=16, padding=(0, 3, 3),
                                  bias=False)
        self.dres1_att = nn.Sequential(convbn_3d(40, 16, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn_3d(16, 16, 3, 1, 1))
        self.dres2_att = hourglass(16)
        self.classif_att = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))

        if self.use_concat_volume:
            # self.concat_channels = 12
            self.feature_extraction = feature_extraction(concat_feature=True,
                                                         concat_feature_channel=self.concat_channels)
        else:
            # self.concat_channels = 0
            self.feature_extraction = feature_extraction(concat_feature=False)

        self.dres0 = nn.Sequential(convbn_3d(self.concat_channels * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        # nn.init.normal(self.sum_weights, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def build_acvnet_volume(self, features_left, features_right):
        gwc_volume1 = build_gwc_volume(features_left, features_right, self.maxdisp // 4,
                                       num_groups=40)
        gwc_volume = self.patch(gwc_volume1)  # >>>>>>> Gwc-p <<<<<<<<<<<<
        patch_l1 = self.patch_l1(gwc_volume[:, :8])
        patch_l2 = self.patch_l2(gwc_volume[:, 8:24])
        patch_l3 = self.patch_l3(gwc_volume[:, 24:40])
        patch_volume = torch.cat((patch_l1, patch_l2, patch_l3), dim=1)  # >>>>>>> Gwc-mp <<<<<<<<<<<<
        cost_attention = self.dres1_att(patch_volume)
        cost_attention = self.dres2_att(cost_attention)  # >>>>>>> Gwc-mp-hg <<<<<<<<<<<<
        att_weights = self.classif_att(cost_attention)
        return att_weights

    def forward(self, left, right):
        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)


        concat_feature_left = self.concatconv(features_left["features"])
        concat_feature_right = self.concatconv(features_right["features"])
        gwc_volume= build_concat_volume(concat_feature_left, concat_feature_right, self.maxdisp // 4)

        att_weights = self.build_acvnet_volume(features_left["features"], features_right["features"])
        volume= F.softmax(att_weights, dim=2) * gwc_volume

        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)

        if self.aux_mode == "train":
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)

            cost0 = F.upsample(cost0, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost0 = torch.squeeze(cost0, 1)

            cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)

            cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)

            if self.apply_regression:
                pred0 = F.softmax(cost0, dim=1)
                pred0 = disparity_regression(pred0, self.maxdisp)

                pred1 = F.softmax(cost1, dim=1)
                pred1 = disparity_regression(pred1, self.maxdisp)

                pred2 = F.softmax(cost2, dim=1)
                pred2 = disparity_regression(pred2, self.maxdisp)
            else:
                pred0, pred1, pred2 = cost0, cost1, cost2


            return [pred0, pred1, pred2]

        else:
            cost2 = self.classif2(out2)
            cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)
            return [pred2]

def LeanStereo(args):
    return LeanStereoNet(args, use_concat_volume=False)
