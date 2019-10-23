'''
@file: deeplab.py

This file contains the class for the DeepLab V3 Plus neural network model.

@contributor: Rukmangadh Sai (not the original author)
@mail: rukman.sai@gmail.com
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone


class DeepLab(nn.Module):
    '''
    Class for the DeepLab V3 Plus based multi-task network.
    '''
    def __init__(self,
                 backbone='resnet',
                 output_stride=16,
                 num_classes=21,
                 sync_bn=True,
                 freeze_bn=False):
        '''
        Initialize the given instance with the given parameters.

        @param backbone: The backbone for the DeepLab V3 Plus.
        @param output_stride: The ratio of, the sizes of the output of the
        DeepLab V3 Plus encoder and it's input.
        @param num_classes: The number of tasks in our segmentation task.
        @param sync_bn: Whether to use synchronized batch norm or not.
        @param freeze_bn: Whether to freeze the BN parameters or not.
        '''
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            raise RuntimeWarning("The output_stride value can only be 8 for "
                                 "the drn backbone. {} was given. "
                                 "Changing it to 8.".format(output_stride))
            output_stride = 8

        if sync_bn is True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x,
                          size=input.size()[2:],
                          mode='bilinear',
                          align_corners=True)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if (isinstance(m[1], nn.Conv2d)
                    or isinstance(m[1], SynchronizedBatchNorm2d)
                        or isinstance(m[1], nn.BatchNorm2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if (isinstance(m[1], nn.Conv2d)
                    or isinstance(m[1], SynchronizedBatchNorm2d)
                        or isinstance(m[1], nn.BatchNorm2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())
