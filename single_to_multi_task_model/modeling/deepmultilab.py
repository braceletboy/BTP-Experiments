#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
from modeling.encoder import build_encoder
from modeling.decoder import build_decoder
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class DeepMultiLab(nn.Module):
    '''
    Class for the multi-task deep lab v3 module.
    '''
    def __init__(self,
                 num_tasks=2,
                 backbone='resnet',
                 output_stride=16,
                 num_classes=21,
                 sync_bn=True,
                 freeze_bn=False):
        '''
        Initialize the given instance with the given parameters.

        @param num_tasks: The number of tasks in the multi task model.
        @param backbone: The backbone for the deeplab v3 plus encoder.
        @param output_stride: The ratio of, the sizes of the output of the
        encoder and it's input.
        @param num_classes: the number of tasks in out segmentation task.
        @param sync_bn: Whether to use synchronized batch norm or not.
        @param freeze_bn: Whether to freeze the Batch Norm parameters or not.
        '''
        super(DeepMultiLab, self).__init__()
        if sync_bn is True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.encoder = build_encoder(backbone, output_stride, BatchNorm)
        self.decoder_list = nn.ModuleList([build_decoder(num_classes, backbone,
                                                         BatchNorm)]*num_tasks)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        '''
        Forward pass for the DeepMulitLab model.

        @param input: The input to the model.
        @returns: The model output.
        '''
        features, low_level_features = self.encoder(input)
        output_list = []
        for decoder in self.decoder_list:
            output_features = decoder(features, low_level_features)
            output = F.interpolate(output_features,
                                   size=input.size()[2:],
                                   mode='bilinear',
                                   align_corners=True)
            output_list.append(output)
        return output_list

    def get_1x_lr_params(self):
        '''
        Return the iterator for parameters that need to change at 1*lr rate.

        @returns: The iterator.
        '''
        return self.encoder.get_1x_lr_params()

    def get_10x_lr_params(self):
        '''
        Return the iterator for parameters that are to be changed at 10*lr
        rate.

        @returns: The iterator.
        '''
        modules = [self.encoder.aspp] + self.decoder_list
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if (isinstance(m[1], nn.Conv2d)
                        or isinstance(m[1], SynchronizedBatchNorm2d)
                        or isinstance(m[1], nn.BatchNorm2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    pass
