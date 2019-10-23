#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@file: encoder.py

This file contains the encoder of the DeepLab V3 Plus model

@author: Rukmangadh Sai Myana.
@mail: rukman.sai@gmail.com
'''

import torch.nn as nn
from modeling.aspp import build_aspp
from modeling.backbone import build_backbone
from modeling.sync_batchnorm import SynchronizedBatchNorm2d


class Encoder(nn.Module):
    '''
    Class for the encoder of the DeepLab V3 Plus model.
    '''
    def __init__(self,
                 backbone,
                 output_stride,
                 BatchNorm):
        '''
        Initialize the instance with the given parameters.

        @param backbone: The backbone that is to be used while building this
        model.
        @param output_stride: The ratio of the size of the output of this
        module to the input of this module.
        @param BatchNorm: The batch norm instance to be used.
        '''
        super(Encoder, self).__init__()
        if backbone == 'drn':
            raise RuntimeWarning("The output_stride value can only be 8 for "
                                 "the drn backbone. {} was given. "
                                 "Changing it to 8.".format(output_stride))
            output_stride = 8

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)

    def foward(self, input):
        '''
        The forward pass function of the encoder module.

        @param input: The input of the encoder.
        @returns: The enriched features for the decoder input.
        '''
        backbone_features, low_level_features = self.backbone
        features = self.aspp(backbone_features)
        return features, low_level_features

    def get_1x_lr_params(self):
        '''
        Build the iterator for the parameter group that needs to vary at
        1*lr rate.

        @returns: The iterator.
        '''
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if (isinstance(m[1], nn.Conv2d)
                    or isinstance(m[1], SynchronizedBatchNorm2d)
                        or isinstance(m[1], nn.BatchNorm2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


def build_encoder(backbone, output_stride, BatchNorm):
    return Encoder(backbone, output_stride, BatchNorm)


if __name__ == "__main__":
    pass
