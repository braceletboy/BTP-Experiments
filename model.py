'''
@file: run.py

This file contains all the models used in the experiments.

@author: Rukmangadh Sai Myana
@mail: rukman.sai@gmail.com
'''


import torch
import torch.nn as nn


_LAYER_MAP = {
    'conv2d': nn.Conv2d,
    'relu': nn.ReLU,
    'softmax': nn.Softmax,
    'linear': nn.Linear,
    'batch_norm1d': nn.BatchNorm1d,
    'batch_norm2d': nn.BatchNorm2d,
    'dropout': nn.Dropout,
    'dropout2d': nn.Dropout2d,
}


class BasicClassificationCNN(nn.Module):
    '''
    This class represents a basic CNN model that is used for the classfication
    of the MNIST Dataset.

    It contains one or more (convolution+pooling) blocks followed by a fully
    connected network. There are three configurations availabe - BCCNN1,
    BCCNN2, BCCNN3.
    '''

    def __init__(self, network_definition):
        '''
        Initilize the model.

        @param network_definition: The definition of the model's computational
        network.
        '''
        # initalize the parent class
        super(BasicClassificationCNN, self).__init__()

        self.conv_network = []
        self.fc_network = []

        # define the model.
        on_conv_network = True
        for layer_definition in network_definition:
            layer_type = layer_definition.pop('layer_type')
            layer_name = layer_definition.pop('layer_name')
            if layer_type == 'flatten':
                on_conv_network = False
                continue  # continue to next layer definition

            # ModuleDict accepts an iterable of type (string, Module)
            if on_conv_network:
                self.conv_network.append(
                    (layer_name, _LAYER_MAP[layer_type](**layer_definition)))
            else:
                self.fc_network.append(
                    (layer_name, _LAYER_MAP[layer_type](**layer_definition)))

        # convert to neural network modules
        self.conv_network = nn.ModuleDict(self.conv_network)
        self.fc_network = nn.ModuleDict(self.fc_network)

    def forward(self, x):
        '''
        Define the forward pass for the model.

        @param x: input
        @returns: output
        '''
        for key in self.conv_network.keys():
            x = self.conv_network[key](x)
        x = torch.flatten(x, 1)
        for key in self.fc_network.keys():
            x = self.fc_network[key](x)
        return x
