#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@file: load_models.py

This file contains useful functions for loading pre trained models.

@author: rukmangadh sai myana
@mail: rukman.sai@gmail.com
'''


import os
import torch
from ..mypath import Path
from ..modeling.deeplab import DeepLab
from glob import glob


def load_teacher_models(args, filename='checkpoint.pth.tar', **kwargs):
    '''
    Load the single task teacher models.

    @param args: The arguments from the argparse in the main file - train.py
    @param filename: The name under which checkpoints are made for each teacher
    model
    @kwargs:
        num_classes: The number of classes in the segmentation teacher models.
    '''
    root_dirs_list = Path.pm_root_dirs('segmentation')
    assert 'num_classes' in kwargs
    num_classes = kwargs['num_classes']
    models_list = []
    for root_dir in root_dirs_list:
        directory = os.path.join(root_dir, 'run', args.dataset,
                                 args.checkname)
        runs = sorted(glob(os.path.join(directory, 'experiment_*')))
        latest_experiment_dir = runs[-1]

        if args.cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        checkpoint = torch.load(os.path.join(latest_experiment_dir, filename),
                                map_location=device)
        model = DeepLab(args.backbone, args.output_stride, num_classes,
                        args.sync_bn, args.freeze_bn)
        model.load_state_dict(checkpoint['state_dict'])
        if args.cuda:
            model.cuda()
        models_list.append(model)
