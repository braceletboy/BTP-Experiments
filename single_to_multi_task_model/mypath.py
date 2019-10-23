'''
@file: mypath.py

This file contains the path configurations for datasets and pre-trianed
model.

@contributor: Rukmangadh Sai Myana
@mail: rukman.sai@gmail.com
'''


class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        '''
        Return the path to the root directories of datasets.

        @param dataset: The name of the dataset.
        @returns: The path to the root directory.
        '''
        if dataset == 'pascal':
            # folder that contains VOCdevkit/.
            return '/path/to/datasets/VOCdevkit/VOC2012/'
        elif dataset == 'sbd':
            # folder that contains dataset/.
            return '/path/to/datasets/benchmark_RELEASE/'
        elif dataset == 'cityscapes':
            # folder that contains leftImg8bit/
            return '/home/Drive3/Rukmangadh/cityscapes/'
        elif dataset == 'coco':
            # folder that contains ....
            return '/path/to/datasets/coco/'
        elif dataset == 'nyu':
            # folder that contains nyu_depth_v2_labeled.mat
            return '/home/Drive3/Rukmangadh/nyu_depth_v2_labeled.mat'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError

    @staticmethod
    def pm_root_dirs(target_problem):
        '''
        Return the path to the root directories of pretrained models.

        @param target_problem: The target problem for which the models were
        pretrained.
        @returns: A list of root directories.
        '''
        if target_problem == 'segmentation':
            return [
                '/home/Drive3/Rukmangadh/pytorch-deeplab-xception',
                '/home/Drive3/Rukmangadh/pytorch-deeplab-xception2'
            ]
        else:
            print('No pretrained model for the {}-target '
                  'problem.'.format(target_problem))
            raise NotImplementedError
