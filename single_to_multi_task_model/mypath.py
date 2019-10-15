class Path(object):
    @staticmethod
    def db_root_dir(dataset):
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
            # folder that contains 
            return '/path/to/datasets/coco/'
        elif dataset == 'nyu':
            # folder that contains nyu_depth_v2_labeled.mat
            return '/home/Drive3/Rukmangadh/nyu_depth_v2_labeled.mat'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
