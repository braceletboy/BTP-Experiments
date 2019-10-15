'''
@file: nyu.py

This file contains the data loader for the NYU Depth Dataset

@author: Rukmangadh Sai Myana
@mail: rukman.sai@gmail.com
'''

import h5py
import numpy as np
from mypath import Path
from torch.utils import data
from torchvision import transforms
from dataloaders import custom_transforms as tr

class NYUDepthSegmentation(data.Dataset):
    '''
    Class for the NYU Depth Dataset.
    '''
    NUM_CLASSES = 895

    def __init__(self, args, root=Path.db_root_dir('nyu'), split="train"):
        '''
        Initialize the class.
        
        @param root: The path to the NYU Depth Dataset mat file
        @param split: "train"/"val"
        '''
        self.root = root
        self.split = split
        self.args = args
        with h5py.File(self.root, 'r') as dataset_file:
            all_targets = np.array(dataset_file['labels'])
            all_images = np.array(dataset_file['images'])
            num_all_samples = labels.shape[0]
            num_val_samples = (2*self.num_all_samples)//10
            num_train_samples = (7*self.num_all_samples)//10
            num_test_samples = (num_all_samples - num_train_samples - 
                    num_val_samples)
            if self.split=="train":
                self.num_samples = num_train_samples
                self.images = all_images[:num_train_samples]
                self.targets = all_targets[:num_train_samples]
            elif self.split=="val":
                self.num_samples = num_val_sampless
                self.images = all_images[
                    num_train_samples:num_train_samples+num_val_samples]
                self.targets = all_targets[
                    num_train_samples:num_train_samples+num_val_samples]
            elif self.split=="test":
                self.num_samples = num_test_samples
                self.images = all_images[num_train_samples+num_val_samples:
                    num_train_samples+num_val_samples+num_test_samples]
                self.targets = all_targets[num_train_samples+num_val_samples:
                    num_train_samples+num_val_samples+num_test_samples]
            else:
                raise ValueError("Unknown value for split argument was given."
                " Only 'train', 'val' and 'test' are allowed")
            
    def __len__(self):
        '''
        Magic method that gives the number of samples in the dataset.
        '''
        self.num_samples
        
    def __getitem__(self, index):
        '''
        Magic method for obtaining the index-ed item.
        '''
        _img = self.images[index]
        _target = self.targets[index]
        sample = {'image' : _img, 'label' : _targe}
        if split == "train":
            return self.transform_tr(sample)
        elif split == "val":
            return self.transform_val(sample)
        elif split == "test":
            return self.transform_ts(sample)
        else:
            raise ValueError("Unknown value for split argument was given."
                " Only 'train', 'val' and 'test' are allowed")
    
    def __str__(self):
        '''
        The string form of the class.
        '''
        return 'NYUDepthV2(split=' + str(self.split) + ')'
        
    def transform_tr(self, sample):
        '''
        Transform the given training sample.
        
        @param sample: The given training sample.
        '''
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, 
                crop_size=self.args.crop_size, fill=255),
            tr.RandomGaussianBlur(),
            tf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        
        return composed_transforms(sample)
        
    def transform_val(self, sample):
        '''
        Transform the given validation sample.
        
        @param sample: The given validation sample.
        '''
        composed_transforms = transforms.Compose([
            tr.FixedScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tf.ToTensor()])
            
        return composed_transforms(sample)

    def transform_ts(self, sample):
        '''
        Transform the given test sample.
        
        @param sample: The given test sample.
        '''
        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tf.ToTensor()])
            
        return composed_transforms(sample)







