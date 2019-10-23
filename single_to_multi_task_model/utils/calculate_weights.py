'''
@file: calculate_weights.py

This file contains the function for calculating the weights for the classes
based on their frequency in the dataset.

@contributor: Rukmangadh Sai Myana (not the original author)
@mail: rukman.sai@gmail.com
'''
import os
from tqdm import tqdm
import numpy as np
from mypath import Path


def calculate_weights_labels(dataset, dataloader, num_classes):
    '''
    Calculate the weights of the classes in the dataset.

    These weights are useful to remove any imbalance caused due to one class
    having more number of samples in the dataset.

    @param dataset: The dataset name.
    @param dataloader: The dataloader for the given dataset.
    @param num_classes: The number of classes in the given dataset.
    @returns: The weight vector for the classes.
    '''
    # Create an instance from the data loader
    z = np.zeros((num_classes, ))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        y = sample['label']
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    classes_weights_path = os.path.join(Path.db_root_dir(dataset),
                                        dataset + '_classes_weights.npy')
    np.save(classes_weights_path, ret)

    return ret
