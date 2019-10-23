import numpy as np


class Evaluator(object):
    '''
    Class for evaluating the model using various metrics.
    '''
    def __init__(self, num_class):
        '''
        Initialize the instance with the given papameters

        @param num_class: The number of classes for the segmentation.
        '''
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class, ) * 2)

    def Pixel_Accuracy(self):
        '''
        Compute the pixel accuray.
        '''
        Acc = np.diag(
            self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        '''
        Obtain the class with
        '''
        Acc = np.diag(
            self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        '''
        Compute MIoU.
        '''
        MIoU = np.diag(
            self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) +
                                      np.sum(self.confusion_matrix, axis=0) -
                                      np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        '''
        Compute Frequency weighted IoU.
        '''
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(
            self.confusion_matrix)
        iu = np.diag(
            self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) +
                                      np.sum(self.confusion_matrix, axis=0) -
                                      np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        '''
        Add the given data to the evalution.

        @param gt_image:
        @param pre_image:
        '''
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        '''
        Reset the instance.
        '''
        self.confusion_matrix = np.zeros((self.num_class, ) * 2)
