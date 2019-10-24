import torch
import torch.nn as nn


class MultiTaskLosses(object):
    '''
    The class for losses of multi-task models
    '''
    def __init__(self,
                 weight=None,
                 size_average=False,
                 batch_average=True,
                 cuda=False):
        '''
        Initialize the instance with the given parameters.
        '''
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, weighting_mode='w', loss_mode='kl'):
        '''
        Return the loss function specified.

        @param weighting_mode: 'eq' or ''
        @param loss_mode: The loss being used for the multi-task learning.
        @returns: The specified loss function functor.
        '''
        self.loss_mode = 'kl'
        if weighting_mode == 'eq':
            return self.DefaultMultiTaskLoss

    def DefaultMultiTaskLoss(self, logit_list, target_list):
        '''
        Compute the loss giving equal weightage to all tasks.

        @param logit_list: The list of logits.
        @param target_list: The list of targets.
        '''
        assert len(logit_list) == len(target_list)
        num_tasks = len(logit_list)
        cumm_loss = 0.0
        for idx in range(logit_list):
            logit = logit_list[idx]
            target = target_list[idx]
            if self.loss_mode == 'kl':
                self.criterion = KnowledgeDistillationLosses(
                    weight=self.weight,
                    cuda=self.cuda,
                    size_average=self.size_average,
                    batch_average=self.batch_average).build_loss(self.loss_mode)
            else:
                raise NotImplementedError
            loss = self.criterion(logit, target)
            cumm_loss += loss
        cumm_loss /= num_tasks
        return cumm_loss


class KnowledgeDistillationLosses(object):
    '''
    Class for Knowledge Distillation Losses.
    '''
    def __init__(self,
                 weight=None,
                 size_average=False,
                 batch_average=True,
                 cuda=False):
        '''
        Initialize the class.

        @param weight: The class weight vector.
        @param reduction: The reduction option in the KLDivLoss class of
        torch.nn.
        @param cuda: Whether to use gpu or not.
        '''
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='kl'):
        '''
        Return the loss function specified.

        @param mode: 'kl' or 'mi'

        @returns: The loss function.
        '''
        if mode == 'kl':
            return self.KLDivergenceLoss
        elif mode == 'mi':
            return self.MutualInformationLoss
        else:
            raise NotImplementedError

    def KLDivergenceLoss(self, logit, target):
        '''
        Calculate and return the KL-Divergence loss used in knowledge
        distillation.

        @param logit: The output of the student model before softmax.
        @param target: The probabilities predicted by the teacher model.
        '''
        n, c, h, w = logit.size()
        criterion = nn.KLDivLoss(weight=self.weight,
                                 size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    # ------* TODO *------ #
    def CrossEntropyLoss(self, logit, target):
        '''
        Calculate and return the cross entropy loss used in knowledge
        distillation.

        @param logit: The output of the student model before softmax.
        @param target: The probabilities predicted by the teacher model.
        '''
        raise NotImplementedError

    # ------* TODO *------ #
    def MutualInformationLoss(self, logit, target):
        '''
        Calculate and return the Mutual Information Loss used in Variational
        Knowledge Distillation.

        @param logit: The output of the student model before softmax.
        @param target: The probabilities predicted by the teacher model.
        '''
        raise NotImplementedError


class SegmentationLosses(object):
    def __init__(self,
                 weight=None,
                 size_average=True,
                 batch_average=True,
                 ignore_index=255,
                 cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight,
                                        ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight,
                                        ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt)**gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
