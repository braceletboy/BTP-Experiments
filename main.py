'''
@file: main.py

The main file containing the skeletal code for training the models.

@author: Rukmangadh Sai Myana
@mail: rukman.sai@gmail.com
'''


import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from model import BasicClassificationCNN, ModularClassificationCNN
from util import (
    set_defaults,
    save_best_model,
)
from sampler import MyMnistSampler


_CHKPNT_IDX = 0  # points to the checkpoint to be created/overwritten
_CUMM_BATCH_IDX = -1  # cummulative batch index for loss graph

_TAG_MAP = {
    'BCCNN': BasicClassificationCNN,
    'MCCNN': ModularClassificationCNN,
}


def train_mnist(args,
                mnist_model,
                train_loader,
                optimizer,
                loss_layer,
                writer,
                epoch):
    '''
    Train the model for the classification task on mnist data.

    @param args: The arguments provided as flags.
    @param mnist_model: The pytorch model we are using.
    @param train_loader: The data loader for training.
    @param optimizer: The optimizer used for training.
    @param loss_layer: The loss layer for loss calculation.
    @param epoch: The value of the current epoch.

    @returns metrics: The metrics of the model for the current epoch
    '''
    mnist_model.train()  # put the model in training mode.

    # iterate over the train dataset
    num_samples = 0
    num_correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        global _CUMM_BATCH_IDX
        _CUMM_BATCH_IDX += 1
        if args.task_number == 2:
            target = target - 5  # range of target values should be (0,4)
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()  # clear the gradients

        # forward propagation
        output = mnist_model(data)
        loss = loss_layer(output, target)

        # backward propagation - gradient calculation
        loss.backward()

        # update the parameters
        optimizer.step()

        # logging
        writer.add_scalar('loss/train', loss.item(), _CUMM_BATCH_IDX)

        num_samples += list(target.size())[0]
        num_correct += int(torch.sum(torch.argmax(output, dim=1) == target
                                     ).item())
        if batch_idx % args.checkpoint_interval == 0:
            # save checkpoint
            global _CHKPNT_IDX
            filepath = os.path.join(args.summary_dir,
                                    'checkpoint_{}.tar'.format(_CHKPNT_IDX))
            _CHKPNT_IDX = (_CHKPNT_IDX + 1) % args.num_checkpoints  # update
            torch.save({
                'model_state_dict': mnist_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'checkpoint_idx': _CHKPNT_IDX,
            }, filepath)

            # save checkpoint as latest_checkpoint also
            filepath = os.path.join(args.summary_dir, 'latest_checkpoint.tar')
            torch.save({
                'model_state_dict': mnist_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'checkpoint_idx': _CHKPNT_IDX,
            }, filepath)

    metrics = {'accuracy': num_correct/num_samples*100}
    writer.add_scalar('accuracy/train', metrics['accuracy'],
                      epoch)
    tqdm.write('Train accuracy after epoch-{} is {:.2f}% \n'.format(
        epoch, metrics['accuracy']))
    return metrics


def test_mnist(args,
               mnist_model,
               test_loader,
               loss_layer,
               writer,
               epoch):
    '''
    Test the model for the classification task on mnist data.

    @param args: The argumennts provided as flags.
    @param mnist_model: The pytorch model we are using.
    @param test_loader: The data loader for testing.
    @param loss_layer: The loss layer for loss calculation.
    @param epoch: The value of the current epoch.

    @returns The metrics of the model for the current epoch.
    '''
    mnist_model.eval()  # put the model in evaluation mode.

    # iterate over the test dataset
    num_samples = 0
    num_correct = 0
    for data, target in test_loader:
        if args.task_number == 2:
            target = target - 5  # range of target values should be (0,4)
        data, target = data.to(args.device), target.to(args.device)
        output = mnist_model(data)
        num_samples += list(target.size())[0]
        num_correct += int(torch.sum(torch.argmax(output, dim=1) == target
                                     ).item())

    metrics = {'accuracy': num_correct/num_samples*100}
    writer.add_scalar('accuracy/test', metrics['accuracy'],
                      epoch)
    tqdm.write('Test accuracy after epoch-{} is {:.2f}% \n'.format(
        epoch, metrics['accuracy']))

    # save model if best
    save_best_model(args.summary_dir, mnist_model, metrics, 'accuracy')
    return metrics


def load_dataset(args):
    '''
    Load the dataset specified by the argument.

    @param args: The arguments provided though the script flags.
    @returns: The data loaders.
    '''
    if args.dataset == 'mnist':
        # create train dataset
        train_dataset = datasets.MNIST(
            './data/', train=True, download=args.download_data,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(28),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ])
        )

        # create the sampler for training
        if args.sampler == 'mysampler':
            train_sampler = MyMnistSampler(args.labels, train_dataset)
        else:
            raise NotImplementedError('Given learning rate scheduler is\
                                          not yet supported.')

        # create train dataloader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=args.pin_memory,
        )

        # create test dataset
        test_dataset = datasets.MNIST('./data/', train=False,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              (0.1307,), (0.3081,))
                                      ])
                                      )

        # create the sampler for testing
        if args.sampler == 'mysampler':
            test_sampler = MyMnistSampler(args.labels, test_dataset)
        else:
            raise NotImplementedError('Given learning rate scheduler is\
                                          not yet supported.')

        # create test dataloader
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.train_batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=args.workers,
            pin_memory=args.pin_memory,
        )
        return {'train': train_loader, 'test': test_loader}


def main(args, network_definition):
    '''
    The main function that gets executed when the script is run.
    '''
    writer = SummaryWriter(log_dir=args.summary_dir)

    if args.task == 'classification':
        if args.dataset == 'mnist':
            mnist_model = _TAG_MAP[args.model_tag](network_definition).to(
                args.device)
            data_loaders = load_dataset(args)
            if args.task_number == 2:
                # load pre-trained network
                pretrained_checkpoint = torch.load(args.pretrained_model)
                mnist_model.load_state_dict(
                    pretrained_checkpoint['model_state_dict'], strict=False)

                # freeze loaded network
                for param in mnist_model.conv_network.parameters():
                    param.requires_grad = False
                for param in mnist_model.fc_network.parameters():
                    param.requires_grad = False

            # set optimizer
            if args.optimizer == 'adam':
                optimizer_kwargs = {'lr': args.lr,
                                    'betas': args.betas,
                                    'eps': args.eps,
                                    'weight_decay': args.weight_decay,
                                    'amsgrad': args.amsgrad}
                default_kwargs = {'lr': 0.001,
                                  'betas': (0.9, 0.999),
                                  'eps': 1e-8,
                                  'weight_decay': 0,
                                  'amsgrad': False}

                # set defaults to values not provided
                optimizer_kwargs = set_defaults(optimizer_kwargs,
                                                default_kwargs)
                optimizer = torch.optim.Adam(mnist_model.parameters(),
                                             **optimizer_kwargs)
            else:
                raise NotImplementedError('Given optimizer is not yet\
                                          supported.')

            # set lr scheduler
            if args.lr_scheduler == 'steplr':
                if args.step_size is None:
                    raise Exception('--step_size is required.')
                scheduler_kwargs = {'step_size': args.step_size,
                                    'gamma': args.gamma,
                                    'last_epoch': args.last_epoch}
                default_kwargs = {'gamma': 0.1, 'last_epoch': -1}

                # set defaults to the values not provided.
                scheduler_kwargs = set_defaults(scheduler_kwargs,
                                                default_kwargs)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                            **scheduler_kwargs)
            else:
                raise NotImplementedError('Given learning rate scheduler is\
                                          not yet supported.')

            # set loss layer
            if args.loss == 'crossentropyloss':
                loss_layer = nn.CrossEntropyLoss(reduction=args.reduction)
            else:
                raise NotImplementedError('Given loss layer is not yet\
                                          supported.')

            # resume training
            global _CHKPNT_IDX
            if args.resume:
                path = os.path.join(args.summary_dir, 'latest_checkpoint.tar')
                latest_checkpoint = torch.load(path, args.device)
                mnist_model.load_state_dict(
                    latest_checkpoint['model_state_dict'])
                optimizer.load_state_dict(
                    latest_checkpoint['optimizer_state_dict'])
                _CHKPNT_IDX = latest_checkpoint['checkpoint_idx']

            # load checkpoint
            elif args.load_checkpoint is not None:
                checkpoint = torch.load(args.load_checkpoint, args.device)
                mnist_model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                _CHKPNT_IDX = checkpoint['checkpoint_idx']

            # model learning
            for epoch in tqdm(range(1, args.num_epochs+1),
                              desc='Epoch Number'):
                train_metrics = train_mnist(args,
                                            mnist_model,
                                            data_loaders['train'],
                                            optimizer,
                                            loss_layer,
                                            writer,
                                            epoch)
                test_metrics = test_mnist(args,
                                          mnist_model,
                                          data_loaders['test'],
                                          loss_layer,
                                          writer,
                                          epoch)
                scheduler.step()

    # close the writer
    writer.close()
