'''
@file: run.py

The main script for running the experiments.

@author: Rukmangadh Sai Myana
@mail: rukman.sai@gmail.com
'''


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from model import BasicClassificationCNN
from util import (
    set_defaults,
    get_summary_dir,
    save_best_model,
)
from sampler import MyMnistSampler

_CHKPNT_IDX = 0  # points to the checkpoint to be created/overwritten
_CUMM_BATCH_IDX = -1


def train_mnist(args, mnist_model, train_loader, optimizer, loss_layer,
                writer, epoch):
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
            filepath = os.path.join(args.summary_dir, 'checkpoint_\
                {}.tar'.format(_CHKPNT_IDX))
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


def test_mnist(args, mnist_model, test_loader, loss_layer, writer, epoch):
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


# ------------------------------ main script -------------------------------- #
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main script for\
        experimentation.')
    # general flags
    parser.add_argument('--task', default='classification', type=str,
                        choices=['classfication', 'segmentation'], help='The\
                        task we are performing the experiments for.')
    parser.add_argument('--task_number', default=1, type=int, help='The\
                        serial number of the task being performing during the\
                        sequential learning')
    parser.add_argument('--dataset', default='mnist', type=str,
                        choices=['mnist'], help='The dataset used for the\
                        experimentation.')
    parser.add_argument('--download_data', action='store_true', help='Whether\
                        to download the data or not.')
    parser.add_argument('--networks_file', default='networks.yaml', type=str,
                        help='The name of the file where all the network\
                        definitions are present.')
    parser.add_argument('--data_folder', default='./data/', type=str,
                        help='The path to the folder where the data is\
                        present.')
    parser.add_argument('--model', default='BCCNN1', type=str,
                        help='Each and every model is given a code. See the\
                        networks file. This options specifies the model to be\
                        used for experimentation through its model code.')
    parser.add_argument('--use_cuda', action='store_true', help='Whether to\
                        use gpu for training the model or not.')
    parser.add_argument('--pin_memory', action='store_true', help='Whether to\
                        fetch data from pinned memory.')
    parser.add_argument('--num_epochs', default=30, type=int, help='The\
                        number of epochs in the training')
    parser.add_argument('--train_batch_size', default=64, type=int,
                        help='The batch size used for training.')
    parser.add_argument('--test_batch_size', default=64, type=int,
                        help='The batch size used for testing.')
    parser.add_argument('--workers', default=2, type=int, help='The number of\
                        workers to be used for data loading.')
    parser.add_argument('--logdir', default='./logs', type=str, help='The\
                        directory where the logs and logging related stuff are\
                        to be stored')

    # loss layer flags
    parser.add_argument('--loss', default='crossentropyloss', type=str,
                        choices=['crossentropyloss'], help='The loss function\
                        to be used for optimization.')
    parser.add_argument('--reduction', default='mean', type=str,
                        choices=['none', 'mean', 'sum'], help='The reduction\
                        option for the loss layer. See pytorch docs.')

    # sampler flags
    parser.add_argument('--sampler', default='mysampler', type=str,
                        choices=['mysampler'], help='The sampler to be used\
                        in the data loader.')
    parser.add_argument('--labels', nargs='+', default=[0, 1, 2, 3, 4],
                        type=int, help='The labels thar mysampler should use\
                        for training.')

    # optimizer flags
    parser.add_argument('--optimizer', default='adam', type=str,
                        choices=['adam'], help='The optimizer to use while\
                        training.')
    parser.add_argument('--lr', type=float, help='Learning rate for the\
                        optimizer.')
    parser.add_argument('--betas', nargs='+', type=float, help='The *betas*\
                        option for the optimizer. See pytorch docs.')
    parser.add_argument('--eps', type=float, help='The *eps* option for the\
                        optimizer. See pytorch docs.')
    parser.add_argument('--weight_decay', type=float, help='The weight decay\
                        for the learning rate.')
    parser.add_argument('--amsgrad', type=bool, help='The amsgrad option for\
                        the optimizer.')
    parser.add_argument('--momentum', type=float, help='Momentum option for \
                        the optimizer.')

    # learning rate scheduler flags
    parser.add_argument('--lr_scheduler', default='steplr', type=str,
                        choices=['steplr'], help='The learning rate scheduler\
                        for the optimizer.')
    parser.add_argument('--step_size', type=int, help='The step size option\
                        for the learning rate scheduler. See pytorch docs.')
    parser.add_argument('--gamma', type=float, help='The gamma option for the\
                        learing rate scheduler. See pytorch docs.')
    parser.add_argument('--last_epoch', type=int, help='The last_epoch option\
                        for the learning rate scheduler. See pytorch docs.')

    # saving and loading flags
    parser.add_argument('--num_checkpoints', default=10, type=int,
                        help='Number of checkpoints to maintain for the\
                        learning process.')
    parser.add_argument('--resume', action='store_true', help='Whether to\
                        resume the learning from the most recent checkpoint.')
    parser.add_argument('--checkpoint_interval', default=100, type=int,
                        help='Number of steps between saving checkpoints.')
    parser.add_argument('--load_checkpoint', type=str, help='Path to the\
                        checkpoint file which is to be loaded to warm start\
                        the training. This option cannot be used if the\
                        --resume option is used as the resume option by\
                        default loads the latest_checkpoint.tar file.')

    # parse the arguments
    args = parser.parse_args()

    # obtain the network definition
    with open(args.networks_file, 'r') as nf:
        network_database = yaml.safe_load(nf)
    network_definition = network_database[args.model]

    # device for experimenting on.
    args.device = torch.device('cuda' if args.use_cuda else 'cpu')

    # multi-data loading not preferable with CUDA
    args.workers = 1 if args.use_cuda else args.workers

    # faster loading in GPU with pinned memories
    args.pin_memory = args.use_cuda or args.pin_memory

    # make sure the option is in lower case
    args.optimizer = (args.optimizer).lower()
    args.lr_scheduler = (args.lr_scheduler).lower()
    args.loss = (args.loss).lower()

    # get the directory for storing the summary logs
    args.summary_dir = get_summary_dir(args)

    writer = SummaryWriter(log_dir=args.summary_dir)

    if args.task == 'classification':
        if args.dataset == 'mnist':
            mnist_model = BasicClassificationCNN(network_definition).to(
                args.device)
            data_loaders = load_dataset(args)

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
