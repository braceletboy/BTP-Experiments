'''
@file: run.py

The script for running the experiments.

@author: Rukmangadh Sai Myana
@mail: rukman.sai@gmail.com
'''


from __future__ import print_function
import argparse
import torch
import yaml
from main import main
from util import get_summary_dir


# --------------------------------- script ---------------------------------- #
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
                        choices=['mnist', 'fashion_mnist', 'kmnist', 'svhn',
                                 'svhn_extra'],
                        help='The dataset used for the experimentation.')
    parser.add_argument('--download_data', action='store_true', help='Whether\
                        to download the data or not.')
    parser.add_argument('--networks_file', default='BCCNN.yaml', type=str,
                        help='The name of the file where all the network\
                        definitions are present.')
    parser.add_argument('--data_folder', default='./data/', type=str,
                        help='The path to the folder where the data is\
                        present.')
    parser.add_argument('--model_tag', default='BCCNN', type=str,
                        help='Each and every model is given a tag. See the\
                        networks file. This options specifies the model type\
                        to be used for experimentation.')
    parser.add_argument('--model_version', default='0.0', type=str,
                        help='Each and every model is given a version. See\
                        the networks file. This option specifies the version\
                        of the model to be used for experimentation.')
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

    # knowledge transfer flags
    parser.add_argument('--pretrained_model', type=str, help='Path to the\
                        pre-trained model that is used for knowledge\
                        transfer.')

    # parse the arguments
    args = parser.parse_args()

    # obtain the network definition
    with open(args.networks_file, 'r') as nf:
        network_database = yaml.safe_load(nf)
    model_code = args.model_tag + '[' + args.model_version + ']'
    network_definition = network_database[model_code]

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

    # run the main function
    main(args, network_definition)
