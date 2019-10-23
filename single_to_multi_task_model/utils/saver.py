'''
@file: saver.py

This file contains the saver for saving the configuration used for training in
a log file and for creating checkpoint files.

@contributor: Rukmangadh Sai Myana (not the original author)
@mail: rukman.sai@gmail.com
'''

import os
import shutil
import torch
from collections import OrderedDict
import glob


class Saver(object):
    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.dataset, args.checkname)
        self.runs = sorted(
            glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory,
                                           'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'),
                      'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory,
                                        'experiment_{}'.format(str(run_id)),
                                        'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(
                        filename,
                        os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(
                    filename, os.path.join(self.directory,
                                           'model_best.pth.tar'))

    def save_experiment_config(self):
        '''
        Save the configuration of the experiment in the logfile.

        The logfile is 'parameters.txt'. It's created inside the experiment_x
        directory of ./run/dataset/ directory.
        '''
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        log_file.write('# This file contains the configuration for the'
                       ' experiment\n')
        p = OrderedDict()
        p['dataset'] = self.args.dataset
        p['backbone'] = self.args.backbone
        p['output_stride'] = self.args.output_stride
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs
        p['base_img_size'] = self.args.base_img_size
        p['crop_img_size'] = self.args.crop_img_size

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()
