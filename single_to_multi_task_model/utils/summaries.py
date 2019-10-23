import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence


class TensorboardSummary(object):
    '''
    Class for the writing to tensorboard.
    '''
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        '''
        Create a SummaryWriter instance for tensorboardX logging.

        @returns: The SummaryWriter instance.
        '''
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target_list, output_list,
                        global_step):
        '''
        Visualize the given images in tensorboard.

        @param writer: The tensorboardX's summary writer instance.
        @param dataset: The name of the dataset we are working on.
        @param image: The input image.
        @param target_list: The list of target images from the teacher models.
        @param output_list: The corresponding segmentation images from the
        student model.
        '''
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        assert len(output_list) == len(target_list)
        num_tasks = len(output_list)
        for idx in range(num_tasks):
            output = output_list[idx]
            target = target_list[idx]
            grid_image = make_grid(decode_seg_map_sequence(torch.max(
                output[:3], 1)[1].detach().cpu().numpy(),
                                                           dataset=dataset),
                                   3,
                                   normalize=False,
                                   range=(0, 255))
            writer.add_image('Predicted label', grid_image, global_step)
            grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(
                target[:3], 1).detach().cpu().numpy(),
                                                           dataset=dataset),
                                   3,
                                   normalize=False,
                                   range=(0, 255))
            writer.add_image('Groundtruth label', grid_image, global_step)
