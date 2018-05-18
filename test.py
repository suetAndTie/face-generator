'''
test.py

Based on https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/
'''

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import util
import model.began as began
import data.data_loader as data_loader
import torchvision.utils as torch_utils

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/CelebA', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/began_base', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--num', default=1, type=int, help='Number of images to create')


def test(g, d, params, num):
    """Test the model on `num_steps` batches.
    Args:
        g
        d
        params: (Params) hyperparameters
	num: number of images to generate
    """

    # set model to evaluation mode
    g.eval()
    d.eval()

    z_fixed = torch.FloatTensor(num, params.h).normal_(0,1)
    return g(z_fixed)


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = util.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(42)
    if params.cuda: torch.cuda.manual_seed(42)

    # Define the model
    g = began.BeganGenerator(params) if params.cuda else began.BeganGenerator(params)
    d = began.BeganDiscriminator(params) if params.cuda else began.BeganDiscriminator(params)


    # Reload weights from the saved file
    util.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), g, d)

    # test
    f_img = test(g, d, params, args.num)

    for i in range(f_img.shape[0]):
        save_path = os.path.join(args.model_dir, "{}.jpg".format(i))
        torch_utils.save_image(f_img[i], save_path)
