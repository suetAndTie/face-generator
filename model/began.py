'''
began.py

Based on https://github.com/pytorch/examples/blob/master/dcgan/main.py
BEGAN Model https://arxiv.org/pdf/1703.10717.pdf
'''


import torch
import torch.nn as nn
import model.model_util as util

class BeganGenerator(nn.Module):
    def __init__(self, options_dict):
        super(BeganGenerator, self).__init__()
        self.opt = options_dict
        self.main = nn.Sequential(
            # Dim: batch_size x h
            nn.Linear(self.opt['h'], self.opt['n'] * 8 * 8),
            # Dim: batch_size x (n * 8 * 8)
            util.View(-1, self.opt['n'], 8, 8),
            # Dim: batch_size x n x 8 x 8
            nn.Conv2d(self.opt['n'], self.opt['n'], kernel_size=3, stride=1, padding=1),
            nn.ELU(self.opt['alpha'], inplace=True),
            # Dim: batch_size x n x 8 x 8
            nn.Conv2d(self.opt['n'], self.opt['n'], kernel_size=3, stride=1, padding=1),
            nn.ELU(self.opt['alpha'], inplace=True),
            # Dim: batch_size x n x 8 x 8
            nn.Upsample(scale_factor=2),
            # Dim: batch_size x n x 16 x 16
            nn.Conv2d(self.opt['n'], self.opt['n'], kernel_size=3, stride=1, padding=1),
            nn.ELU(self.opt['alpha'], inplace=True),
            # Dim: batch_size x n x 16 x 16
            nn.Conv2d(self.opt['n'], self.opt['n'], kernel_size=3, stride=1, padding=1),
            nn.ELU(self.opt['alpha'], inplace=True),
            # Dim: batch_size x n x 16 x 16
            nn.Upsample(scale_factor=2),
            # Dim: batch_size x n x 32 x 32
            nn.Conv2d(self.opt['n'], self.opt['n'], kernel_size=3, stride=1, padding=1),
            nn.ELU(self.opt['alpha'], inplace=True),
            # Dim: batch_size x n x 32 x 32
            nn.Conv2d(self.opt['n'], self.opt['n'], kernel_size=3, stride=1, padding=1),
            nn.ELU(self.opt['alpha'], inplace=True),
            # Dim: batch_size x n x 32 x 32
            nn.Upsample(scale_factor=2),
            # Dim: batch_size x n x 64 x 64
            nn.Conv2d(self.opt['n'], self.opt['n'], kernel_size=3, stride=1, padding=1),
            nn.ELU(self.opt['alpha'], inplace=True),
            # Dim: batch_size x n x 64 x 64
            nn.Conv2d(self.opt['n'], self.opt['n'], kernel_size=3, stride=1, padding=1),
            nn.ELU(self.opt['alpha'], inplace=True),
            # Dim: batch_size x n x 64 x 64
            nn.Upsample(scale_factor=2),
            # Dim: batch_size x n x 128 x 128
            nn.Conv2d(self.opt['n'], self.opt['n'], kernel_size=3, stride=1, padding=1),
            nn.ELU(self.opt['alpha'], inplace=True),
            # Dim: batch_size x n x 128 x 128
            nn.Conv2d(self.opt['n'], self.opt['n'], kernel_size=3, stride=1, padding=1),
            nn.ELU(self.opt['alpha'], inplace=True),
            # Dim: batch_size x n x 128 x 128
            nn.Conv2d(self.opt['n'], 3, kernel_size=3, stride=1, padding=1),
            # Dim: batch_size x 3 (RGB Channels) x 128 x 128
        )

    def forward(self, input):
        if input.is_cuda and self.opt['ngpu'] > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.opt['ngpu']))
        else:
            output = self.main(input)
        return output

class BeganDiscriminator(nn.Module):
    def __init__(self, options_dict):
        super(BeganDiscriminator, self).__init__()
        self.opt = options_dict
        self.main = nn.Sequential(
            # Dim: batch_size x 3 (RGB Channels) x 128 x 128
            nn.Conv2d(3, self.opt['n'], kernel_size=3, stride=1, padding=1),
            nn.ELU(self.opt['alpha'], inplace=True),
            # Dim: batch_size x n x 128 x 128
            nn.Conv2d(self.opt['n'], self.opt['n'], kernel_size=3, stride=1, padding=1),
            nn.ELU(self.opt['alpha'], inplace=True),
            # Dim: batch_size x n x 128 x 128
            nn.Conv2d(self.opt['n'], self.opt['n'], kernel_size=3, stride=1, padding=1),
            nn.ELU(self.opt['alpha'], inplace=True),
            # Dim: batch_size x n x 128 x 128
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Dim: batch_size x n x 64 x 64
            nn.Conv2d(self.opt['n'], self.opt['n'], kernel_size=3, stride=1, padding=1),
            nn.ELU(self.opt['alpha'], inplace=True),
            # Dim: batch_size x n x 64 x 64
            nn.Conv2d(self.opt['n'], self.opt['n'], kernel_size=3, stride=1, padding=1),
            nn.ELU(self.opt['alpha'], inplace=True),
            # Dim: batch_size x n x 64 x 64
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Dim: batch_size x n x 32 x 32
            nn.Conv2d(self.opt['n'], self.opt['n'], kernel_size=3, stride=1, padding=1),
            nn.ELU(self.opt['alpha'], inplace=True),
            # Dim: batch_size x n x 32 x 32
            nn.Conv2d(self.opt['n'], self.opt['n'], kernel_size=3, stride=1, padding=1),
            nn.ELU(self.opt['alpha'], inplace=True),
            # Dim: batch_size x n x 32 x 32
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Dim: batch_size x n x 16 x 16
            nn.Conv2d(self.opt['n'], self.opt['n'], kernel_size=3, stride=1, padding=1),
            nn.ELU(self.opt['alpha'], inplace=True),
            # Dim: batch_size x n x 16 x 16
            nn.Conv2d(self.opt['n'], self.opt['n'], kernel_size=3, stride=1, padding=1),
            nn.ELU(self.opt['alpha'], inplace=True),
            # Dim: batch_size x n x 16 x 16
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Dim: batch_size x n x 8 x 8
            nn.Conv2d(self.opt['n'], self.opt['n'], kernel_size=3, stride=1, padding=1),
            nn.ELU(self.opt['alpha'], inplace=True),
            # Dim: batch_size x n x 8 x 8
            nn.Conv2d(self.opt['n'], self.opt['n'], kernel_size=3, stride=1, padding=1),
            nn.ELU(self.opt['alpha'], inplace=True),
            # Dim: batch_size x n x 8 x 8
            util.View(-1, self.opt['n'] * 8 * 8),
            # Dim: batch_size x (n * 8 * 8)
            nn.Linear(self.opt['n'] * 8 * 8, self.opt['h'])
            # Dim: batch_size x h
        )
        
    def forward(self, input):
        if input.is_cuda and self.opt['ngpu'] > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.opt['ngpu']))
        else:
            output = self.main(input)
        return output
