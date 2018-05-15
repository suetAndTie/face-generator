'''
data_loader.py

Modified from https://github.com/carpedm20/BEGAN-pytorch/blob/master/data_loader.py
'''


import os
import torch
from torchvision import transforms

from data.folder import ImageFolder

def get_loader(root, split, batch_size, scale_size, num_workers=2, shuffle=True):
    dataset_name = os.path.basename(root)
    image_root = os.path.join(root, 'splits', split)

    if dataset_name in ['CelebA']:
        dataset = ImageFolder(root=image_root, transform=transforms.Compose([
            transforms.CenterCrop(160),
            transforms.Scale(scale_size),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    else:
        dataset = ImageFolder(root=image_root, transform=transforms.Compose([
            transforms.Scale(scale_size),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=int(num_workers))
    data_loader.shape = [int(num) for num in dataset[0][0].size()]

    return data_loader
