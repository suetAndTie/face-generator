'''
inception.py

Based on https://github.com/sbarratt/inception-score-pytorch
'''


import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as torch_utils
import numpy as np
from scipy.stats import entropy

def inception_score(imgs, params, batch_size=32, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    params -- params for testing
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    # Normalize images
    for i in range(N):
        imgs[i] = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(imgs[i])

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=32)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(params.device)
    inception_model.eval();

    # Get predictions
    preds = np.zeros((N, 1000))

    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(params.device)
    
    for i, batch in enumerate(dataloader, 0):
        if params.cuda: batch = batch.cuda(async=True)
        batch_size_i = batch.size()[0]
        x = up(batch)
        preds[i*batch_size : i*batch_size + batch_size_i] = F.softmax(inception_model(x), dim=1).data.cpu().numpy()

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)
