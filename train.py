'''
train.py
'''
import torch


if __name__ == '__main__':

    torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed(42)
