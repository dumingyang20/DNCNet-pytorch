import csv
import os
from glob import glob
import random
import numpy as np
import math
import torch


def split_data(data, mode=None):
    if mode == 'train':  # 60%
        np.random.seed(20)
        np.random.shuffle(data)
        data_info = data[:int(0.6 * len(data))]

    elif mode == 'test':  # 20% = 60%->80%
        np.random.seed(60)
        np.random.shuffle(data)
        data_info = data[int(0.6 * len(data)):int(0.8 * len(data))]

    else:  # 20% = 80%->100%
        data_info = data[int(0.8 * len(data)):]

    return data_info


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_update_freq):
    if not epoch % lr_update_freq and epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


def get_real_imag(x, transpose=False):
    real_part = [x[i].real for i in range(len(x))]
    imag_part = [x[i].imag for i in range(len(x))]
    if transpose is False:
        return np.concatenate((real_part, imag_part)).reshape(2, -1)
    else:
        return np.concatenate((real_part, imag_part)).reshape(-1, 2)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    # outputs have the same size as the input by '.view()'
    size = feat.size()
    assert (len(size) == 3)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat):
    # assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    # style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * content_feat.expand(size) + content_feat.expand(size)


def _calc_feat_flatten_mean_std(feat):
    # takes 2D feat (H, W), return mean and std of array within channels
    assert (feat.size()[0] == 2)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(2, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def upsample(x):
    """
    :param x: (n, C, W, H)
    :return: (n, C/4, W*2, H*2)
    """
    N, Cin, Win = x.size()
    idxL = [0, 1]

    Cout = Cin // 4
    Wout = Win * 2

    up_feature = torch.zeros((N, Cout, Wout)).type(x.type())
    for idx in range(3):
        up_feature[:, :, idxL[idx][0]::2] = x[:, idx:Cin:4, :]

    return up_feature
