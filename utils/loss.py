# -*- coding: utf-8 -*-
# @Time    : 2022/4/27 15:32
# @Author  : zhaoxiaoyu
# @File    : loss.py
import torch


class OHEM2d(torch.nn.Module):
    """
    Weighted Loss
    """
    def __init__(self, loss_fun, weight=None):
        super(OHEM2d, self).__init__()
        self.weight = weight
        self.loss_fun = loss_fun

    def forward(self, inputs, targets):
        diff = self.loss_fun(inputs, targets, reduction='none').detach()
        min, max = torch.min(diff.view(diff.shape[0], -1), dim=1)[0], torch.max(diff.view(diff.shape[0], -1), dim=1)[0]
        if inputs.ndim == 4:
            min, max = min.reshape(diff.shape[0], 1, 1, 1).expand(diff.shape), \
                       max.reshape(diff.shape[0], 1, 1, 1).expand(diff.shape)
        elif inputs.ndim == 3:
            min, max = min.reshape(diff.shape[0], 1, 1).expand(diff.shape), \
                       max.reshape(diff.shape[0], 1, 1).expand(diff.shape)
        diff = 10.0 * (diff - min) / (max - min)
        return torch.mean(torch.abs(diff * (inputs - targets)))
