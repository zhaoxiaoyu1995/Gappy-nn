# -*- coding: utf-8 -*-
# @Time    : 2023/1/9 14:54
# @Author  : zhaoxiaoyu
# @File    : pruning.py
import numpy as np
import torch


def binary_mask_weight(param, cutoff_value):
    return torch.where(
        param.data.abs().le(cutoff_value), torch.zeros_like(param), torch.ones_like(param)
    )
    # temp = torch.rand_like(param)
    # return torch.where(
    #     temp.le(0.1), torch.zeros_like(param), torch.ones_like(param)
    # )


class GlobalRatioPruning:

    def __init__(self, params, ratio):
        self.ratio = ratio
        self.prune_masks = list(self.compute_prune_masks(params))
        self.weight_num = 0.0
        self.diff_params = []

    def compute_cutoff_values(self, params):
        n_weights = 0

        # compute number of items for allocating numpy array
        for param in params:
            if param.requires_grad is False:
                continue
            n_weights += len(param.view(-1))

        # allocate array
        weights = np.empty((n_weights, 1))
        index = 0

        # copy weights into numpy array
        for param in params:
            if param.requires_grad is False:
                continue
            n_param = len(param.view(-1))
            if n_param > 0:
                weights[index: (index + n_param)] = (
                    param.data.view(n_param, -1).cpu().numpy()
                )
                index += n_param

        # compute cutoff value
        global_cutoff_value = np.quantile(np.abs(weights), self.ratio)
        del (weights)

        # create list of global cutoff values
        cutoff_values = list()
        for param in params:
            if param.requires_grad is False:
                cutoff_values.append(None)
                continue
            cutoff_values.append(global_cutoff_value)

        return cutoff_values

    def compute_prune_masks(self, params):
        params = list(params)

        binary_mask = binary_mask_weight
        cutoff_values = self.compute_cutoff_values(params)

        for param, cutoff_value in zip(params, cutoff_values):
            if param.requires_grad is False:
                assert cutoff_value is None
                yield None
            else:
                prune_mask = binary_mask(param, cutoff_value)
                yield prune_mask

    def prune_by_mask(self, params):
        for param, prune_mask in zip(params, self.prune_masks):
            if param.requires_grad is False:
                assert prune_mask is None
                continue

            assert param.data.size() == prune_mask.size()
            param.data.mul_(prune_mask)

    def combine_by_mask(self, params1, params2):
        for param1, param2, prune_mask in zip(params1, params2, self.prune_masks):
            param1.data.mul_(1 - prune_mask).add_(param2.data)

    def save_diff_params(self, params):
        weight_num = 0
        for param, prune_mask in zip(params, self.prune_masks):
            if param.requires_grad is False:
                assert prune_mask is None
                continue

            assert param.data.size() == prune_mask.size()
            weight_num += torch.sum(prune_mask < 0.5)
            params_temp = torch.where(prune_mask > 0.5, param.clone().detach(), torch.zeros_like(param))
            self.diff_params.append(params_temp)
        self.weight_num = weight_num

    def update_mask(self, mask2s):
        lists = []
        for mask1, mask2 in zip(self.prune_masks, mask2s):
            mask1.add_(mask2)
            lists.append(torch.where(mask1.le(0.5), torch.zeros_like(mask1), torch.ones_like(mask1)))
        self.prune_masks = lists


if __name__ == '__main__':
    net = torch.nn.Sequential(
        torch.nn.Linear(10, 200),
        torch.nn.GELU(),
        torch.nn.Linear(200, 200),
        torch.nn.GELU(),
        torch.nn.Linear(200, 1)
    )
    net2 = torch.nn.Sequential(
        torch.nn.Linear(10, 200),
        torch.nn.GELU(),
        torch.nn.Linear(200, 200),
        torch.nn.GELU(),
        torch.nn.Linear(200, 1)
    )
    pruning = GlobalRatioPruning(net.parameters(), 0.8)
    pruning.prune_by_mask(net2.parameters())
    for mask in list(pruning.prune_masks):
        print(mask)
    for params in list(net2.parameters()):
        print(params)
