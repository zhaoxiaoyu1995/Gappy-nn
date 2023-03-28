# -*- coding: utf-8 -*-
# @Time    : 2022/4/20 16:08
# @Author  : zhaoxiaoyu
# @File    : heat2D_gappy_pod.py
import torch
import torch.nn.functional as F
import os
import sys
from torch.utils.data import DataLoader
import numpy as np
import h5py
from tqdm import tqdm

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(filename)

from model.gappy_pod import GappyPod
from data.dataset import HeatPodDataset
from utils.visualization import plot3x1
from utils.utils import cre

# Configure the arguments
batch_size = 16
torch.cuda.set_device(0)


def test(index, n_components=20, positions=np.array([[50, 25], [62, 30]])):
    # Define data loader
    pod_index = [i for i in range(4000)]
    test_dataset = HeatPodDataset(pod_index=pod_index, index=index, std=1.0, n_components=20)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    # Load data
    f = h5py.File('/mnt/jfs/zhaoxiaoyu/data/heat/temperature.h5', 'r')
    data = f['u'][pod_index, :, :, :] - 298.0
    gappy_pod = GappyPod(
        data=data, map_size=data.shape[-2:], n_components=n_components,
        positions=positions
    )

    test_mae, test_rmse, test_num = 0.0, 0.0, 0.0
    test_max_ae = 0
    for i, (inputs, _, outputs) in enumerate(tqdm(test_loader)):
        N, _ = inputs.shape
        inputs, _, outputs = inputs.cuda(), _, outputs.cuda()

        pre = gappy_pod.reconstruct(inputs)
        test_num += N
        test_mae += F.l1_loss(outputs, pre).item() * N
        test_rmse += torch.sum(cre(outputs, pre, 2)).item()
        test_max_ae += torch.sum(torch.max(torch.abs(outputs - pre).flatten(1), dim=1)[0]).item()
    print('test mae:', test_mae / test_num)
    print('test rmse:', test_rmse / test_num)
    print('test max_ae:', test_max_ae / test_num)

    # plot3x1(outputs[-1, 0, :, :].cpu().numpy(), pre[-1, 0, :, :].cpu().numpy(), './test.png')
    import scipy.io as sio
    sio.savemat('gappy.mat', {
        'true': outputs[-1, 0, :, :].cpu().numpy(),
        'pre': pre[-1, 0, :, :].cpu().numpy()
    })
    return test_mae / test_num


if __name__ == '__main__':
    positions = [[40, 40], [40, 80], [40, 120], [40, 160], [80, 40], [80, 80], [80, 120], [80, 160], [120, 40],
                 [120, 80],
                 [120, 120], [120, 160], [160, 40], [160, 80], [160, 120], [160, 160]]

    # select the optimal number of modes
    # index = [i for i in range(4000, 5000)]
    # results = []
    # for i in range(10, len(positions) + 1):
    #     results.append([i, test(index, n_components=i, positions=np.array(positions))])
    # results = np.array(results)
    # print(results[np.argmin(results[:, 1]), :])

    # Evaluate the Gappy POD method
    test(index=[i for i in range(5999, 6000)], n_components=14, positions=np.array(positions))
