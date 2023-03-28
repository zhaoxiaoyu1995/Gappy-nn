# -*- coding: utf-8 -*-
# @Time    : 2022/4/20 16:08
# @Author  : zhaoxiaoyu
# @File    : airfoil_gappy_pod.py
import torch
import torch.nn.functional as F
import os
import sys
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import scipy.io as sio

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(filename)

from model.gappy_pod import GappyPod
from data.dataset import AirfoilPodDataset
from utils.utils import cre
from utils.visualization import plot3x1

# Configure the arguments
batch_size = 16
torch.cuda.set_device(0)


def test(index, n_components=20, positions=np.array([[50, 25], [62, 30]])):
    # Define data loader
    pod_index = [i for i in range(index[0])]
    test_dataset = AirfoilPodDataset(pod_index=pod_index, index=index, positions=positions, type='vy')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    # Load data
    f = sio.loadmat('/mnt/jfs/zhaoxiaoyu/Gappy_POD/airfoil_incompressible.mat')
    data = f['vy'][:][pod_index, :, :, :]
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
    # sio.savemat('gappy_vx.mat', {
    #     'true': outputs[-1, 0, :, :].cpu().numpy(),
    #     'pre': pre[-1, 0, :, :].cpu().numpy()
    # })
    return test_mae / test_num


if __name__ == '__main__':
    positions = [[65, 126], [90, 139], [65, 151], [90, 114], [90, 164], [115, 139], [65, 101], [115, 164]]
    positions = positions[:8]

    # select the optimal number of modes
    index = [i for i in range(500, 700)]
    results = []
    for i in range(0, 10):
        results.append([i, test(index, n_components=i, positions=np.array(positions))])
    results = np.array(results)
    print(results[np.argmin(results[:, 1]), :])

    print(int(results[np.argmin(results[:, 1]), :][0]))
    test(index=[i for i in range(700, 1000)], n_components=int(results[np.argmin(results[:, 1]), :][0]),
         positions=np.array(positions))
