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
import pickle
from tqdm import tqdm

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(filename)

from model.gappy_pod import GappyPod
from data.dataset import CylinderPodDataset
from utils.visualization import plot3x1
from utils.utils import cre

# Configure the arguments
batch_size = 16
torch.cuda.set_device(0)


def test(index, n_components=20, positions=np.array([[50, 25], [62, 30]])):
    import time
    # Define data loader
    pod_index = [i for i in range(3500)]
    test_dataset = CylinderPodDataset(pod_index=pod_index, index=index)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    # Load data
    df = open('/home/ubuntu/zhaoxiaoyu/data/cylinder/Cy_Taira.pickle', 'rb')
    data = np.transpose(pickle.load(df), (0, 3, 1, 2))[pod_index, :, :, :]
    df.close()
    gappy_pod = GappyPod(
        data=data, map_size=data.shape[-2:], n_components=n_components,
        positions=positions
    )

    test_mae, test_rmse, test_num = 0.0, 0.0, 0.0
    test_max_ae = 0
    start = time.time()
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
    print(time.time() - start)

    plot3x1(outputs[-1, 0, :, :].cpu().numpy(), pre[-1, 0, :, :].cpu().numpy(), './test.png')

    import scipy.io as sio
    sio.savemat('gappy.mat', {
        'true': outputs[-1, 0, :, :].cpu().numpy(),
        'pre': pre[-1, 0, :, :].cpu().numpy()
    })

    return test_mae / test_num


if __name__ == '__main__':
    positions = [[50, 25], [62, 30], [50, 35], [62, 40], [50, 45], [62, 50], [50, 55], [62, 60]]
    test(index=[i for i in range(4900, 4901)], n_components=6, positions=np.array(positions))
    # select the optimal number of modes
    # index = [i for i in range(3500, 4250)]
    # results = []
    # for i in range(1, len(positions) + 1):
    #     results.append([i, test(index, n_components=i, positions=np.array(positions))])
    # results = np.array(results)
    # print(results[np.argmin(results[:, 1]), :])
