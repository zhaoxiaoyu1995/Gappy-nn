# -*- coding: utf-8 -*-
# @Time    : 2022/4/20 16:08
# @Author  : zhaoxiaoyu
# @File    : airfoil_mlp_gappy.py
import torch
import torch.nn.functional as F
import os
import sys
from torch.utils.data import DataLoader

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(filename)

from model.mlp import MLP
from utils.options import parses
from utils.visualization import plot3x1
from utils.utils import cre

# Configure the arguments
args = parses()
args.exp = 'voronoiunet_cylinder_64'
args.batch_size = 4
print(args)
torch.cuda.set_device(args.gpu_id)


def test(net, test_loader, pod_index, observe_weight=50, n_components=50, test_dataset=None):
    # Load data
    import scipy.io as sio
    from model.gappy_pod import GappyPodWeight
    import numpy as np

    f = sio.loadmat('/mnt/jfs/zhaoxiaoyu/Gappy_POD/airfoil_incompressible.mat')
    data = f['p'][pod_index, :, :, :]
    data = (data - test_dataset.min) / (test_dataset.max - test_dataset.min + 1e-8)
    gappy_pod = GappyPodWeight(
        data=data, map_size=data.shape[-2:], n_components=n_components,
        positions=np.array(
            [[65, 126], [90, 139], [65, 151], [90, 114], [90, 164], [115, 139], [65, 101], [115, 164],
             [65, 176], [115, 114], [90, 89], [90, 189], [115, 89], [115, 189], [40, 164], [65, 76],
             [65, 201], [140, 137], [92, 214], [90, 64], [40, 189], [140, 162], [190, 127], [117, 214],
             [140, 187], [67, 226], [40, 94], [115, 64], [40, 214], [140, 112], [92, 239], [65, 51]]
        ),
        observe_weight=observe_weight
    )

    # Test procedure
    net.eval()
    test_mae, test_rmse, test_num = 0.0, 0.0, 0.0
    test_max_ae = 0
    mean, std = test_dataset.min, test_dataset.max - test_dataset.min
    mean, std = torch.from_numpy(mean).cuda(), torch.from_numpy(std).cuda()
    for i, (inputs, outputs) in enumerate(test_loader):
        N, _ = inputs.shape
        inputs, outputs = inputs.cuda(), outputs.cuda()
        outputs = outputs * std + mean
        with torch.no_grad():
            pre = net(inputs).reshape(-1, 1, 256, 256)
            pre[:, :, test_dataset.airfoil_mask] = 0
            pre = gappy_pod.reconstruct(pre, inputs, weight=torch.ones_like(pre))
            pre = pre * std + mean
            pre[:, :, test_dataset.airfoil_mask] = 0
        test_num += N
        test_mae += F.l1_loss(outputs, pre).item() * N
        test_rmse += torch.sum(cre(outputs, pre, 2))
        test_max_ae += torch.sum(torch.max(torch.abs(outputs - pre).flatten(1), dim=1)[0]).item()
    print('test mae:', test_mae / test_num)
    print('test rmse:', test_rmse / test_num)
    print('test max_ae:', test_max_ae / test_num)

    plot3x1(outputs[-1, 0, :, :].cpu().numpy(), pre[-1, 0, :, :].cpu().numpy(), './test.png')
    return test_mae / test_num


if __name__ == '__main__':
    # Define data loader
    from data.dataset import AirfoilDataset

    # test_dataset = AirfoilDataset(index=[i for i in range(500, 700)], type='p')
    test_dataset = AirfoilDataset(index=[i for i in range(700, 1000)], type='p')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)
    # pod_index = [i for i in range(500)]
    pod_index = [i for i in range(700)]

    # Path of trained network
    args.snapshot = '/mnt/jfs/zhaoxiaoyu/Gappy_POD/airfoil/logs/ckpt/mlp_airfoil_32_p/best_epoch_295_loss_0.00024823.pth'

    # Load trained network
    net = MLP(layers=[32, 128, 1280, 4800, 256 * 256]).cuda()
    net.load_state_dict(torch.load(args.snapshot)['state_dict'])
    print('load models: ' + args.snapshot)

    # observe_weight_c = [50, 100, 200, 500]
    # n_components_c = [10, 15, 20, 25]
    # min_mae, min_observe_weight, min_n_components = 999, 0, 0
    # for n_components in n_components_c:
    #     for observe_weight in observe_weight_c:
    #         mae = test(net, test_loader, pod_index, observe_weight, n_components, test_dataset)
    #         print('n_components: {}, observe_weight: {}, mae: {:.6f}'.format(n_components, observe_weight, mae))
    #         if mae < min_mae:
    #             min_mae, min_observe_weight, min_n_components = mae, observe_weight, n_components
    # print('observe_weight: {}, n_components: {}, mae: {:.6f}'.format(min_observe_weight, min_n_components, min_mae))

    test(net, test_loader, pod_index, 500, 15, test_dataset)
