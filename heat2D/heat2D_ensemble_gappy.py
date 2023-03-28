# -*- coding: utf-8 -*-
# @Time    : 2022/4/20 16:08
# @Author  : zhaoxiaoyu
# @File    : heat2D_ensemble_gappy.py
import torch
import numpy as np
import torch.nn.functional as F
import h5py
import os
import sys
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(filename)

from model.cnn import UNet
from model.mlp import MLP
from model.gappy_pod import GappyPodWeight
from utils.options import parses
from utils.visualization import plot3x1
from utils.utils import cre

# Configure the arguments
args = parses()
args.batch_size = 16
print(args)
torch.cuda.set_device(args.gpu_id)
cudnn.benchmark = True


def test(net1, net2, test_loader, observe_weight=50, n_components=50):
    # Load data
    pod_index = [i for i in range(5000)]
    f = h5py.File('/mnt/jfs/zhaoxiaoyu/data/heat/temperature.h5', 'r')
    data = f['u'][pod_index, :, :, :]
    data = (data - 298.0) / 50.0
    gappy_pod = GappyPodWeight(
        data=data, map_size=data.shape[-2:], n_components=n_components,
        positions=np.array([[33, 33], [33, 66], [33, 99], [33, 132], [33, 165], [66, 33], [66, 66], [66, 99], [66, 132], [66, 165],
[99, 33], [99, 66], [99, 99], [99, 132], [99, 165], [132, 33], [132, 66], [132, 99], [132, 132], [132, 165],
[165, 33], [165, 66], [165, 99], [165, 132], [165, 165]]),
        observe_weight=observe_weight
    )

    # Test procedure
    net1.eval()
    net2.eval()
    test_mae, test_rmse, test_num = 0.0, 0.0, 0.0
    test_max_ae = 0
    for i, (inputs, outputs, observes) in enumerate(test_loader):
        N, _, _, _ = inputs.shape
        inputs, outputs, observes = inputs.cuda(), outputs.cuda(), observes.cuda()
        with torch.no_grad():
            pre1 = net1(inputs).flatten(1)
            pre2 = net2(observes)

            pre_abs = torch.abs(pre2 - pre1)
            mins, maxs = torch.min(pre_abs.flatten(1), dim=-1, keepdim=True)[0], \
                         torch.max(pre_abs.flatten(1), dim=-1, keepdim=True)[0]
            weight = 1 - (pre_abs - mins) / (maxs - mins)

            # pre = (pre1 + pre2) * 0.5
            pre = pre1
            pre = gappy_pod.reconstruct(pre, observes, weight=weight)
        test_num += N
        test_mae += F.l1_loss(outputs, pre).item() * N
        test_rmse += torch.sum(cre(outputs, pre, 2))
        test_max_ae += torch.sum(torch.max(torch.abs(outputs - pre).flatten(1), dim=1)[0]).item()
    print('test mae:', 50 * test_mae / test_num)
    print('test rmse:', 50 * test_rmse / test_num)
    print('test max_ae:', 50 * test_max_ae / test_num)

    plot3x1(outputs[-1, 0, :, :].cpu().numpy(), pre[-1, 0, :, :].cpu().numpy(), './test.png')
    return 50 * test_mae / test_num


if __name__ == '__main__':
    # train()

    # Define data loader
    from data.dataset import HeatInterpolGappyDataset

    test_dataset = HeatInterpolGappyDataset(index=[i for i in range(5000, 6000)])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)

    # Path of trained network
    args.snapshot1 = '/mnt/jfs/zhaoxiaoyu/Gappy_POD/heat2D/logs/ckpt/voronoicnn_heat_25/best_epoch_297_loss_0.00008896.pth'
    args.snapshot2 = '/mnt/jfs/zhaoxiaoyu/Gappy_POD/heat2D/logs/ckpt/mlp_heat2D_25/best_epoch_287_loss_0.00003694.pth'
    # Load trained network
    net1 = UNet(in_channels=2, out_channels=1).cuda()
    net1.load_state_dict(torch.load(args.snapshot1)['state_dict'])
    print('load models: ' + args.snapshot1)
    net2 = MLP(layers=[25, 128, 1280, 4800, 200 * 200]).cuda()
    net2.load_state_dict(torch.load(args.snapshot2)['state_dict'])
    print('load models: ' + args.snapshot2)

    # observe_weight_c = [20, 50, 100, 200, 300, 500]
    # n_components_c = [20, 25, 30, 40, 50]
    # min_mae, min_observe_weight, min_n_components = 999, 0, 0
    # for n_components in n_components_c:
    #     for observe_weight in observe_weight_c:
    #         mae = test(net1, net2, test_loader, observe_weight, n_components)
    #         print('n_components: {}, observe_weight: {}, mae: {:.6f}'.format(n_components, observe_weight, mae))
    #         if mae < min_mae:
    #             min_mae, min_observe_weight, min_n_components = mae, observe_weight, n_components
    # print('observe_weight: {}, n_components: {}, mae: {:.6f}'.format(min_observe_weight, min_n_components, min_mae))

    test(net1, net2, test_loader, 300, 25)
