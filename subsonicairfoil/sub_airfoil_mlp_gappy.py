# -*- coding: utf-8 -*-
# @Time    : 2022/4/20 16:08
# @Author  : zhaoxiaoyu
# @File    : sub_airfoil_mlp_gappy.py
import torch
import torch.nn.functional as F
import logging
import os
import sys
from torch.utils.data import DataLoader

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(filename)

from model.mlp import MLP
from data.dataset import SubsonicAirfoilDataset
from utils.options import parses
from utils.visualization import plot3x1_coor
from utils.utils import cre

# Configure the arguments
args = parses()
args.exp = 'mlp_sub_airfoil_4_p'
args.epochs = 300
args.batch_size = 2
args.plot_freq = 10
print(args)
torch.cuda.set_device(args.gpu_id)


def test(net, test_loader, observe_weight=50, n_components=50, test_dataset=None, positions=[2, 3]):
    # Load data
    import h5py
    from model.gappy_pod import GappyPodWeight1D
    import numpy as np

    f = h5py.File('/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_train.h5', 'r')
    data = f['vy'][:]
    data = (data - test_dataset.min) / (test_dataset.max - test_dataset.min + 1e-10)
    f.close()
    gappy_pod = GappyPodWeight1D(
        data=data, map_size=16339, n_components=n_components,
        positions=np.array(positions),
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
            pre = net(inputs)
            # pre = gappy_pod.reconstruct(pre, inputs, weight=torch.ones_like(pre))
            pre = pre * std + mean
        test_num += N
        test_mae += F.l1_loss(outputs, pre).item() * N
        test_rmse += torch.sum(cre(outputs, pre, 2))
        test_max_ae += torch.sum(torch.max(torch.abs(outputs - pre).flatten(1), dim=1)[0]).item()
    print('test mae:', test_mae / test_num)
    print('test rmse:', test_rmse / test_num)
    print('test max_ae:', test_max_ae / test_num)

    plot3x1_coor(outputs[-1, :].cpu().numpy(), pre[-1, :].cpu().numpy(),
                 file_name='test.png', x_coor=test_dataset.x_coor,
                 y_coor=test_dataset.y_coor)
    return 0.001 * test_mae / test_num


if __name__ == '__main__':
    # Define data loader
    positions = [16097, 16016, 16087, 15986, 15987, 15978, 15981, 15980, 16018, 16046, 16061, 16069, 16075, 15657,
                 15641, 15633, 15665, 15605, 15598, 15603, 15604, 15654, 15684, 15699, 15707, 15713, 15141, 15124,
                 15135, 15113, 15150, 15157, 15149, 15152, 15146, 15179, 15194, 15202, 15208, 14569, 14590, 14574,
                 14489, 14468, 14482, 14471, 14466, 14470, 14510, 14525, 14533, 14539, 13473, 13584, 13554, 13557,
                 13611, 13580, 13703, 13612, 13603, 13645, 13660, 13668, 13674, 11998, 12094, 11981, 12144, 11978,
                 12106, 11997, 12097, 11996, 12017, 12032, 12040, 12046, 8636, 8506, 8634, 8674, 9277, 9971, 9821,
                 9273, 8676, 8558, 8573, 8581, 8587, 4792, 4638, 4734, 4743, 4719, 4715, 4714, 4720, 4620, 4672,
                 4687, 4695, 4701, 2949, 3067, 2891, 2944, 2947, 3034, 3061, 3018, 3039, 2981, 2996, 3004, 3010,
                 2056, 2012, 1920, 2004, 2013, 1998, 2008, 2006, 2005, 1953, 1968, 1976, 1982, 1356, 1245, 1350,
                 1362, 1328, 1322, 1326, 1321, 1324, 1274, 1289, 1297, 1303, 830, 838, 842, 815, 803, 741, 798, 735,
                 808, 761, 776, 784, 790, 374, 459, 437, 364, 362, 360, 363, 365, 439, 395, 410, 418, 424]
    positions = positions[:169]

    test_dataset = SubsonicAirfoilDataset(type='vy', data_path='/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_test.h5',
                                          positions=positions)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    # Path of trained network
    args.snapshot = '/mnt/jfs/zhaoxiaoyu/Gappy_POD/subsonicairfoil/logs/ckpt/mlp_sub_airfoil_169_vy/best_epoch_289_loss_0.00191551.pth'

    # Load trained network
    net = MLP(layers=[169, 128, 1280, 4800, 16339]).cuda()
    net.load_state_dict(torch.load(args.snapshot)['state_dict'])
    print('load models: ' + args.snapshot)

    # observe_weight_c = [10, 20, 30, 50, 100]
    # n_components_c = [50, 80, 100, 300, 400]
    # min_mae, min_observe_weight, min_n_components = 999, 0, 0
    # for n_components in n_components_c:
    #     for observe_weight in observe_weight_c:
    #         mae = test(net, test_loader, observe_weight, n_components, test_dataset, positions)
    #         print('n_components: {}, observe_weight: {}, mae: {:.6f}'.format(n_components, observe_weight, mae))
    #         if mae < min_mae:
    #             min_mae, min_observe_weight, min_n_components = mae, observe_weight, n_components
    # print('observe_weight: {}, n_components: {}, mae: {:.6f}'.format(min_observe_weight, min_n_components, min_mae))

    test(net, test_loader, 100, 300, test_dataset, positions)
