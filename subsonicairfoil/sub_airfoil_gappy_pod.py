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
import h5py

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(filename)

from model.gappy_pod import GappyPod1D
from data.dataset import SubsonicAirfoilPodDataset
from utils.utils import cre
from utils.visualization import plot3x1_coor

# Configure the arguments
batch_size = 16
torch.cuda.set_device(0)


def test(n_components=20, positions=np.array([2644, 4263, 2618, 9630]),
         data_path='/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_test.h5'):
    # Define data loader
    test_dataset = SubsonicAirfoilPodDataset(positions, n_components=20, type='vy',
                                             data_path=data_path,
                                             norm=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    # Load data
    f = h5py.File('/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_train.h5', 'r')
    data = f['vy'][:]
    data = torch.from_numpy(data).float()
    f.close()
    gappy_pod = GappyPod1D(
        data=data, map_size=data.shape[1], n_components=n_components,
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

    # plot3x1_coor(outputs[-1, :].cpu().numpy(), pre[-1, :].cpu().numpy(),
    #              file_name='test.png', x_coor=test_dataset.x_coor,
    #              y_coor=test_dataset.y_coor)
    return test_mae / test_num


if __name__ == '__main__':
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
    # positions = positions[:100]

    # select the optimal number of modes
    results = []
    for i in range(30, 50):
        results.append([i, test(n_components=i, positions=np.array(positions),
                                data_path='/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_val.h5')])
    results = np.array(results)
    print(results[np.argmin(results[:, 1]), :])

    print(int(results[np.argmin(results[:, 1]), :][0]))
    test(n_components=int(results[np.argmin(results[:, 1]), :][0]),
         positions=np.array(positions),
         data_path='/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_test.h5')
