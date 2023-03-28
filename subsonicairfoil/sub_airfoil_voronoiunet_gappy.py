# -*- coding: utf-8 -*-
# @Time    : 2022/4/20 16:08
# @Author  : zhaoxiaoyu
# @File    : airfoil_voronoiunet.py
import torch
import torch.nn.functional as F
import logging
import os
import sys
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from sklearn.decomposition import PCA
import numpy as np
import scipy.io as sio

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(filename)

from model.cnn import UNet
from data.dataset import SubsonicAirfoilInterpolGappyDataset
from utils.options import parses
from utils.visualization import plot3x1_coor
from utils.utils import cre

# Configure the arguments
args = parses()
args.exp = 'voronoiunet_subairfoil_p_32'
args.epochs = 300
args.batch_size = 4
print(args)
torch.cuda.set_device(args.gpu_id)
cudnn.benchmark = True


class GappyPodWeight1D():
    def __init__(self, data, map_size=16339, n_components=50,
                 positions=np.array([50, 25, 62, 30]), observe_weight=50):
        self.data = data
        self.pca = PCA(n_components=n_components)
        self.pca.fit(self.data.reshape(data.shape[0], -1))

        self.positions_observe = positions
        self.positions_pre = sio.loadmat('/mnt/jfs/zhaoxiaoyu/Gappy_POD/index2.mat')['data'].reshape(-1)
        self.map_size = map_size
        self.observe_weight = observe_weight

        components = self.pca.components_
        means = self.pca.mean_
        component, mean = components.reshape(-1, self.map_size), means.reshape(-1, self.map_size)

        component_mask, mean_mask = [], []
        for i in range(self.positions_pre.shape[0]):
            component_mask.append(component[:, self.positions_pre[i]].reshape(-1, 1))
            mean_mask.append(mean[:, self.positions_pre[i]].reshape(-1, 1))
        for i in range(self.positions_observe.shape[0]):
            component_mask.append(
                self.observe_weight * component[:, self.positions_observe[i]].reshape(-1, 1)
            )
            mean_mask.append(
                self.observe_weight * mean[:, self.positions_observe[i]].reshape(-1, 1)
            )

        self.component_mask = torch.from_numpy(
            np.concatenate(component_mask, axis=-1)).float().cuda().unsqueeze(0)
        self.mean_mask = torch.from_numpy(np.concatenate(mean_mask, axis=-1).T).float().cuda()
        self.components_ = torch.from_numpy(components).float().cuda()
        self.mean_ = torch.from_numpy(means).reshape(1, -1).float().cuda()

    def reconstruct(self, pres, inputs, weight):
        component_mask = self.component_mask.repeat(inputs.shape[0], 1, 1)
        observe = torch.cat([(weight * pres).flatten(1), self.observe_weight * inputs], dim=-1)
        mask_temp = torch.cat([weight.flatten(1), torch.ones(inputs.shape[0], self.positions_observe.shape[0]).cuda()],
                              dim=1)
        observe = (observe.T - self.mean_mask * mask_temp.T).unsqueeze(dim=1).permute(2, 0, 1)
        component_mask_temp = component_mask * mask_temp.unsqueeze(dim=1)
        coff_pre = torch.linalg.inv(
            component_mask_temp @ component_mask_temp.permute(0, 2, 1)) @ component_mask_temp @ observe
        coff_pre = coff_pre.squeeze(dim=-1)
        recons = self.inverse_transform(coff_pre)
        return recons

    def inverse_transform(self, coff):
        return coff @ self.components_ + self.mean_


def test(net, test_loader, observe_weight=50, n_components=50, test_dataset=None, positions=[1, 2]):
    # Load data
    import h5py
    import numpy as np

    f = h5py.File('/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_train.h5', 'r')
    data = f['vy'][:]
    data = (data - test_dataset.min) / (test_dataset.max - test_dataset.min + 1e-8)
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
    for i, (inputs, outputs, observes, labels) in enumerate(test_loader):
        N, _, _, _ = inputs.shape
        inputs, outputs, observes, labels = inputs.cuda(), outputs.cuda(), observes.cuda(), labels.cuda()
        outputs = labels * std + mean
        with torch.no_grad():
            pre = net(inputs)
            pre = gappy_pod.reconstruct(pre, observes, weight=torch.ones_like(pre))
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
    import scipy.io as sio
    sio.savemat('mlp_cnn_vy.mat', {
        'true': outputs[-1, :].cpu().numpy(),
        'pre': pre[-1, :].cpu().numpy()
    })
    return test_mae / test_num


if __name__ == '__main__':
    # test_dataset = AirfoilDataset(index=[i for i in range(500, 700)], type='p')
    positions = [15733, 15844, 15842, 15751, 15847, 15840, 15819, 15831, 15091, 15016, 15118, 15096, 15010, 15083,
                 15062, 15074, 13844, 13871, 13996, 13887, 13892, 13977, 13955, 13967, 11431, 11400, 11428, 11432,
                 11549, 11509, 11482, 11494, 5136, 5106, 5152, 5180, 5097, 5015, 5056, 5068, 2493, 2468, 2471, 2464,
                 2592, 2507, 2544, 2556, 1248, 1323, 1359, 1244, 1329, 1255, 1289, 1301, 582, 494, 568, 598, 492,
                 498, 532, 544]
    positions = positions[:64]
    test_dataset = SubsonicAirfoilInterpolGappyDataset(type='vy',
                                                       data_path='/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_test.h5',
                                                       positions=positions)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)

    # Path of trained network
    args.snapshot = '/mnt/jfs/zhaoxiaoyu/Gappy_POD/subsonicairfoil/logs/ckpt/voronoiunet_subairfoil_vy_64/best_epoch_286_loss_0.00188506.pth'

    # Load trained network
    net = UNet(in_channels=3, out_channels=1).cuda()
    net.load_state_dict(torch.load(args.snapshot)['state_dict'])
    print('load models: ' + args.snapshot)

    # observe_weight_c = [10, 20, 50, 100]
    # n_components_c = [50, 80, 100, 300, 400]
    # min_mae, min_observe_weight, min_n_components = 999, 0, 0
    # for n_components in n_components_c:
    #     for observe_weight in observe_weight_c:
    #         mae = test(net, test_loader, observe_weight, n_components, test_dataset, positions)
    #         print('n_components: {}, observe_weight: {}, mae: {:.6f}'.format(n_components, observe_weight, mae))
    #         if mae < min_mae:
    #             min_mae, min_observe_weight, min_n_components = mae, observe_weight, n_components
    # print('observe_weight: {}, n_components: {}, mae: {:.6f}'.format(min_observe_weight, min_n_components, min_mae))

    test(net, test_loader, 50, 80, test_dataset, positions)
