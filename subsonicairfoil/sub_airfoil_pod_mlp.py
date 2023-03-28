# -*- coding: utf-8 -*-
# @Time    : 2022/4/20 16:08
# @Author  : zhaoxiaoyu
# @File    : suub_airfoil_pod_mlp.py
import torch
import torch.nn.functional as F
import logging
import os
import sys
import numpy as np
from torch.utils.data import DataLoader

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(filename)

from model.mlp import MLP
from data.dataset import SubsonicAirfoilPodDataset
from utils.misc import save_model, prep_experiment
from utils.options import parses
from utils.visualization import plot3x1_coor
from utils.utils import cre

# Configure the arguments
args = parses()
args.exp = 'pod_sub_airfoil_169_p'
args.epochs = 300
args.batch_size = 4
print(args)
torch.cuda.set_device(args.gpu_id)


def train():
    # Prepare the experiment environment
    tb_writer = prep_experiment(args)
    # Create figure dir
    args.fig_path = args.exp_path + '/figure'
    os.makedirs(args.fig_path, exist_ok=True)
    args.best_record = {'epoch': -1, 'loss': 1e10}

    # Build neural network
    net = MLP(layers=[169, 64, 64, 64, 35]).cuda()

    # Build data loader
    positions = np.array(
        [16097, 16016, 16087, 15986, 15987, 15978, 15981, 15980, 16018, 16046, 16061, 16069, 16075, 15657,
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
    )
    positions = positions[:169]
    train_dataset = SubsonicAirfoilPodDataset(positions, n_components=35, type='p',
                                              data_path='/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_train.h5')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    val_dataset = SubsonicAirfoilPodDataset(positions, n_components=35, type='p',
                                            data_path='/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_val.h5')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    # Build optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)

    for epoch in range(args.epochs):
        # Training procedure
        train_loss, train_num = 0., 0.
        for i, (inputs, outputs, _) in enumerate(train_loader):
            inputs, outputs = inputs.cuda(), outputs.cuda()
            pre = net(inputs)
            loss = F.l1_loss(outputs.flatten(1), pre)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record results by tensorboard
            tb_writer.add_scalar('train_loss', loss, i + epoch * len(train_loader))
            train_loss += loss.item() * inputs.shape[0]
            train_num += inputs.shape[0]

        train_loss = train_loss / train_num
        logging.info("Epoch: {}, Avg_loss: {}".format(epoch, train_loss))
        scheduler.step()

        # Validation procedure
        if epoch % args.val_interval == 0:
            net.eval()
            val_loss, val_num, val_mae = 0., 0., 0.
            for i, (inputs, outputs, labels) in enumerate(val_loader):
                inputs, outputs, labels = inputs.cuda(), outputs.cuda(), labels.cuda()
                with torch.no_grad():
                    pre = net(inputs)
                loss = F.l1_loss(outputs, pre)
                pre_maps = val_dataset.inverse_transform(pre)
                mae = F.l1_loss(labels, pre_maps)

                val_loss += loss.item() * inputs.shape[0]
                val_mae += mae.item() * inputs.shape[0]
                val_num += inputs.shape[0]

            # Record results by tensorboard
            val_loss = val_loss / val_num
            val_mae = val_mae / val_num
            tb_writer.add_scalar('val_loss', val_loss, epoch)
            tb_writer.add_scalar('val_mae', val_mae, epoch)
            logging.info("Epoch: {}, Val_loss: {}, Val_mae: {}".format(epoch, val_loss, val_mae))
            if val_mae < args.best_record['loss']:
                save_model(args, epoch, val_mae, net)
            net.train()

            # Plotting
            if epoch % args.plot_freq == 0:
                plot3x1_coor(labels[-1, :].cpu().numpy(), pre_maps[-1, :].cpu().numpy(),
                             file_name=args.fig_path + f'/epoch{epoch}.png',
                             x_coor=train_dataset.x_coor, y_coor=train_dataset.y_coor)


def test():
    # Path of trained network
    args.snapshot = '/mnt/jfs/zhaoxiaoyu/Gappy_POD/subsonicairfoil/logs/ckpt/pod_sub_airfoil_64_vy/best_epoch_283_loss_0.41012971.pth'

    # Define data loader
    positions = np.array(
        [15733, 15844, 15842, 15751, 15847, 15840, 15819, 15831, 15091, 15016, 15118, 15096, 15010, 15083,
         15062, 15074, 13844, 13871, 13996, 13887, 13892, 13977, 13955, 13967, 11431, 11400, 11428, 11432,
         11549, 11509, 11482, 11494, 5136, 5106, 5152, 5180, 5097, 5015, 5056, 5068, 2493, 2468, 2471, 2464,
         2592, 2507, 2544, 2556, 1248, 1323, 1359, 1244, 1329, 1255, 1289, 1301, 582, 494, 568, 598, 492,
         498, 532, 544]
    )
    positions = positions[:64]
    test_dataset = SubsonicAirfoilPodDataset(positions, n_components=35, type='vy',
                                             data_path='/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_test.h5')
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4)

    # Load trained network
    net = MLP(layers=[64, 64, 64, 64, 35]).cuda()
    net.load_state_dict(torch.load(args.snapshot)['state_dict'])
    print('load models: ' + args.snapshot)

    # Test procedure
    net.eval()
    test_mae, test_rmse, test_num = 0.0, 0.0, 0.0
    test_max_ae = 0
    for i, (inputs, outputs, labels) in enumerate(test_loader):
        N, _ = inputs.shape
        inputs, outputs, labels = inputs.cuda(), outputs.cuda(), labels.cuda()
        with torch.no_grad():
            pre = net(inputs)
        pre = test_dataset.inverse_transform(pre)
        test_num += N
        test_mae += F.l1_loss(labels, pre).item() * N
        test_rmse += torch.sum(cre(labels, pre, 2))
        test_max_ae += torch.sum(torch.max(torch.abs(labels - pre).flatten(1), dim=1)[0]).item()
    print('test mae:', test_mae / test_num)
    print('test rmse:', test_rmse / test_num)
    print('test max_ae:', test_max_ae / test_num)

    plot3x1_coor(labels[-1, :].cpu().numpy(), pre[-1, :].cpu().numpy(),
                 './test.png', test_dataset.x_coor, test_dataset.y_coor)

    import scipy.io as sio
    sio.savemat('mlp_pod_vy.mat', {
        'true': labels[-1, :].cpu().numpy(),
        'pre': pre[-1, :].cpu().numpy()
    })


if __name__ == '__main__':
    # train()
    test()
