# -*- coding: utf-8 -*-
# @Time    : 2022/4/20 16:08
# @Author  : zhaoxiaoyu
# @File    : cylinder_mlp.py
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
from utils.misc import save_model, prep_experiment
from utils.options import parses
from utils.visualization import plot3x1_coor
from utils.utils import cre

# Configure the arguments
args = parses()
args.exp = 'mlp_sub_airfoil_64_vy'
args.epochs = 300
args.batch_size = 2
args.plot_freq = 50
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
    net = MLP(layers=[64, 128, 1280, 4800, 16339]).cuda()

    # Build data loader
    positions = [15733, 15844, 15842, 15751, 15847, 15840, 15819, 15831, 15091, 15016, 15118, 15096, 15010, 15083,
                 15062, 15074, 13844, 13871, 13996, 13887, 13892, 13977, 13955, 13967, 11431, 11400, 11428, 11432,
                 11549, 11509, 11482, 11494, 5136, 5106, 5152, 5180, 5097, 5015, 5056, 5068, 2493, 2468, 2471, 2464,
                 2592, 2507, 2544, 2556, 1248, 1323, 1359, 1244, 1329, 1255, 1289, 1301, 582, 494, 568, 598, 492,
                 498, 532, 544]
    positions = positions[:64]
    train_dataset = SubsonicAirfoilDataset(type='vy', data_path='/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_train.h5',
                                           positions=positions)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    val_dataset = SubsonicAirfoilDataset(type='vy', data_path='/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_val.h5',
                                         positions=positions)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    # Build optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)

    mean, std = train_dataset.min, train_dataset.max - train_dataset.min
    mean, std = torch.from_numpy(mean).cuda(), torch.from_numpy(std).cuda()
    for epoch in range(args.epochs):
        # Training procedure
        train_loss, train_num = 0., 0.
        for i, (inputs, outputs) in enumerate(train_loader):
            inputs, outputs = inputs.cuda(), outputs.cuda()
            pre = net(inputs)
            loss = F.l1_loss(outputs, pre)

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
            val_loss, val_num = 0., 0.
            for i, (inputs, outputs) in enumerate(val_loader):
                inputs, outputs = inputs.cuda(), outputs.cuda()
                with torch.no_grad():
                    pre = net(inputs)
                loss = F.l1_loss(outputs, pre)

                val_loss += loss.item() * inputs.shape[0]
                val_num += inputs.shape[0]

            # Record results by tensorboard
            val_loss = val_loss / val_num
            tb_writer.add_scalar('val_loss', val_loss, epoch)
            logging.info("Epoch: {}, Val_loss: {}".format(epoch, val_loss))
            if val_loss < args.best_record['loss']:
                save_model(args, epoch, val_loss, net)
            net.train()

            # Plotting
            if epoch % args.plot_freq == 0:
                pre = pre * std + mean
                outputs = outputs * std + mean
                plot3x1_coor(outputs[-1, :].cpu().numpy(), pre[-1, :].cpu().numpy(),
                             file_name=args.fig_path + f'/epoch{epoch}.png', x_coor=train_dataset.x_coor,
                             y_coor=train_dataset.y_coor)


def test():
    # Path of trained network
    args.snapshot = '/mnt/jfs/zhaoxiaoyu/Gappy_POD/subsonicairfoil/logs/ckpt/mlp_sub_airfoil_4_vx/best_epoch_294_loss_0.00558066.pth'

    # Define data loader
    positions = [2644, 4263, 2618, 10286, 10720, 11823, 13903, 7968, 15217, 15974, 13358, 15984, 15142, 15232,
                 11937, 4852, 16271, 3856, 14399, 2242, 14014, 4693, 16320, 1026, 1879, 15573, 2259, 1040, 294,
                 808, 16298, 305]
    positions = positions[:4]
    test_dataset = SubsonicAirfoilDataset(type='vx', data_path='/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_test.h5',
                                          positions=positions)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    # Load trained network
    net = MLP(layers=[4, 128, 1280, 4800, 16339]).cuda()
    net.load_state_dict(torch.load(args.snapshot)['state_dict'])
    print('load models: ' + args.snapshot)

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


if __name__ == '__main__':
    train()
    # test()
