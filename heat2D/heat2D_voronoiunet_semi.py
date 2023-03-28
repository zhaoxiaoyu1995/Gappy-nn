# -*- coding: utf-8 -*-
# @Time    : 2022/4/20 16:08
# @Author  : zhaoxiaoyu
# @File    : heat2D_voronoiunet.py
import torch
import torch.nn.functional as F
import logging
import os
import sys
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import numpy as np
import h5py

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(filename)

from model.cnn import UNet
from model.gappy_pod import GappyPod
from data.dataset import HeatInterpolDataset, HeatInterpolGappyDataset
from utils.misc import save_model, prep_experiment
from utils.options import parses
from utils.visualization import plot3x1
from utils.utils import cre

# Configure the arguments
args = parses()
args.exp = 'voronoicnn_semi2_heat_200'
args.epochs = 300
args.batch_size = 8
print(args)
torch.cuda.set_device(args.gpu_id)
cudnn.benchmark = True


def train():
    # Prepare the experiment environment
    tb_writer = prep_experiment(args)
    # Create figure dir
    args.fig_path = args.exp_path + '/figure'
    os.makedirs(args.fig_path, exist_ok=True)
    args.best_record = {'epoch': -1, 'loss': 1e10}

    # Build neural network
    net = UNet(in_channels=2, out_channels=1).cuda()
    pod_index = [i for i in range(200)]
    f = h5py.File('/mnt/jfs/zhaoxiaoyu/data/heat/temperature.h5', 'r')
    data = f['u'][pod_index, :, :, :]
    data = (data - 298.0) / 50.0
    gappy_pod = GappyPod(
        data=data, map_size=data.shape[-2:], n_components=17,
        positions=np.array(
            [[33, 33], [33, 66], [33, 99], [33, 132], [33, 165], [66, 33], [66, 66], [66, 99], [66, 132], [66, 165],
             [99, 33], [99, 66], [99, 99], [99, 132], [99, 165], [132, 33], [132, 66], [132, 99], [132, 132],
             [132, 165],
             [165, 33], [165, 66], [165, 99], [165, 132], [165, 165]])
    )

    # Build data loader
    train_dataset = HeatInterpolDataset(index=[i for i in range(200)], expand=15)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    val_dataset = HeatInterpolDataset(index=[i for i in range(4800, 5000)])
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
    semi_dataset = HeatInterpolGappyDataset(index=[i for i in range(1000, 4000)])

    # Build optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    # observe_weight = 20000
    for epoch in range(args.epochs):
        # Training procedure
        train_loss, train_num = 0., 0.
        semi_loader = DataLoader(semi_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
        semi_loader = iter(semi_loader)

        # if epoch in [100, 150, 200, 250]:
        #     observe_weight = observe_weight / 5.0
        #     gappy_pod = GappyPod(
        #         data=data, map_size=data.shape[-2:], n_components=12,
        #         positions=np.array(
        #             [[33, 33], [33, 66], [33, 99], [33, 132], [33, 165], [66, 33], [66, 66], [66, 99], [66, 132],
        #              [66, 165],
        #              [99, 33], [99, 66], [99, 99], [99, 132], [99, 165], [132, 33], [132, 66], [132, 99], [132, 132],
        #              [132, 165],
        #              [165, 33], [165, 66], [165, 99], [165, 132], [165, 165]])
        #     )
        for i, (inputs, outputs) in enumerate(train_loader):
            inputs, outputs = inputs.cuda(), outputs.cuda()
            pre = net(inputs)

            semi_inputs, _, observes = next(semi_loader)
            semi_inputs, observes = semi_inputs.cuda(), observes.cuda()
            semi_pre = net(semi_inputs)
            semi_outputs = gappy_pod.reconstruct(observes)

            loss = F.l1_loss(outputs, pre) + 0.1 * F.l1_loss(semi_pre, semi_outputs)

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
                plot3x1(outputs[-1, 0, :, :].cpu().numpy(), pre[-1, 0, :, :].cpu().numpy(),
                        file_name=args.fig_path + f'/epoch{epoch}.png')


def test(index):
    # Path of trained network
    args.snapshot = '/mnt/jfs/zhaoxiaoyu/Gappy_POD/heat2D/logs/ckpt/voronoicnn_heat_49/best_epoch_296_loss_0.00006503.pth'

    # Define data loader
    test_dataset = HeatInterpolDataset(index=index)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    # Load trained network
    net = UNet(in_channels=2, out_channels=1).cuda()
    net.load_state_dict(torch.load(args.snapshot)['state_dict'])
    print('load models: ' + args.snapshot)

    # Test procedure
    net.eval()
    test_mae, test_rmse, test_num = 0.0, 0.0, 0.0
    test_max_ae = 0
    for i, (inputs, outputs) in enumerate(test_loader):
        N, _, _, _ = inputs.shape
        inputs, outputs = inputs.cuda(), outputs.cuda()
        with torch.no_grad():
            pre = net(inputs)
        test_num += N
        test_mae += F.l1_loss(outputs, pre).item() * N
        test_rmse += torch.sum(cre(outputs, pre, 2))
        test_max_ae += torch.sum(torch.max(torch.abs(outputs - pre).flatten(1), dim=1)[0]).item()
    print('test mae:', 50 * test_mae / test_num)
    print('test rmse:', 50 * test_rmse / test_num)
    print('test max_ae:', 50 * test_max_ae / test_num)

    plot3x1(outputs[-1, 0, :, :].cpu().numpy(), pre[-1, 0, :, :].cpu().numpy(), './test.png')


if __name__ == '__main__':
    train()
    # test(index=[i for i in range(0, 4000)])
    # test(index=[i for i in range(4000, 5000)])
    # test(index=[i for i in range(5000, 6000)])
