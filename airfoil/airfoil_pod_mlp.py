# -*- coding: utf-8 -*-
# @Time    : 2022/4/20 16:08
# @Author  : zhaoxiaoyu
# @File    : airfoil_pod_mlp.py
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
from data.dataset import AirfoilPodDataset
from utils.misc import save_model, prep_experiment
from utils.options import parses
from utils.visualization import plot3x1
from utils.utils import cre

# Configure the arguments
args = parses()
args.exp = 'pod_airfoil_32_p'
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
    net = MLP(layers=[32, 64, 64, 64, 20]).cuda()

    # Build data loader
    positions = np.array(
        [[65, 126], [90, 139], [65, 151], [90, 114], [90, 164], [115, 139], [65, 101], [115, 164],
         [65, 176], [115, 114], [90, 89], [90, 189], [115, 89], [115, 189], [40, 164], [65, 76],
         [65, 201], [140, 137], [92, 214], [90, 64], [40, 189], [140, 162], [190, 127], [117, 214],
         [140, 187], [67, 226], [40, 94], [115, 64], [40, 214], [140, 112], [92, 239], [65, 51]]
    )
    train_dataset = AirfoilPodDataset(pod_index=[i for i in range(700)], index=[i for i in range(500)],
                                      positions=positions, n_components=20, type='p', expand=4)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    val_dataset = AirfoilPodDataset(pod_index=[i for i in range(700)], index=[i for i in range(500, 700)],
                                    positions=positions, n_components=20, type='p')
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
                plot3x1(labels[-1, 0, :, :].cpu().numpy(), pre_maps[-1, 0, :, :].cpu().numpy(),
                        file_name=args.fig_path + f'/epoch{epoch}.png')


def test(index):
    # Path of trained network
    args.snapshot = '/mnt/jfs/zhaoxiaoyu/Gappy_POD/airfoil/logs/ckpt/pod_airfoil_8_p/best_epoch_227_loss_0.37525984.pth'

    # Define data loader
    positions = np.array(
        [[65, 126], [90, 139], [65, 151], [90, 114], [90, 164], [115, 139], [65, 101], [115, 164]]
    )
    test_dataset = AirfoilPodDataset(pod_index=[i for i in range(700)], index=index,
                                     positions=positions, n_components=20, type='p')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    # Load trained network
    net = MLP(layers=[8, 64, 64, 64, 20]).cuda()
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

    # plot3x1(labels[-1, 0, :, :].cpu().numpy(), pre[-1, 0, :, :].cpu().numpy(), './test.png')
    import scipy.io as sio
    sio.savemat('mlp_pod_p.mat', {
        'true': labels[-1, 0, :, :].cpu().numpy(),
        'pre': pre[-1, 0, :, :].cpu().numpy()
    })


if __name__ == '__main__':
    # train()
    test(index=[i for i in range(999, 1000)])
