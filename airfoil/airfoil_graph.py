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
from torch_geometric.nn import radius_graph

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(filename)

from model.mlp import MLP
from model.gnn import GraphSAGE
from data.dataset import AirfoilGraphDataset
from utils.misc import save_model, prep_experiment
from utils.options import parses
from utils.visualization import plot3x1
from utils.utils import cre

# Configure the arguments
args = parses()
args.exp = 'graph_airfoil_8_p'
args.epochs = 200
args.batch_size = 2
args.plot_freq = 10
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
    encoder = MLP(layers=[7, 64, 64, 16])
    decoder = MLP(layers=[16, 64, 64, 1])
    hparams = {'encoder': [7, 64, 64, 16], 'decoder': [16, 64, 64, 1], 'nb_hidden_layers': 5,
               'size_hidden_layers': 64, 'max_neighbors': 64, 'bn_bool': True,
               'subsampling': 32000, 'r': 0.05}
    net = GraphSAGE(hparams, encoder, decoder).cuda()

    # Build data loader
    train_dataset = AirfoilGraphDataset(index=[i for i in range(500)], type='vx', expand=4)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    val_dataset = AirfoilGraphDataset(index=[i for i in range(500, 700)], type='vx')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    # Build optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.965)

    edge_index = radius_graph(x=train_dataset.pos.cuda(), r=0.04, loop=True, max_num_neighbors=64)
    for epoch in range(args.epochs):
        # Training procedure
        train_loss, train_num = 0., 0.
        for i, (inputs, outputs) in enumerate(train_loader):
            inputs, outputs = inputs.cuda(), outputs.cuda()
            pre = net(inputs.flatten(2).permute(0, 2, 1), edge_index).reshape(-1, 1, 256, 256)
            pre[:, :, train_dataset.airfoil_mask] = 0
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
                    pre = net(inputs.flatten(2).permute(0, 2, 1), edge_index).reshape(-1, 1, 256, 256)
                    pre[:, :, train_dataset.airfoil_mask] = 0
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
    args.snapshot = '/mnt/jfs/zhaoxiaoyu/Gappy_POD/airfoil/logs/ckpt/mlp_airfoil_32_p/best_epoch_295_loss_0.00024823.pth'

    # Define data loader
    test_dataset = AirfoilGraphDataset(index=index, type='vx')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    # Load trained network
    net = MLP(layers=[32, 128, 1280, 4800, 256 * 256]).cuda()
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
            pre = net(inputs).reshape(-1, 1, 256, 256)
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


if __name__ == '__main__':
    train()
    # test(index=[i for i in range(700, 1000)])
