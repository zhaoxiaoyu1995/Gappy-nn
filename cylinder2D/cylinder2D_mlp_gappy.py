# -*- coding: utf-8 -*-
# @Time    : 2022/4/20 16:08
# @Author  : zhaoxiaoyu
# @File    : cylinder_mlp.py
import torch
import numpy as np
import pickle
import torch.nn.functional as F
import logging
import os
import sys
from torch.utils.data import DataLoader

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(filename)

from model.mlp import MLP
from model.gappy_pod import GappyPodWeight
from data.dataset import CylinderDataset
from utils.misc import save_model, prep_experiment
from utils.options import parses
from utils.visualization import plot3x1
from utils.utils import cre

# Configure the arguments
args = parses()
args.exp = 'mlp_cylinder_32'
args.epochs = 300
args.batch_size = 16
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
    net = MLP(layers=[32, 128, 1280, 4800, 21504]).cuda()

    # Build data loader
    train_dataset = CylinderDataset(index=[i for i in range(3500)])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    val_dataset = CylinderDataset(index=[i for i in range(3500, 4250)])
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    # Build optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    for epoch in range(args.epochs):
        # Training procedure
        train_loss, train_num = 0., 0.
        for i, (inputs, outputs) in enumerate(train_loader):
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
            val_loss, val_num = 0., 0.
            for i, (inputs, outputs) in enumerate(val_loader):
                inputs, outputs = inputs.cuda(), outputs.cuda()
                with torch.no_grad():
                    pre = net(inputs)
                loss = F.l1_loss(outputs.flatten(1), pre)

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
                plot3x1(outputs[-1, 0, :, :].cpu().numpy(), pre[-1, :].reshape(112, 192).cpu().numpy(),
                        file_name=args.fig_path + f'/epoch{epoch}.png')


def test(index, observe_weight=50, n_components=50):
    # Path of trained network
    args.snapshot = '/mnt/jfs/zhaoxiaoyu/Gappy_POD/cylinder2D/logs/ckpt/mlp_cylinder_8/best_epoch_298_loss_0.00004666.pth'

    # Load data
    pod_index = [i for i in range(4250)]
    df = open('/home/ubuntu/zhaoxiaoyu/data/cylinder/Cy_Taira.pickle', 'rb')
    data = np.transpose(pickle.load(df), (0, 3, 1, 2))[pod_index, :, :, :]
    df.close()
    gappy_pod = GappyPodWeight(
        data=data, map_size=data.shape[-2:], n_components=n_components,
        positions=np.array([[50, 25], [62, 30], [50, 35], [62, 40], [50, 45], [62, 50], [50, 55], [62, 60]]),
        observe_weight=observe_weight
    )

    # Define data loader
    test_dataset = CylinderDataset(index=index)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    # Load trained network
    net = MLP(layers=[8, 128, 1280, 4800, 21504]).cuda()
    net.load_state_dict(torch.load(args.snapshot)['state_dict'])
    print('load models: ' + args.snapshot)

    # Test procedure
    net.eval()
    test_mae, test_rmse, test_num = 0.0, 0.0, 0.0
    test_max_ae = 0
    for i, (inputs, outputs) in enumerate(test_loader):
        N, _ = inputs.shape
        inputs, outputs = inputs.cuda(), outputs.cuda()
        with torch.no_grad():
            pre = net(inputs)
            pre = gappy_pod.reconstruct(pre, inputs, weight=torch.ones_like(pre))
        test_num += N
        test_mae += F.l1_loss(outputs, pre).item() * N
        test_rmse += torch.sum(cre(outputs, pre, 2))
        test_max_ae += torch.sum(torch.max(torch.abs(outputs - pre).flatten(1), dim=1)[0]).item()
    print('test mae:', test_mae / test_num)
    print('test rmse:', test_rmse / test_num)
    print('test max_ae:', test_max_ae / test_num)

    plot3x1(outputs[-1, 0, :, :].cpu().numpy(), pre[-1, :].reshape(112, 192).cpu().numpy(), './test.png')

    import scipy.io as sio
    sio.savemat('gappy_mlp.mat', {
        'true': outputs[-1, 0, :, :].cpu().numpy(),
        'pre': pre[-1, :].reshape(112, 192).cpu().numpy()
    })

    return test_mae / test_num


if __name__ == '__main__':
    # train()

    # observe_weight_c = [10, 20, 50, 100, 200]
    # n_components_c = [20, 30, 40, 50, 60]
    # min_mae, min_observe_weight, min_n_components = 999, 0, 0
    # for n_components in n_components_c:
    #     for observe_weight in observe_weight_c:
    #         mae = test([i for i in range(3500, 4250)], observe_weight, n_components)
    #         print('n_components: {}, observe_weight: {}, mae: {:.6f}'.format(n_components, observe_weight, mae))
    #         if mae < min_mae:
    #             min_mae, min_observe_weight, min_n_components = mae, observe_weight, n_components
    # print('observe_weight: {}, n_components: {}, mae: {:.6f}'.format(min_observe_weight, min_n_components, min_mae))

    test([i for i in range(4900, 4901)], 100, 30)
