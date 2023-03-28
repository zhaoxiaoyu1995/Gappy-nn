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
from torch_geometric.nn import radius_graph, knn_graph

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(filename)

from model.mlp import MLP
from model.gnn import GraphSAGEMLP, GCN_NET
from data.dataset import SubsonicAirfoilDataset
from utils.misc import save_model, prep_experiment
from utils.options import parses
from utils.visualization import plot3x1_coor
from utils.utils import cre

# Configure the arguments
args = parses()
args.exp = 'graph2_airfoil_169_p'
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
    encoder = MLP(layers=[169, 16339])
    decoder = MLP(layers=[64, 64, 64, 1])
    hparams = {'encoder': [4, 16339, 1], 'decoder': [64, 64, 64, 1], 'nb_hidden_layers': 5,
               'size_hidden_layers': 64, 'max_neighbors': 64, 'bn_bool': True,
               'subsampling': 32000, 'r': 0.05}
    net = GraphSAGEMLP(hparams, encoder, decoder).cuda()
    # net = GCN_NET(hparams, encoder, decoder).cuda()

    # Build data loader
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
    train_dataset = SubsonicAirfoilDataset(type='p', data_path='/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_train.h5',
                                           positions=positions)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    val_dataset = SubsonicAirfoilDataset(type='p', data_path='/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_val.h5',
                                         positions=positions)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    # Build optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)

    edge_index = knn_graph(x=train_dataset.pos.cuda(), loop=True, k=100)
    for epoch in range(args.epochs):
        # Training procedure
        train_loss, train_num = 0., 0.
        for i, (inputs, outputs) in enumerate(train_loader):
            inputs, outputs = inputs.cuda(), outputs.cuda()
            pre = net(inputs, edge_index).reshape(inputs.shape[0], -1)
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
                    pre = net(inputs, edge_index).reshape(inputs.shape[0], -1)
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
                plot3x1_coor(outputs[-1, :].cpu().numpy(), pre[-1, :].cpu().numpy(),
                             file_name=args.fig_path + f'/epoch{epoch}.png', x_coor=train_dataset.x_coor,
                             y_coor=train_dataset.y_coor)


def test():
    # Path of trained network
    args.snapshot = '/mnt/jfs/zhaoxiaoyu/Gappy_POD/subsonicairfoil/logs/ckpt/graph2_airfoil_64_vx/best_epoch_293_loss_0.00654930.pth'

    # Define data loader
    positions = [15733, 15844, 15842, 15751, 15847, 15840, 15819, 15831, 15091, 15016, 15118, 15096, 15010, 15083,
                 15062, 15074, 13844, 13871, 13996, 13887, 13892, 13977, 13955, 13967, 11431, 11400, 11428, 11432,
                 11549, 11509, 11482, 11494, 5136, 5106, 5152, 5180, 5097, 5015, 5056, 5068, 2493, 2468, 2471, 2464,
                 2592, 2507, 2544, 2556, 1248, 1323, 1359, 1244, 1329, 1255, 1289, 1301, 582, 494, 568, 598, 492,
                 498, 532, 544]
    positions = positions[:64]
    test_dataset = SubsonicAirfoilDataset(type='vx', data_path='/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_test.h5',
                                          positions=positions)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    # Load trained network
    encoder = MLP(layers=[64, 16339])
    decoder = MLP(layers=[64, 64, 64, 1])
    hparams = {'encoder': [4, 16339, 1], 'decoder': [64, 64, 64, 1], 'nb_hidden_layers': 5,
               'size_hidden_layers': 64, 'max_neighbors': 64, 'bn_bool': True,
               'subsampling': 32000, 'r': 0.05}
    net = GraphSAGEMLP(hparams, encoder, decoder).cuda()
    # net = GCN_NET(hparams, encoder, decoder).cuda()
    net.load_state_dict(torch.load(args.snapshot)['state_dict'])
    print('load models: ' + args.snapshot)

    # Test procedure
    net.eval()
    test_mae, test_rmse, test_num = 0.0, 0.0, 0.0
    test_max_ae = 0
    mean, std = test_dataset.min, test_dataset.max - test_dataset.min
    mean, std = torch.from_numpy(mean).cuda(), torch.from_numpy(std).cuda()
    edge_index = knn_graph(x=test_dataset.pos.cuda(), loop=True, k=100)
    for i, (inputs, outputs) in enumerate(test_loader):
        N, _ = inputs.shape
        inputs, outputs = inputs.cuda(), outputs.cuda()
        outputs = outputs * std + mean
        with torch.no_grad():
            pre = net(inputs, edge_index).reshape(inputs.shape[0], -1)
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
    sio.savemat('mlp_graph_vx.mat', {
        'true': outputs[-1, :].cpu().numpy(),
        'pre': pre[-1, :].cpu().numpy()
    })


if __name__ == '__main__':
    # train()
    test()
