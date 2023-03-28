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
from sklearn.decomposition import PCA
import h5py
import numpy as np
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from tqdm import tqdm

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(filename)

from model.cnn import UNet
from data.dataset import HeatLayout
from utils.misc import save_model, prep_experiment, save_model2
from utils.options import parses
from utils.visualization import plot3x1
from utils.utils import cre

# Configure the arguments
args = parses()
args.exp = 'voronoicnn_heatlayout2_100'
args.epochs = 30
args.batch_size = 16
print(args)
torch.cuda.set_device(args.gpu_id)
cudnn.benchmark = True


class PODHeat:
    def __init__(self, n_components, positions, pod_index):
        self.positions = positions
        # Loading data
        f = h5py.File('/mnt/jfs/zhaoxiaoyu/Gappy_POD/heat_layout.h5', 'r')
        pca_data = f['u'][pod_index, :, :, :] - 298.0
        self.n_components = n_components

        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        X_t = self.pca.fit_transform(pca_data.reshape(len(pod_index), -1))
        self.max_coeff, self.min_coeff = np.max(X_t, axis=0).reshape(1, -1), np.min(X_t, axis=0).reshape(1, -1)
        self.size = pca_data.shape[-3:]

        self.train_data_coff = (X_t - self.min_coeff) / (self.max_coeff - self.min_coeff + 1e-10)

    def prepare_data(self, data_index, data_pre, flag=False):
        # Loading data
        f = h5py.File('/mnt/jfs/zhaoxiaoyu/Gappy_POD/heat_layout.h5', 'r')
        data = f['u'][data_index, :, :, :] - 298.0
        f.close()
        data_pre = data_pre[data_index, :, :, :]

        # Normalize the coefficients
        data_coff = self.pca.transform(data.reshape(len(data_index), -1))
        data_coff = (data_coff - self.min_coeff) / (self.max_coeff - self.min_coeff)

        sparse_data = []
        for i in range(self.positions.shape[0]):
            sparse_data.append(data_pre[:, 0, self.positions[i, 0], :][:, self.positions[i, 1]].reshape(-1, 1))
        observe = np.concatenate(sparse_data, axis=-1)
        return observe, data_coff, data

    def inverse_transform(self, coff):
        inverse_coff = coff * (self.max_coeff - self.min_coeff) + self.min_coeff
        return self.pca.inverse_transform(inverse_coff).reshape(coff.shape[0], self.size[0], self.size[1], self.size[2])

    def sample(self, numbers=1000):
        # Loading data
        data_index = (np.random.rand(numbers) * 500).tolist()
        data_index = [int(i) for i in data_index]
        f = h5py.File('/mnt/jfs/zhaoxiaoyu/Gappy_POD/heat_layout.h5', 'r')
        data = f['u'][:][data_index, :, :, :] - 298.0
        f.close()

        # Normalize the coefficients
        data_coff = self.pca.transform(data.reshape(len(data_index), -1))
        data_coff = (data_coff - self.min_coeff) / (self.max_coeff - self.min_coeff)

        sparse_data = []
        for i in range(self.positions.shape[0]):
            sparse_data.append(data[:, 0, self.positions[i, 0], :][:, self.positions[i, 1]].reshape(-1, 1))
        observe = np.concatenate(sparse_data, axis=-1)
        observe = observe + (np.random.randn(numbers, self.positions.shape[0]) - 0.5) * 0.1
        return observe, data_coff

    # def sample(self, n=1000):
    # data_coff = np.random.rand(n, self.n_components)
    # inverse_coff = data_coff * (self.max_coeff - self.min_coeff) + self.min_coeff
    # observe = []
    # for i in tqdm(range(int(n / 100.0))):
    #     data = self.pca.inverse_transform(inverse_coff[i * 100:i * 100 + 100]).reshape(100, 1, 200, 200)
    #     data = data + 0.3 * np.random.randn(100, 1, 200, 200)
    #     sparse_data = []
    #     for j in range(self.positions.shape[0]):
    #         sparse_data.append(data[:, 0, self.positions[j, 0], :][:, self.positions[j, 1]].reshape(-1, 1))
    #     sparse_data = np.concatenate(sparse_data, axis=-1)
    #     observe.append(sparse_data)
    # observe = np.concatenate(observe, axis=0)
    # return observe, data_coff


# Gaussian Process Regression
def ridge_regression(train_inputs, train_outputs, val_inputs, val_outputs):
    X = np.concatenate([train_inputs, val_inputs], axis=0)
    y = np.concatenate([train_outputs, val_outputs], axis=0)
    ps = PredefinedSplit(
        test_fold=np.concatenate([-1 * np.ones((train_inputs.shape[0], 1)), np.zeros((val_inputs.shape[0], 1))],
                                 axis=0)
    )
    tuned_parameter = [{'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.25, 0.30, 0.5, 1.0, 10.0]}]
    ridge = linear_model.Ridge()
    gpr_c = GridSearchCV(estimator=ridge, param_grid=tuned_parameter, cv=ps, n_jobs=4)
    gpr_c.fit(X, y)
    alpha = gpr_c.best_params_['alpha']
    print('The optimal parameters are: \n alpha:', alpha)

    ridge = linear_model.Ridge(alpha=alpha)
    ridge.fit(train_inputs, train_outputs)
    return ridge


# ridge
def ridge_regression2(train_inputs, train_outputs, val_inputs, val_outputs):
    alphas = []
    for i in range(train_outputs.shape[1]):
        X = np.concatenate([train_inputs / 50, val_inputs / 50], axis=0)
        y = np.concatenate([train_outputs[:, i:i + 1], val_outputs[:, i:i + 1]], axis=0)
        ps = PredefinedSplit(
            test_fold=np.concatenate([-1 * np.ones((train_inputs.shape[0], 1)), np.zeros((val_inputs.shape[0], 1))],
                                     axis=0)
        )
        tuned_parameter = [{'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]}]
        ridge = linear_model.Ridge()
        gpr_c = GridSearchCV(estimator=ridge, param_grid=tuned_parameter, cv=ps, n_jobs=4)
        gpr_c.fit(X, y)
        alpha = gpr_c.best_params_['alpha']
        print('The optimal parameters are: \n alpha:', alpha)
        alphas.append(alpha)
    print(alphas)
    ridge = linear_model.Ridge(alpha=np.array(alpha))
    ridge.fit(train_inputs / 50.0, train_outputs)
    return ridge


def train():
    # Prepare the experiment environment
    tb_writer = prep_experiment(args)
    # Create figure dir
    args.fig_path = args.exp_path + '/figure'
    os.makedirs(args.fig_path, exist_ok=True)
    args.best_record = {'epoch': -1, 'loss': 1e10, 'epoch2': -1, 'loss2': 1e10}

    # Build neural network
    net = UNet(in_channels=1, out_channels=1).cuda()

    # Build data loader
    train_dataset = HeatLayout(index=[i for i in range(100)], expand=80)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    val_dataset = HeatLayout(index=[i for i in range(0, 200)])
    val_loader = DataLoader(val_dataset, batch_size=12, num_workers=4)
    n_components = 100
    positions = np.array([[int(200 * np.random.rand(1)), int(200 * np.random.rand(1))] for i in range(1600)])
    pod_index = [i for i in range(100)]
    train_index = [i for i in range(100)]
    val_index = [i for i in range(100, 200)]
    pod_cylinder = PODHeat(n_components, positions, pod_index)

    # Build optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.80)

    for epoch in range(args.epochs):
        # Training procedure
        train_loss, train_num = 0., 0.
        for i, (inputs, outputs) in tqdm(enumerate(train_loader)):
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
            pres, output_list = [], []
            for i, (inputs, outputs) in tqdm(enumerate(val_loader)):
                inputs, outputs = inputs.cuda(), outputs.cuda()
                with torch.no_grad():
                    pre = net(inputs)
                    pres.append(pre)
                    output_list.append(outputs)
            pres = torch.cat(pres, dim=0).detach().cpu().numpy() * 50
            output_list = torch.cat(output_list, dim=0).detach().cpu().numpy() * 50

            val_loss2 = np.mean(np.abs(pres - output_list)[val_index, :, :, :])
            if val_loss2 < args.best_record['loss2']:
                save_model2(args, epoch, val_loss2, net)

            train_inputs, train_outputs, _ = pod_cylinder.prepare_data(train_index, pres)
            val_inputs, val_outputs, _ = pod_cylinder.prepare_data(val_index, pres)
            model = ridge_regression(train_inputs, train_outputs, val_inputs, val_outputs)
            test_inputs, test_outputs, test_data = pod_cylinder.prepare_data(val_index, pres)
            test_pres = model.predict(test_inputs)
            test_maps = pod_cylinder.inverse_transform(coff=test_pres)

            val_loss = np.mean(np.abs(test_maps - output_list[val_index, :, :, :]))

            # Record results by tensorboard
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
    args.snapshot = '/mnt/jfs/zhaoxiaoyu/Gappy_POD/heat2D/logs/ckpt/voronoicnn_heatlayout2/best_epoch_29_loss_0.01949539.pth'

    # Define data loader
    test_dataset = HeatLayout(index=index)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)
    n_components = 15
    positions = np.array([[int(i*2), int(j*2)] for i in range(100) for j in range(100)])
    pod_index = [i for i in range(500)]
    train_index = [i for i in range(0, 2000)]
    val_index = [i for i in range(2000, 3000)]
    pod_cylinder = PODHeat(n_components, positions, pod_index)

    # Load trained network
    net = UNet(in_channels=1, out_channels=1).cuda()
    net.load_state_dict(torch.load(args.snapshot)['state_dict'])
    print('load models: ' + args.snapshot)

    # Test procedure
    net.eval()
    test_mae, test_rmse, test_num = 0.0, 0.0, 0.0
    test_max_ae = 0
    pres, output_list = [], []
    for i, (inputs, outputs) in enumerate(test_loader):
        N, _, _, _ = inputs.shape
        inputs, outputs = inputs.cuda(), outputs.cuda()
        with torch.no_grad():
            pre = net(inputs)

            # coff = pod_cylinder.pca.transform(pre.cpu().numpy().reshape(outputs.shape[0], -1) * 50)
            # pre2 = pod_cylinder.pca.inverse_transform(coff)
            # pre = torch.from_numpy(pre2).reshape(pre2.shape[0], 1, 200, 200)

            pres.append(pre)
            output_list.append(outputs)
    pres = torch.cat(pres, dim=0).detach().cpu().numpy() * 50
    output_list = torch.cat(output_list, dim=0).detach().cpu().numpy() * 50
    train_inputs, train_outputs, _ = pod_cylinder.prepare_data(train_index, pres, flag=False)
    # aug_inputs, aug_outputs = pod_cylinder.sample(1000)
    # train_inputs = np.concatenate([train_inputs, aug_inputs], axis=0)
    # train_outputs = np.concatenate([train_outputs, aug_outputs], axis=0)

    val_inputs, val_outputs, _ = pod_cylinder.prepare_data(val_index, pres)
    model = ridge_regression2(train_inputs, train_outputs, val_inputs, val_outputs)
    test_inputs, test_outputs, test_data = pod_cylinder.prepare_data(val_index, pres)
    test_pres = model.predict(test_inputs / 50.0)
    pre = pod_cylinder.inverse_transform(coff=test_pres)
    test_num += pre.shape[0]
    pre = torch.from_numpy(pre)
    output_list = torch.from_numpy(output_list[val_index])

    pre = torch.from_numpy(pres[val_index, :, :, :])

    test_mae += F.l1_loss(output_list, pre).item() * test_num
    test_rmse += torch.sum(cre(output_list, pre, 2))
    test_max_ae += torch.sum(torch.max(torch.abs(output_list - pre).flatten(1), dim=1)[0]).item()
    print('test mae:', test_mae / test_num)
    print('test rmse:', test_rmse / test_num)
    print('test max_ae:', test_max_ae / test_num)

    # pre_data, u_data = torch.cat(pre_data, dim=0).cpu().numpy(), torch.cat(u_data, dim=0).cpu().numpy()
    #
    # f = h5py.File('heat_test.h5', 'w')
    # f['pre'] = pre_data
    # f['u'] = u_data
    # f.close()

    # plot3x1(outputs[-1, 0, :, :].cpu().numpy(), pre[-1, 0, :, :].cpu().numpy(), './test.png')


if __name__ == '__main__':
    # train()
    test(index=[i for i in range(0, 3000)])
    # test(index=[i for i in range(4000, 5000)])
    # test(index=[i for i in range(1800, 1940)])
