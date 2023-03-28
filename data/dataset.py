# -*- coding: utf-8 -*-
# @Time    : 2022/4/20 15:19
# @Author  : zhaoxiaoyu
# @File    : dataset.py
import pickle
import h5py
import torch
import numpy as np
from tqdm import tqdm
from scipy.interpolate import griddata
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
import torchvision.transforms as transforms
import scipy.io as sio
import os


def awgn(s, SNRdB, L=1):
    """
    AWGN channel
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal
    's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power
    spectral density N0 of noise added
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        L : oversampling factor (applicable for waveform simulation) default L = 1.
    Returns:
        r : received signal vector (r=s+n)
    """
    gamma = 10 ** (SNRdB / 10)  # SNR to linear scale
    if s.ndim == 1:  # if s is single dimensional vector
        P = L * sum(abs(s) ** 2) / len(s)  # Actual power in the vector
    else:  # multi-dimensional signals like MFSK
        P = L * sum(sum(abs(s) ** 2)) / len(s)  # if s is a matrix [MxN]
    N0 = P / gamma  # Find the noise spectral density
    if np.isrealobj(s):  # check if input is real/complex object type
        n = np.sqrt(N0 / 2) * np.random.standard_normal(s.shape)  # computed noise
    else:
        n = np.sqrt(N0 / 2) * (np.random.standard_normal(s.shape) + 1j * np.random.standard_normal(s.shape))
    r = s + n  # received signal
    return r


class CylinderDataset(Dataset):
    def __init__(self, index, mean=0, std=1):
        """
        圆柱绕流数据集
        :param index:
        """
        super(CylinderDataset, self).__init__()
        self.mean, self.std = mean, std
        df = open('/home/ubuntu/zhaoxiaoyu/data/cylinder/Cy_Taira.pickle', 'rb')
        self.data = torch.from_numpy(pickle.load(df)).float().permute(0, 3, 1, 2)[index, :, :, :]
        self.data = (self.data - mean) / std
        df.close()

        positions = np.array([[50, 25], [62, 30], [50, 35], [62, 40], [50, 45], [62, 50], [50, 55], [62, 60]])

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        self.observe = np.concatenate(sparse_data, axis=-1)

    def __getitem__(self, index):
        return self.observe[index, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class CylinderInterpolDataset(Dataset):
    def __init__(self, index, mean=0, std=1):
        """
        圆柱绕流数据集：输入进行近邻插值转换为泰森多边形形式
        :param index:
        """
        super(CylinderInterpolDataset, self).__init__()

        self.mean, self.std = mean, std
        df = open('/home/ubuntu/zhaoxiaoyu/data/cylinder/Cy_Taira.pickle', 'rb')
        self.data = pickle.load(df)[index, :, :, :].transpose(0, 3, 1, 2)
        self.data = (self.data - mean) / std
        df.close()
        positions = np.array(
            [[56, 37], [50, 77], [62, 50], [47, 60]]
        )

        _, _, h, w = self.data.shape
        x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)
        sparse_locations_ex = np.zeros_like(positions)
        for i in range(positions.shape[0]):
            sparse_locations_ex[i, 0] = x_coor[positions[i, 0], positions[i, 1]]
            sparse_locations_ex[i, 1] = y_coor[positions[i, 0], positions[i, 1]]

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        sparse_data = np.concatenate(sparse_data, axis=-1)

        sparse_datas = []
        print("processing data...")
        for i in tqdm(range(sparse_data.shape[0])):
            input_1 = griddata(sparse_locations_ex, sparse_data[i], (x_coor, y_coor), method='nearest')
            sparse_datas.append(np.expand_dims(input_1, axis=0))
        sparse_datas = np.concatenate(sparse_datas, axis=0)
        mask = np.zeros_like(sparse_datas[0, :, :])
        for i in range(positions.shape[0]):
            mask[positions[i, 0], positions[i, 1]] = 1

        self.data = torch.from_numpy(self.data).float()
        self.observe = torch.from_numpy(sparse_datas).float().unsqueeze(dim=1)
        self.mask = torch.from_numpy(mask).float().unsqueeze(dim=0)

    def __getitem__(self, index):
        return torch.cat([self.observe[index, :, :, :], self.mask], dim=0), self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class CylinderInterpolGappyDataset(Dataset):
    def __init__(self, index, mean=0, std=1):
        """
        圆柱绕流数据集：输入进行近邻插值转换为泰森多边形形式
        :param index:
        """
        super(CylinderInterpolGappyDataset, self).__init__()
        self.mean, self.std = mean, std
        df = open('/home/ubuntu/zhaoxiaoyu/data/cylinder/Cy_Taira.pickle', 'rb')
        self.data = pickle.load(df)[index, :, :, :].transpose(0, 3, 1, 2)
        self.data = (self.data - mean) / std
        df.close()
        positions = np.array(
            [[50, 25], [62, 30], [50, 35], [62, 40], [50, 45], [62, 50], [50, 55], [62, 60]]
        )
        _, _, h, w = self.data.shape
        x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)
        sparse_locations_ex = np.zeros_like(positions)
        for i in range(positions.shape[0]):
            sparse_locations_ex[i, 0] = x_coor[positions[i, 0], positions[i, 1]]
            sparse_locations_ex[i, 1] = y_coor[positions[i, 0], positions[i, 1]]

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        sparse_data = np.concatenate(sparse_data, axis=-1)

        sparse_datas = []
        print("processing data...")
        for i in tqdm(range(sparse_data.shape[0])):
            input_1 = griddata(sparse_locations_ex, sparse_data[i], (x_coor, y_coor), method='nearest')
            sparse_datas.append(np.expand_dims(input_1, axis=0))
        sparse_datas = np.concatenate(sparse_datas, axis=0)
        mask = np.zeros_like(sparse_datas[0, :, :])
        for i in range(positions.shape[0]):
            mask[positions[i, 0], positions[i, 1]] = 1

        self.data = torch.from_numpy(self.data).float()
        self.observe = torch.from_numpy(sparse_datas).float().unsqueeze(dim=1)
        self.mask = torch.from_numpy(mask).float().unsqueeze(dim=0)
        self.coor = torch.cat([torch.from_numpy(x_coor).unsqueeze(dim=0) / w,
                               torch.from_numpy(y_coor).unsqueeze(dim=0) / h], dim=0).float()
        self.gappy = torch.from_numpy(sparse_data).float()

    def __getitem__(self, index):
        return torch.cat([self.observe[index, :, :, :], self.mask], dim=0), self.data[index, :], self.gappy[index, :]

    def __len__(self):
        return self.data.shape[0]


class CylinderPodDataset(Dataset):
    def __init__(self, pod_index, index, n_components=20, mean=0, std=1):
        """
        圆柱绕流数据集：对输出进行POD，并返回POD系数
        :param pod_index: 进行POD的数据索引
        :param index: 数据索引
        :param n_components: POD模态数量
        """
        super(CylinderPodDataset, self).__init__()
        self.mean, self.std = mean, std
        df = open('/home/ubuntu/zhaoxiaoyu/data/cylinder/Cy_Taira.pickle', 'rb')
        self.pca_data = torch.from_numpy(pickle.load(df)).float().permute(0, 3, 1, 2)[pod_index, :, :, :]
        self.pca_data = (self.pca_data - mean) / std
        df.close()
        df = open('/home/ubuntu/zhaoxiaoyu/data/cylinder/Cy_Taira.pickle', 'rb')
        self.data = torch.from_numpy(pickle.load(df)).float().permute(0, 3, 1, 2)[index, :, :, :]
        self.data = (self.data - mean) / std
        df.close()

        self.pca = PCA(n_components=n_components)
        X_t = self.pca.fit_transform(self.pca_data.reshape(len(pod_index), -1).numpy())
        self.max, self.min = np.max(X_t, axis=0).reshape(1, -1), np.min(X_t, axis=0).reshape(1, -1)

        # 计算POD系数并归一化
        self.coff = self.pca.transform(self.data.reshape(len(index), -1).numpy())
        self.coff = (self.coff - self.min) / (self.max - self.min)

        positions = np.array([[50, 25], [62, 30], [50, 35], [62, 40], [50, 45], [62, 50], [50, 55], [62, 60]])

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        self.observe = np.concatenate(sparse_data, axis=-1)
        self.size = self.data.shape[-3:]

    def __getitem__(self, index):
        return self.observe[index, :], self.coff[index, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]

    def inverse_transform(self, coff):
        inverse_coff = coff.cpu().numpy() * (self.max - self.min) + self.min
        return torch.from_numpy(self.pca.inverse_transform(inverse_coff)).to(coff.device) \
            .float().reshape(coff.shape[0], self.size[0], self.size[1], self.size[2])


class AirfoilDataset(Dataset):
    def __init__(self, index, type='vx', expand=1):
        """
        不可压缩流机翼数据集
        :param index:
        """
        super(AirfoilDataset, self).__init__()
        index = np.array(index).reshape(1, -1).repeat(expand, axis=0).reshape(-1).tolist()

        data = sio.loadmat('/mnt/jfs/zhaoxiaoyu/Gappy_POD/airfoil_incompressible.mat')[type][:500, :, :, :]
        self.max = np.max(data, axis=0, keepdims=True)
        self.min = np.min(data, axis=0, keepdims=True)
        self.airfoil_mask = (self.max[0, 0, :, :] == 0)

        data = sio.loadmat('/mnt/jfs/zhaoxiaoyu/Gappy_POD/airfoil_incompressible.mat')[type]
        self.data = (data[index, :, :, :] - self.min) / (self.max - self.min + 1e-8)
        positions = np.array(
            [[65, 125], [90, 141], [191, 126], [90, 114]]
        )

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        self.observe = torch.from_numpy(np.concatenate(sparse_data, axis=-1)).float()
        self.data = torch.from_numpy(self.data).float()

    def __getitem__(self, index):
        return self.observe[index, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class AirfoilPodDataset(Dataset):
    def __init__(self, pod_index, index, positions, n_components=20, type='vx', expand=1):
        """
        不可压缩流机翼数据集：对输出进行POD，并返回POD系数
        :param pod_index: 进行POD的数据索引
        :param index: 数据索引
        :param n_components: POD模态数量
        """
        super(AirfoilPodDataset, self).__init__()
        index = np.array(index).reshape(1, -1).repeat(expand, axis=0).reshape(-1).tolist()

        self.data = sio.loadmat('/mnt/jfs/zhaoxiaoyu/Gappy_POD/airfoil_incompressible.mat')[type]

        self.pca_data = torch.from_numpy(self.data[pod_index, :, :, :]).float()
        self.data = torch.from_numpy(self.data[index, :, :, :]).float()

        self.pca = PCA(n_components=n_components)
        X_t = self.pca.fit_transform(self.pca_data.reshape(len(pod_index), -1).numpy())
        self.max, self.min = np.max(X_t, axis=0).reshape(1, -1), np.min(X_t, axis=0).reshape(1, -1)

        # 计算POD系数并归一化
        self.coff = self.pca.transform(self.data.reshape(len(index), -1).numpy())
        self.coff = (self.coff - self.min) / (self.max - self.min)

        self.positions = positions

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        self.observe = np.concatenate(sparse_data, axis=-1)
        self.size = self.data.shape[-3:]

    def __getitem__(self, index):
        return self.observe[index, :], self.coff[index, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]

    def inverse_transform(self, coff):
        inverse_coff = coff.cpu().numpy() * (self.max - self.min) + self.min
        return torch.from_numpy(self.pca.inverse_transform(inverse_coff)).to(coff.device) \
            .float().reshape(coff.shape[0], self.size[0], self.size[1], self.size[2])


class AirfoilInterpolDataset(Dataset):
    def __init__(self, index, type='vx', expand=1):
        """
        不可压缩流机翼数据集：输入进行近邻插值转换为泰森多边形形式
        :param index:
        """
        super(AirfoilInterpolDataset, self).__init__()
        index = np.array(index).reshape(1, -1).repeat(expand, axis=0).reshape(-1).tolist()

        data = sio.loadmat('/mnt/jfs/zhaoxiaoyu/Gappy_POD/airfoil_incompressible.mat')[type][:500, :, :, :]
        self.max = np.max(data, axis=0, keepdims=True)
        self.min = np.min(data, axis=0, keepdims=True)
        self.airfoil_mask = (self.max[0, 0, :, :] == 0)

        data = sio.loadmat('/mnt/jfs/zhaoxiaoyu/Gappy_POD/airfoil_incompressible.mat')[type]
        self.data = (data[index, :, :, :] - self.min) / (self.max - self.min + 1e-8)
        positions = np.array(
            [[65, 126], [90, 139], [65, 151], [90, 114], [90, 164], [115, 139], [65, 101], [115, 164]]
        )

        _, _, h, w = self.data.shape
        x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)
        sparse_locations_ex = np.zeros_like(positions)
        for i in range(positions.shape[0]):
            sparse_locations_ex[i, 0] = x_coor[positions[i, 0], positions[i, 1]]
            sparse_locations_ex[i, 1] = y_coor[positions[i, 0], positions[i, 1]]

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        sparse_data = np.concatenate(sparse_data, axis=-1)

        sparse_datas = []
        print("processing data...")
        for i in tqdm(range(sparse_data.shape[0])):
            input_1 = griddata(sparse_locations_ex, sparse_data[i], (x_coor, y_coor), method='nearest')
            sparse_datas.append(np.expand_dims(input_1, axis=0))
        sparse_datas = np.concatenate(sparse_datas, axis=0)
        mask = np.zeros_like(sparse_datas[0, :, :])
        for i in range(positions.shape[0]):
            mask[positions[i, 0], positions[i, 1]] = 1

        self.data = torch.from_numpy(self.data).float()
        self.observe = torch.from_numpy(sparse_datas).float().unsqueeze(dim=1)
        self.mask = torch.from_numpy(mask).float().unsqueeze(dim=0)

    def __getitem__(self, index):
        return torch.cat([self.observe[index, :, :, :], self.mask], dim=0), self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class AirfoilInterpolGappyDataset(Dataset):
    def __init__(self, index, type='vx', expand=1):
        """
        不可压缩流机翼数据集：输入进行近邻插值转换为泰森多边形形式
        :param index:
        """
        super(AirfoilInterpolGappyDataset, self).__init__()

        index = np.array(index).reshape(1, -1).repeat(expand, axis=0).reshape(-1).tolist()

        data = sio.loadmat('/mnt/jfs/zhaoxiaoyu/Gappy_POD/airfoil_incompressible.mat')[type][:500, :, :, :]
        self.max = np.max(data, axis=0, keepdims=True)
        self.min = np.min(data, axis=0, keepdims=True)
        self.airfoil_mask = (self.max[0, 0, :, :] == 0)

        data = sio.loadmat('/mnt/jfs/zhaoxiaoyu/Gappy_POD/airfoil_incompressible.mat')[type]
        self.data = (data[index, :, :, :] - self.min) / (self.max - self.min + 1e-8)
        positions = np.array(
            [[65, 125], [90, 141], [191, 126], [90, 114], [65, 150], [90, 166], [115, 141], [65, 100]]
        )

        _, _, h, w = self.data.shape
        x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)
        sparse_locations_ex = np.zeros_like(positions)
        for i in range(positions.shape[0]):
            sparse_locations_ex[i, 0] = x_coor[positions[i, 0], positions[i, 1]]
            sparse_locations_ex[i, 1] = y_coor[positions[i, 0], positions[i, 1]]

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        sparse_data = np.concatenate(sparse_data, axis=-1)

        sparse_datas = []
        print("processing data...")
        for i in tqdm(range(sparse_data.shape[0])):
            input_1 = griddata(sparse_locations_ex, sparse_data[i], (x_coor, y_coor), method='nearest')
            sparse_datas.append(np.expand_dims(input_1, axis=0))
        sparse_datas = np.concatenate(sparse_datas, axis=0)
        mask = np.zeros_like(sparse_datas[0, :, :])
        for i in range(positions.shape[0]):
            mask[positions[i, 0], positions[i, 1]] = 1

        self.data = torch.from_numpy(self.data).float()
        self.observe = torch.from_numpy(sparse_datas).float().unsqueeze(dim=1)
        self.mask = torch.from_numpy(mask).float().unsqueeze(dim=0)
        self.gappy = torch.from_numpy(sparse_data).float()

    def __getitem__(self, index):
        return torch.cat([self.observe[index, :, :, :], self.mask], dim=0), self.data[index, :], self.gappy[index, :]

    def __len__(self):
        return self.data.shape[0]


class HeatDataset(Dataset):
    def __init__(self, index, mean=308, std=50, expand=1):
        """
        热布局数据集
        :param index:
        """
        super(HeatDataset, self).__init__()
        if expand > 1:
            index = np.array(index).reshape(1, -1).repeat(expand, axis=0).reshape(-1).tolist()
        self.mean, self.std = mean, std
        f = h5py.File('/mnt/jfs/zhaoxiaoyu/data/heat/temperature.h5', 'r')
        self.data = torch.from_numpy(f['u'][:, :, :, :]).float()[index, :, :, :]
        self.data = (self.data - mean) / std
        f.close()

        positions = np.array(
            [[40, 40], [40, 80], [40, 120], [40, 160], [80, 40], [80, 80], [80, 120], [80, 160], [120, 40], [120, 80],
             [120, 120], [120, 160], [160, 40], [160, 80], [160, 120], [160, 160]]
        )
        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        self.observe = np.concatenate(sparse_data, axis=-1)

    def __getitem__(self, index):
        return self.observe[index, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class HeatInterpolDataset(Dataset):
    def __init__(self, index, mean=308, std=50, expand=1):
        """
        热布局数据集：输入进行近邻插值转换为泰森多边形形式
        :param index:
        """
        super(HeatInterpolDataset, self).__init__()
        index = np.array(index).reshape(1, -1).repeat(expand, axis=0).reshape(-1).tolist()
        self.mean, self.std = mean, std
        f = h5py.File('/mnt/jfs/zhaoxiaoyu/data/heat/temperature.h5', 'r')
        self.data = f['u'][:][index, :, :, :]
        self.data = (self.data - mean) / std
        f.close()
        positions = np.array(
            [[40, 40], [40, 80], [40, 120], [40, 160], [80, 40], [80, 80], [80, 120], [80, 160], [120, 40], [120, 80],
             [120, 120], [120, 160], [160, 40], [160, 80], [160, 120], [160, 160]]
        )

        _, _, h, w = self.data.shape
        x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)
        sparse_locations_ex = np.zeros_like(positions)
        for i in range(positions.shape[0]):
            sparse_locations_ex[i, 0] = x_coor[positions[i, 0], positions[i, 1]]
            sparse_locations_ex[i, 1] = y_coor[positions[i, 0], positions[i, 1]]

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        sparse_data = np.concatenate(sparse_data, axis=-1)

        sparse_datas = []
        print("processing data...")
        for i in tqdm(range(sparse_data.shape[0])):
            input_1 = griddata(sparse_locations_ex, sparse_data[i], (x_coor, y_coor), method='nearest')
            sparse_datas.append(np.expand_dims(input_1, axis=0))
        sparse_datas = np.concatenate(sparse_datas, axis=0)
        mask = np.zeros_like(sparse_datas[0, :, :])
        for i in range(positions.shape[0]):
            mask[positions[i, 0], positions[i, 1]] = 1

        self.data = torch.from_numpy(self.data).float()
        self.observe = torch.from_numpy(sparse_datas).float().unsqueeze(dim=1)
        self.mask = torch.from_numpy(mask).float().unsqueeze(dim=0)

    def __getitem__(self, index):
        return torch.cat([self.observe[index, :, :, :], self.mask], dim=0), self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class HeatInterpolGappyDataset(Dataset):
    def __init__(self, index, mean=308, std=50):
        """
        热布局数据集：输入进行近邻插值转换为泰森多边形形式
        :param index:
        """
        super(HeatInterpolGappyDataset, self).__init__()
        self.mean, self.std = mean, std
        f = h5py.File('/mnt/jfs/zhaoxiaoyu/data/heat/temperature.h5', 'r')
        self.data = f['u'][index, :, :, :]
        self.data = (self.data - mean) / std
        f.close()
        positions = np.array(
            [[40, 40], [40, 80], [40, 120], [40, 160], [80, 40], [80, 80], [80, 120], [80, 160], [120, 40], [120, 80],
             [120, 120], [120, 160], [160, 40], [160, 80], [160, 120], [160, 160]]
        )

        _, _, h, w = self.data.shape
        x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)
        sparse_locations_ex = np.zeros_like(positions)
        for i in range(positions.shape[0]):
            sparse_locations_ex[i, 0] = x_coor[positions[i, 0], positions[i, 1]]
            sparse_locations_ex[i, 1] = y_coor[positions[i, 0], positions[i, 1]]

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        sparse_data = np.concatenate(sparse_data, axis=-1)

        sparse_datas = []
        print("processing data...")
        for i in tqdm(range(sparse_data.shape[0])):
            input_1 = griddata(sparse_locations_ex, sparse_data[i], (x_coor, y_coor), method='nearest')
            sparse_datas.append(np.expand_dims(input_1, axis=0))
        sparse_datas = np.concatenate(sparse_datas, axis=0)
        mask = np.zeros_like(sparse_datas[0, :, :])
        for i in range(positions.shape[0]):
            mask[positions[i, 0], positions[i, 1]] = 1

        self.data = torch.from_numpy(self.data).float()
        self.observe = torch.from_numpy(sparse_datas).float().unsqueeze(dim=1)
        self.mask = torch.from_numpy(mask).float().unsqueeze(dim=0)
        self.gappy = torch.from_numpy(sparse_data).float()

    def __getitem__(self, index):
        return torch.cat([self.observe[index, :, :, :], self.mask], dim=0), \
               self.data[index, :], self.gappy[index, :]

    def __len__(self):
        return self.data.shape[0]


class HeatObserveDataset(Dataset):
    def __init__(self, index, mean=308, std=50):
        """
        热布局数据集：输入采用掩码表示
        :param index:
        """
        super(HeatObserveDataset, self).__init__()
        self.mean, self.std = mean, std
        f = h5py.File('/home/ubuntu/zhaoxiaoyu/data/heat/temperature.h5', 'r')
        self.data = f['u'][index, :, :, :]
        self.data = (self.data - mean) / std
        f.close()
        positions = np.array(
            [[28, 28], [28, 56], [28, 84], [28, 112], [28, 140], [28, 168], [56, 28], [56, 56], [56, 84], [56, 112],
             [56, 140], [56, 168], [84, 28], [84, 56], [84, 84], [84, 112], [84, 140], [84, 168], [112, 28], [112, 56],
             [112, 84], [112, 112], [112, 140], [112, 168], [140, 28], [140, 56], [140, 84], [140, 112], [140, 140],
             [140, 168], [168, 28], [168, 56], [168, 84], [168, 112], [168, 140], [168, 168]]
        )

        _, _, h, w = self.data.shape
        x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)

        sparse_data = np.zeros_like(self.data)
        for i in range(positions.shape[0]):
            sparse_data[:, 0, positions[i, 0], positions[i, 1]] = self.data[:, 0, positions[i, 0], positions[i, 1]]

        self.data = torch.from_numpy(self.data).float()
        self.observe = torch.from_numpy(sparse_data).float()
        self.coor = torch.cat([torch.from_numpy(x_coor).unsqueeze(dim=0) / w,
                               torch.from_numpy(y_coor).unsqueeze(dim=0) / h], dim=0).float()

    def __getitem__(self, index):
        return torch.cat([self.observe[index, :, :, :], self.coor], dim=0), self.data[index, :]
        # return self.observe[index, :, :, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class HeatPodDataset(Dataset):
    def __init__(self, pod_index, index, n_components=25, mean=308, std=50):
        """
        热布局数据集：对输出进行POD，并返回POD系数
        :param pod_index: 进行POD的数据索引
        :param index: 数据索引
        :param n_components: POD模态数量
        """
        super(HeatPodDataset, self).__init__()
        self.mean, self.std = mean, std
        f = h5py.File('/mnt/jfs/zhaoxiaoyu/data/heat/temperature.h5', 'r')
        self.data = f['u'][:, :, :, :]
        self.data = (self.data - mean) / std
        f.close()

        self.pca_data = torch.from_numpy(self.data[pod_index, :, :, :]).float()
        self.data = torch.from_numpy(self.data[index, :, :, :]).float()

        self.pca = PCA(n_components=n_components)
        X_t = self.pca.fit_transform(self.pca_data.reshape(len(pod_index), -1).numpy())
        self.max, self.min = np.max(X_t, axis=0).reshape(1, -1), np.min(X_t, axis=0).reshape(1, -1)

        # 计算POD系数并归一化
        self.coff = self.pca.transform(self.data.reshape(len(index), -1).numpy())
        self.coff = (self.coff - self.min) / (self.max - self.min)

        positions = np.array(
            [[40, 40], [40, 80], [40, 120], [40, 160], [80, 40], [80, 80], [80, 120], [80, 160], [120, 40],
             [120, 80],
             [120, 120], [120, 160], [160, 40], [160, 80], [160, 120], [160, 160]]
        )

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        self.observe = np.concatenate(sparse_data, axis=-1)
        self.size = self.data.shape[-3:]

    def __getitem__(self, index):
        return self.observe[index, :], self.coff[index, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]

    def inverse_transform(self, coff):
        inverse_coff = coff.cpu().numpy() * (self.max - self.min) + self.min
        return torch.from_numpy(self.pca.inverse_transform(inverse_coff)).to(coff.device) \
            .float().reshape(coff.shape[0], self.size[0], self.size[1], self.size[2])


class SubsonicAirfoilDataset(Dataset):
    def __init__(self, type='p', data_path='/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_train.h5',
                 positions=[7703, 6394, 5738, 5129]):
        """
        NACA0012数据集
        :param index:
        """
        super(SubsonicAirfoilDataset, self).__init__()
        f = h5py.File('/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_train.h5', 'r')
        self.min = np.min(f[type][:], axis=0, keepdims=True)
        self.max = np.max(f[type][:], axis=0, keepdims=True)
        f.close()

        f = h5py.File(data_path, 'r')
        self.data = f[type][:]
        self.data = (self.data - self.min) / (self.max - self.min + 1e-10)
        self.x_coor = f['x_coor'][:]
        self.y_coor = f['y_coor'][:]
        f.close()
        self.data = torch.from_numpy(self.data).float()

        positions = np.array(positions)
        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, positions[i]].reshape(-1, 1))
        self.observe = torch.cat(sparse_data, axis=-1)
        self.pos = np.concatenate([self.x_coor.reshape(-1, 1), self.y_coor.reshape(-1, 1)], axis=-1)
        self.pos = torch.from_numpy(self.pos).float().cuda()

    def __getitem__(self, index):
        return self.observe[index, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class SubsonicAirfoilInterpolDataset(Dataset):
    def __init__(self, type='vx', data_path='/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_train.h5',
                 positions=[7703, 6394, 5738, 5129]):
        """
        不可压缩流机翼数据集：输入进行近邻插值转换为泰森多边形形式
        :param index:
        """
        super(SubsonicAirfoilInterpolDataset, self).__init__()
        f = h5py.File('/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_train.h5', 'r')
        self.data = f[type][:]
        self.min = np.min(self.data, axis=0, keepdims=True)
        self.max = np.max(self.data, axis=0, keepdims=True)
        self.x_coor = f['x_coor'][:]
        self.y_coor = f['y_coor'][:]
        f.close()

        f = h5py.File(data_path, 'r')
        self.data = f[type][:]
        self.data = (self.data - self.min) / (self.max - self.min + 1e-10)
        self.x_coor = f['x_coor'][:]
        self.y_coor = f['y_coor'][:]
        f.close()
        self.data = torch.from_numpy(self.data).float()
        positions = np.array(positions)
        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, positions[i]].reshape(-1, 1))
        sparse_data = np.concatenate(sparse_data, axis=-1)
        sparse_locations_ex = np.zeros((len(positions), 2))
        for i in range(positions.shape[0]):
            sparse_locations_ex[i, 0] = self.x_coor[positions[i]]
            sparse_locations_ex[i, 1] = self.y_coor[positions[i]]

        sorted_index = sio.loadmat('/mnt/jfs/zhaoxiaoyu/Gappy_POD/index2.mat')['data']
        self.data = self.data[:, sorted_index].reshape(-1, 1, 64, 64)

        _, _, h, w = self.data.shape
        x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)

        sparse_datas = []
        print("processing data...")
        for i in tqdm(range(sparse_data.shape[0])):
            input_1 = griddata(sparse_locations_ex, sparse_data[i], (x_coor, y_coor), method='nearest')
            sparse_datas.append(np.expand_dims(input_1, axis=0))
        sparse_datas = np.concatenate(sparse_datas, axis=0)

        self.observe = torch.from_numpy(sparse_datas).float().unsqueeze(dim=1)
        self.coor = torch.from_numpy(
            np.concatenate([x_coor.reshape(1, 64, 64) / w, y_coor.reshape(1, 64, 64) / h], axis=0)).float()

    def __getitem__(self, index):
        return torch.cat([self.observe[index, :, :, :], self.coor], dim=0), self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class SubsonicAirfoilInterpolGappyDataset(Dataset):
    def __init__(self, type='vx', data_path='/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_train.h5',
                 positions=[7703, 6394, 5738, 5129]):
        """
        不可压缩流机翼数据集：输入进行近邻插值转换为泰森多边形形式
        :param index:
        """
        super(SubsonicAirfoilInterpolGappyDataset, self).__init__()
        f = h5py.File('/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_train.h5', 'r')
        self.data = f[type][:]
        self.min = np.min(self.data, axis=0, keepdims=True)
        self.max = np.max(self.data, axis=0, keepdims=True)
        self.x_coor = f['x_coor'][:]
        self.y_coor = f['y_coor'][:]
        f.close()

        f = h5py.File(data_path, 'r')
        self.data = f[type][:][:1, :]
        self.data = (self.data - self.min) / (self.max - self.min + 1e-10)
        self.x_coor = f['x_coor'][:]
        self.y_coor = f['y_coor'][:]
        f.close()
        self.labels = torch.from_numpy(self.data).float()
        positions = np.array(positions)
        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.labels[:, positions[i]].reshape(-1, 1))
        sparse_data = np.concatenate(sparse_data, axis=-1)
        sparse_locations_ex = np.zeros((len(positions), 2))
        for i in range(positions.shape[0]):
            sparse_locations_ex[i, 0] = self.x_coor[positions[i]]
            sparse_locations_ex[i, 1] = self.y_coor[positions[i]]

        sorted_index = sio.loadmat('/mnt/jfs/zhaoxiaoyu/Gappy_POD/index2.mat')['data']
        # self.min = self.min[:, sorted_index].reshape(1, 1, 96, 96)
        # self.max = self.max[:, sorted_index].reshape(1, 1, 96, 96)
        self.data = self.labels[:, sorted_index].reshape(-1, 1, 64, 64)

        _, _, h, w = self.data.shape
        x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)

        sparse_datas = []
        print("processing data...")
        for i in tqdm(range(sparse_data.shape[0])):
            input_1 = griddata(sparse_locations_ex, sparse_data[i], (x_coor, y_coor), method='nearest')
            sparse_datas.append(np.expand_dims(input_1, axis=0))
        sparse_datas = np.concatenate(sparse_datas, axis=0)

        self.observe = torch.from_numpy(sparse_datas).float().unsqueeze(dim=1)
        self.gappy = torch.from_numpy(sparse_data).float()
        self.coor = torch.from_numpy(
            np.concatenate([x_coor.reshape(1, 64, 64) / w, y_coor.reshape(1, 64, 64) / h], axis=0)).float()

    def __getitem__(self, index):
        return torch.cat([self.observe[index, :, :, :], self.coor], dim=0), \
               self.data[index, :], self.gappy[index, :], self.labels[index, :]

    def __len__(self):
        return self.data.shape[0]


class SubsonicAirfoilPodDataset(Dataset):
    def __init__(self, positions, n_components=20, type='vx',
                 data_path='/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_train.h5', norm=True):
        """
        压缩流机翼数据集：对输出进行POD，并返回POD系数
        :param pod_index: 进行POD的数据索引
        :param index: 数据索引
        :param n_components: POD模态数量
        """
        super(SubsonicAirfoilPodDataset, self).__init__()
        f = h5py.File('/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_train.h5', 'r')
        self.pca_data = torch.from_numpy(f[type][:]).float()
        self.x_coor = f['x_coor'][:]
        self.y_coor = f['y_coor'][:]
        self.min = np.min(f[type][:], axis=0, keepdims=True)
        self.max = np.max(f[type][:], axis=0, keepdims=True)
        f.close()

        f = h5py.File(data_path, 'r')
        self.data = torch.from_numpy(f[type][:][:1, :]).float()
        if norm:
            self.data_norm = torch.from_numpy((f[type][:] - self.min) / (self.max - self.min + 1e-10))
            self.data_min, self.data_max = self.min, self.max
        else:
            self.data_norm = self.data
        f.close()

        self.pca = PCA(n_components=n_components)
        X_t = self.pca.fit_transform(self.pca_data.reshape(self.pca_data.shape[0], -1).numpy())
        self.max, self.min = np.max(X_t, axis=0).reshape(1, -1), np.min(X_t, axis=0).reshape(1, -1)

        # 计算POD系数并归一化
        self.coff = self.pca.transform(self.data.reshape(self.data.shape[0], -1).numpy())
        self.coff = (self.coff - self.min) / (self.max - self.min)

        self.positions = positions

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data_norm[:, positions[i]].reshape(-1, 1))
        self.observe = torch.from_numpy(np.concatenate(sparse_data, axis=-1)).float()
        self.size = self.data.shape[-3:]
        self.coff = torch.from_numpy(self.coff).float()

    def __getitem__(self, index):
        return self.observe[index, :], self.coff[index, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]

    def inverse_transform(self, coff):
        inverse_coff = coff.cpu().numpy() * (self.max - self.min) + self.min
        return torch.from_numpy(self.pca.inverse_transform(inverse_coff)).to(coff.device) \
            .float().reshape(coff.shape[0], -1)


class HeatLayout(Dataset):
    def __init__(self, index, expand=1):
        """
        热布局数据集
        :param index:
        """
        super(HeatLayout, self).__init__()
        index = np.array(index).reshape(1, -1).repeat(expand, axis=0).reshape(-1).tolist()
        f = h5py.File('/mnt/jfs/zhaoxiaoyu/Gappy_POD/heat_layout.h5', 'r')
        self.layout = f['F'][:] / 10000.0
        self.data = (f['u'][:] - 298.0) / 50.0
        f.close()

        self.layout = torch.from_numpy(self.layout[index, :, :, :]).float()
        self.data = torch.from_numpy(self.data[index, :, :, :]).float()

    def __getitem__(self, index):
        return self.layout[index, :, :, :], self.data[index, :, :, :]

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import scicomap as sc

    ccmap = sc.ScicoDiverging('seismic')
    mpl_map = ccmap.get_mpl_color_map()

    dataset = iter(DataLoader(CylinderInterpolDataset(index=[0]), batch_size=1))
    inputs, outputs = next(dataset)
    plt.imshow(inputs[0, 0, :, :].numpy(), cmap=mpl_map)
    plt.axis('off')
    plt.savefig('voronoi.png', bbox_inches='tight', pad_inches=0, dpi=200)
