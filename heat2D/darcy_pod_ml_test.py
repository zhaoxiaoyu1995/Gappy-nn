# -*- coding: utf-8 -*-
# @Time    : 2022/10/8 2:08
# @Author  : zhaoxiaoyu
# @File    : heat2D_pod_ml.py
from sklearn.decomposition import PCA
import numpy as np
import h5py
import os
import sys
from tqdm import tqdm
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import torch
import pickle

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(filename)

from model.ml import svr_regression, rf_regression, gp_regression


def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res


class PODHeat:
    def __init__(self, n_components, positions, pod_index):
        self.positions = positions
        # Loading data
        f = h5py.File('/home/ubuntu/zhaoxiaoyu/data/darcy/darcy12000.h5', 'r')
        pca_data = torch.from_numpy(f['sol'][pod_index, ::2, ::2]).float().unsqueeze(dim=1).numpy() * 100
        f.close()
        self.n_components = n_components

        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        X_t = self.pca.fit_transform(pca_data.reshape(len(pod_index), -1))
        self.max_coeff, self.min_coeff = np.max(X_t, axis=0).reshape(1, -1), np.min(X_t, axis=0).reshape(1, -1)
        self.size = pca_data.shape[-3:]

        self.train_data_coff = (X_t - self.min_coeff) / (self.max_coeff - self.min_coeff + 1e-10)

    def prepare_data(self, data_index):
        # Loading data
        f = h5py.File('/home/ubuntu/zhaoxiaoyu/data/darcy/darcy12000.h5', 'r')
        data = torch.from_numpy(f['sol'][data_index, ::2, ::2]).float().unsqueeze(dim=1).numpy() * 100
        f.close()

        # Normalize the coefficients
        data_coff = self.pca.transform(data.reshape(len(data_index), -1))
        data_coff = (data_coff - self.min_coeff) / (self.max_coeff - self.min_coeff)

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        observe = np.concatenate(sparse_data, axis=-1)
        return observe, data_coff, data

    def inverse_transform(self, coff):
        inverse_coff = coff * (self.max_coeff - self.min_coeff) + self.min_coeff
        return self.pca.inverse_transform(inverse_coff).reshape(coff.shape[0], self.size[0], self.size[1], self.size[2])

    def sample(self, n=1000):
        data_coff = []
        for _ in tqdm(range(n)):
            while True:
                coff = np.random.rand(1, self.n_components)
                similar = get_cos_similar_matrix(coff, self.train_data_coff)
                if np.sum(similar > 0.900) > 1:
                    data_coff.append(coff)
                    break
        data_coff = np.concatenate(data_coff, axis=0)
        inverse_coff = data_coff * (self.max_coeff - self.min_coeff) + self.min_coeff
        observe = []
        for i in tqdm(range(int(n / 100.0))):
            data = self.pca.inverse_transform(inverse_coff[i * 100:i * 100 + 100]).reshape(100, self.size[0],
                                                                                           self.size[1],
                                                                                           self.size[2])
            sparse_data = []
            for i in range(self.positions.shape[0]):
                sparse_data.append(data[:, 0, self.positions[i, 0], :][:, self.positions[i, 1]].reshape(-1, 1))
            sparse_data = np.concatenate(sparse_data, axis=-1)
            observe.append(sparse_data)
        observe = np.concatenate(observe, axis=0)
        return observe, data_coff


# Gaussian Process Regression
def ridge_regression(train_inputs, train_outputs, val_inputs, val_outputs):
    X = np.concatenate([train_inputs, val_inputs], axis=0)
    y = np.concatenate([train_outputs, val_outputs], axis=0)
    ps = PredefinedSplit(
        test_fold=np.concatenate([-1 * np.ones((train_inputs.shape[0], 1)), np.zeros((val_inputs.shape[0], 1))],
                                 axis=0)
    )
    tuned_parameter = [{'alpha': [0.1, 1.0, 2.0, 5.0, 10.0]}]
    ridge = linear_model.Ridge()
    gpr_c = GridSearchCV(estimator=ridge, param_grid=tuned_parameter, cv=ps, n_jobs=4)
    gpr_c.fit(X, y)
    alpha = gpr_c.best_params_['alpha']
    print('The optimal parameters are: \n alpha:', alpha)

    ridge = linear_model.Ridge(alpha=alpha)
    ridge.fit(train_inputs, train_outputs)
    return ridge


if __name__ == '__main__':
    n_components = 400
    positions = np.array(
        [[18, 18], [18, 36], [18, 54], [18, 72], [18, 90], [18, 108], [36, 18], [36, 36], [36, 54], [36, 72],
         [36, 90], [36, 108], [54, 18], [54, 36], [54, 54], [54, 72], [54, 90], [54, 108], [72, 18], [72, 36],
         [72, 54], [72, 72], [72, 90], [72, 108], [90, 18], [90, 36], [90, 54], [90, 72], [90, 90], [90, 108],
         [108, 18], [108, 36], [108, 54], [108, 72], [108, 90], [108, 108]])
    positions = np.array([[int(128 * np.random.rand(1)), int(128 * np.random.rand(1))] for i in range(300)])
    pod_index = [i for i in range(2000)]
    train_index = [i for i in range(2000)]
    val_index = [i for i in range(2000, 3000)]
    test_index = [i for i in range(3000, 4000)]

    # Training
    pod_cylinder = PODHeat(n_components, positions, pod_index)
    train_inputs, train_outputs, _ = pod_cylinder.prepare_data(train_index)
    val_inputs, val_outputs, _ = pod_cylinder.prepare_data(val_index)

    # aug_inputs, aug_outputs = pod_cylinder.sample(1000)
    # train_inputs = np.concatenate([train_inputs, aug_inputs], axis=0)
    # train_outputs = np.concatenate([train_outputs, aug_outputs], axis=0)

    # model = svr_regression(train_inputs, train_outputs, val_inputs, val_outputs)
    # model = rf_regression(train_inputs, train_outputs, val_inputs, val_outputs)
    # model = gp_regression(train_inputs, train_outputs, val_inputs, val_outputs)
    model = ridge_regression(train_inputs, train_outputs, val_inputs, val_outputs)

    # Testing
    test_inputs, test_outputs, test_data = pod_cylinder.prepare_data(test_index)
    test_pres = model.predict(test_inputs)
    test_maps = pod_cylinder.inverse_transform(coff=test_pres)
    print("MAE of coefficients", np.mean(np.abs(test_pres - test_outputs)))
    print("MAE of reconstruct fields", np.mean(np.abs(test_maps - test_data)))
    print("Max-AE of reconstruct fields",
          np.mean(np.max(np.abs(test_maps - test_data).reshape(test_maps.shape[0], -1), axis=1)))
