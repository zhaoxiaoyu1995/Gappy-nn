# -*- coding: utf-8 -*-
# @Time    : 2022/10/8 2:08
# @Author  : zhaoxiaoyu
# @File    : heat2D_pod_ml.py
from sklearn.decomposition import PCA
import numpy as np
import h5py
import os
import sys

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(filename)

from model.ml import svr_regression, rf_regression, gp_regression


class PODHeat:
    def __init__(self, n_components, positions, pod_index):
        self.positions = positions
        # Loading data
        f = h5py.File('/mnt/jfs/zhaoxiaoyu/data/heat/temperature.h5', 'r')
        pca_data = f['u'][pod_index, :, :, :]

        self.pca = PCA(n_components=n_components)
        X_t = self.pca.fit_transform(pca_data.reshape(len(pod_index), -1))
        self.max_coeff, self.min_coeff = np.max(X_t, axis=0).reshape(1, -1), np.min(X_t, axis=0).reshape(1, -1)
        self.size = pca_data.shape[-3:]

    def prepare_data(self, data_index):
        # Loading data
        f = h5py.File('/mnt/jfs/zhaoxiaoyu/data/heat/temperature.h5', 'r')
        data = f['u'][data_index, :, :, :]

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


if __name__ == '__main__':
    n_components = 25
    positions = np.array(
        [[40, 40], [40, 80], [40, 120], [40, 160], [80, 40], [80, 80], [80, 120], [80, 160], [120, 40], [120, 80],
         [120, 120], [120, 160], [160, 40], [160, 80], [160, 120], [160, 160]])
    pod_index = [i for i in range(5000)]
    train_index = [i for i in range(4000)]
    val_index = [i for i in range(4000, 5000)]
    test_index = [i for i in range(5999, 6000)]

    # Training
    pod_cylinder = PODHeat(n_components, positions, pod_index)
    train_inputs, train_outputs, _ = pod_cylinder.prepare_data(train_index)
    val_inputs, val_outputs, _ = pod_cylinder.prepare_data(val_index)

    # model = svr_regression(train_inputs, train_outputs, val_inputs, val_outputs)
    model = rf_regression(train_inputs, train_outputs, val_inputs, val_outputs)
    # model = gp_regression(train_inputs, train_outputs, val_inputs, val_outputs)

    # Testing
    test_inputs, test_outputs, test_data = pod_cylinder.prepare_data(test_index)
    test_pres = model.predict(test_inputs)
    test_maps = pod_cylinder.inverse_transform(coff=test_pres)
    print("MAE of coefficients", np.mean(np.abs(test_pres - test_outputs)))
    print("MAE of reconstruct fields", np.mean(np.abs(test_maps - test_data)))
    print("Max-AE of reconstruct fields",
          np.mean(np.max(np.abs(test_maps - test_data).reshape(test_maps.shape[0], -1), axis=1)))

    import scipy.io as sio

    sio.savemat('rfr.mat', {
        'true': test_data[-1, 0, :, :],
        'pre': test_maps[-1, 0, :, :]
    })
