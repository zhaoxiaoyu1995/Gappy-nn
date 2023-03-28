# -*- coding: utf-8 -*-
# @Time    : 2022/5/6 10:57
# @Author  : zhaoxiaoyu
# @File    : generate_locations_airfoil.py
import scipy.io as sio
import numpy as np
import h5py

from utils.utils import generate_locations
from utils.visualization import plot_locations

f = h5py.File('/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_train.h5', 'r')
data = f['p'][:]
sorted_index = sio.loadmat('/mnt/jfs/zhaoxiaoyu/Gappy_POD/index3.mat')['data']
data = data[:, sorted_index].reshape(-1, 1, 200, 200)
locations = generate_locations(data.squeeze(axis=1), observe_num=64, interval=24)

sorted_index = sorted_index.reshape(200, 200)
locations = [[14 + 14 * i, 14 + 14 * j] for i in range(13) for j in range(13)]
index = []
for location in locations:
    index.append(sorted_index[location[0], location[1]])
print(index)
print(len(index))
print(locations)
plot_locations(np.array(locations), data[0, 0, :, :])
