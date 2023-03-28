# -*- coding: utf-8 -*-
# @Time    : 2022/5/17 22:16
# @Author  : zhaoxiaoyu
# @File    : visualization.py
import scipy.io as sio
import numpy as np

from utils.visualization import plot_locations

f = sio.loadmat('/mnt/jfs/zhaoxiaoyu/Gappy_POD/airfoil_incompressible.mat')
data = f['p'][0, 0, :, :]

positions = np.array(
    [[65, 126], [90, 139], [65, 151], [90, 114], [90, 164], [115, 139], [65, 101], [115, 164],
     [65, 176], [115, 114], [90, 89], [90, 189], [115, 89], [115, 189], [40, 164], [65, 76]]
)
plot_locations(positions, data)
