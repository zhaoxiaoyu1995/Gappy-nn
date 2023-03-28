# -*- coding: utf-8 -*-
# @Time    : 2022/5/6 10:57
# @Author  : zhaoxiaoyu
# @File    : generate_locations_airfoil.py
import scipy.io as sio
import numpy as np

from utils.utils import generate_locations
from utils.visualization import plot_locations

data = sio.loadmat('/mnt/jfs/zhaoxiaoyu/Gappy_POD/airfoil_incompressible.mat')['vx']
locations = generate_locations(data.squeeze(axis=1), observe_num=8, interval=24)

print(locations)
plot_locations(np.array(locations), data[0, 0, :, :])
