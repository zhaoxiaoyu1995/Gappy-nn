# -*- coding: utf-8 -*-
# @Time    : 2023/3/15 23:22
# @Author  : zhaoxiaoyu
# @File    : plot_dataset.py
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import cmaps
import numpy as np

data = sio.loadmat('/mnt/jfs/zhaoxiaoyu/data/heat200x200/Example3.mat')
fig = plt.figure(figsize=(9.0, 4))
h, w = 200, 200
x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
x_coor, y_coor = np.meshgrid(x_coor, y_coor)
x_coor, y_coor = x_coor / 100.0, y_coor / 100.0

# The 1th figure
gs0 = gridspec.GridSpec(1, 2)
ax = plt.subplot(gs0[0, :1])
h = ax.imshow(data['F'][::-1, :], cmap='hot', interpolation='bilinear', origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title('Heat source', fontsize=13)

# The 2nd figure
ax = plt.subplot(gs0[0, 1:2])
h = ax.contourf(x_coor, y_coor, data['u'], levels=50, cmap=cmaps.BlAqGrYeOrReVi200, origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title('Temperature Field', fontsize=13)

plt.savefig('{}.pdf'.format('./figure/heat_dataset'), bbox_inches='tight', pad_inches=0)
