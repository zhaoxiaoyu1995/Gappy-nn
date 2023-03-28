# -*- coding: utf-8 -*-
# @Time    : 2022/12/4 3:57
# @Author  : zhaoxiaoyu
# @File    : plot_cylinder.py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import scipy.io as sio
import numpy as np
import matplotlib.ticker
import cmaps

model_name = ['gappy', 'mlp_pod', 'rfr', 'mlp', 'mlp_gappy', 'cnn', 'gappy_cnn']
data = sio.loadmat('./data/heat/' + model_name[0] + '.mat')

fig = plt.figure(figsize=(14, 5.8))

# The 1th figure
gs0 = gridspec.GridSpec(2, 4)
ax = plt.subplot(gs0[0, :1])
h = ax.imshow(data['true'][::-1, :] + 298.0, interpolation='bilinear', cmap=cmaps.BlAqGrYeOrReVi200, origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
positions = np.array(
    [[40, 40], [40, 80], [40, 120], [40, 160], [80, 40], [80, 80], [80, 120], [80, 160], [120, 40], [120, 80],
     [120, 120], [120, 160], [160, 40], [160, 80], [160, 120], [160, 160]])
ax.scatter(positions[:, 1], positions[:, 0], c='black', s=8, label='sensors')
ax.legend()
ax.set_title('Referenced field $T$', fontsize=10)

# The 2nd figure
cbformat = matplotlib.ticker.ScalarFormatter()
cbformat.set_powerlimits((-2, 2))
ax = plt.subplot(gs0[0, 1:2])
h = ax.imshow(data['true'][::-1, :] - data['pre'][::-1, :], interpolation='bilinear', cmap=cmaps.BlAqGrYeOrReVi200, origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax, format=cbformat)
ax.axis('off')
ax.set_title(r'Error ${T  - {T ^*}}$ (Gappy-POD)', fontsize=10)

# The 3rd figure
cbformat = matplotlib.ticker.ScalarFormatter()
cbformat.set_powerlimits((-2, 2))
data = sio.loadmat('./data/heat/' + model_name[1] + '.mat')
ax = plt.subplot(gs0[0, 2:3])
h = ax.imshow(50.0 * (data['true'][::-1, :] - data['pre'][::-1, :]), interpolation='bilinear', cmap=cmaps.BlAqGrYeOrReVi200, origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax, format=cbformat)
ax.axis('off')
ax.set_title(r'Error ${T  - {T ^*}}$ (MLP-POD)', fontsize=10)

# The 4th figure
cbformat = matplotlib.ticker.ScalarFormatter()
cbformat.set_powerlimits((-2, 2))
data = sio.loadmat('./data/heat/' + model_name[2] + '.mat')
ax = plt.subplot(gs0[0, 3:4])
h = ax.imshow(data['true'][::-1, :] - data['pre'][::-1, :], interpolation='bilinear', cmap=cmaps.BlAqGrYeOrReVi200, origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax, format=cbformat)
ax.axis('off')
ax.set_title(r'Error ${T  - {T ^*}}$ (RFR-POD)', fontsize=10)

# The 5th figure
cbformat = matplotlib.ticker.ScalarFormatter()
cbformat.set_powerlimits((-2, 2))
data = sio.loadmat('./data/heat/' + model_name[3] + '.mat')
ax = plt.subplot(gs0[1, 0:1])
h = ax.imshow(50.0 * (data['true'][::-1, :] - data['pre'][::-1, :]), interpolation='bilinear', cmap=cmaps.BlAqGrYeOrReVi200, origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax, format=cbformat)
ax.axis('off')
ax.set_title(r'Error ${T  - {T ^*}}$ (MLP)', fontsize=10)

# The 6th figure
data = sio.loadmat('./data/heat/' + model_name[4] + '.mat')
cbformat = matplotlib.ticker.ScalarFormatter()
cbformat.set_powerlimits((-2, 2))
ax = plt.subplot(gs0[1, 1:2])
h = ax.imshow((data['true'][::-1, :] - data['pre'][::-1, :]) * 50.0, interpolation='bilinear', cmap=cmaps.BlAqGrYeOrReVi200, origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax, format=cbformat)
ax.axis('off')
ax.set_title(r'Error ${T  - {T ^*}}$ (Gappy-MLP)', fontsize=10)

# The 7th figure
data = sio.loadmat('./data/heat/' + model_name[5] + '.mat')
ax = plt.subplot(gs0[1, 2:3])
h = ax.imshow((data['true'][::-1, :] - data['pre'][::-1, :]) * 50.0, interpolation='bilinear', cmap=cmaps.BlAqGrYeOrReVi200, origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${T  - {T ^*}}$ (CNN)', fontsize=10)

# The 8th figure
cbformat = matplotlib.ticker.ScalarFormatter()
cbformat.set_powerlimits((-2, 2))
data = sio.loadmat('./data/heat/' + model_name[6] + '.mat')
ax = plt.subplot(gs0[1, 3:4])
h = ax.imshow((data['true'][::-1, :] - data['pre'][::-1, :]) * 50.0, interpolation='bilinear', cmap=cmaps.BlAqGrYeOrReVi200, origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax, format=cbformat)
ax.axis('off')
ax.set_title(r'Error ${T  - {T ^*}}$ (Gappy-CNN)', fontsize=10)


# plt.show()
plt.savefig('{}.pdf'.format('./figure/heat'), bbox_inches='tight', pad_inches=0)
