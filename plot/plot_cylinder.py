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

model_name = ['gappy', 'mlp_pod', 'rfr', 'svr', 'mlp', 'gappy_mlp', 'cnn', 'gappy_cnn']
data = sio.loadmat('./data/cylinder/' + model_name[0] + '.mat')

fig = plt.figure(figsize=(12, 7.5))

# The 1th figure
cbformat = matplotlib.ticker.ScalarFormatter()
cbformat.set_powerlimits((-2, 2))
gs0 = gridspec.GridSpec(3, 3)
ax = plt.subplot(gs0[0, :1])
h = ax.imshow(data['true'], interpolation='bilinear', cmap='seismic', origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax, format=cbformat)
ax.axis('off')
positions = np.array([[50, 25], [62, 30], [50, 35], [62, 40], [50, 45], [62, 50], [50, 55], [62, 60]])
ax.scatter(positions[:, 1], 112 - positions[:, 0], c='black', s=8, label='sensors')
ax.legend(frameon=False)
ax.set_title('Referenced field $\Omega$', fontsize=10)

# The 2nd figure
cbformat = matplotlib.ticker.ScalarFormatter()
cbformat.set_powerlimits((-2, 2))
ax = plt.subplot(gs0[0, 1:2])
h = ax.imshow(data['true'] - data['pre'], interpolation='bilinear', cmap='seismic', origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax, format=cbformat)
ax.axis('off')
ax.set_title(r'Error ${\Omega  - {\Omega ^*}}$ (Gappy-POD)', fontsize=10)

# The 3rd figure
cbformat = matplotlib.ticker.ScalarFormatter()
cbformat.set_powerlimits((-2, 2))
data = sio.loadmat('./data/cylinder/' + model_name[1] + '.mat')
ax = plt.subplot(gs0[0, 2:3])
h = ax.imshow(data['true'] - data['pre'], interpolation='bilinear', cmap='seismic', origin='lower', aspect='auto',
              vmin=-1 * np.max(np.abs(data['true'] - data['pre'])), vmax=np.max(np.abs(data['true'] - data['pre'])))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax, format=cbformat)
ax.axis('off')
ax.set_title(r'Error ${\Omega  - {\Omega ^*}}$ (MLP-POD)', fontsize=10)

# The 4th figure
cbformat = matplotlib.ticker.ScalarFormatter()
cbformat.set_powerlimits((-2, 2))
data = sio.loadmat('./data/cylinder/' + model_name[2] + '.mat')
ax = plt.subplot(gs0[1, :1])
h = ax.imshow(data['true'] - data['pre'], interpolation='bilinear', cmap='seismic', origin='lower', aspect='auto',
              vmin=-1 * np.max(np.abs(data['true'] - data['pre'])), vmax=np.max(np.abs(data['true'] - data['pre'])))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax, format=cbformat)
ax.axis('off')
ax.set_title(r'Error ${\Omega  - {\Omega ^*}}$ (RFR-POD)', fontsize=10)

# The 5th figure
cbformat = matplotlib.ticker.ScalarFormatter()
cbformat.set_powerlimits((-2, 2))
data = sio.loadmat('./data/cylinder/' + model_name[3] + '.mat')
ax = plt.subplot(gs0[1, 1:2])
h = ax.imshow(data['true'] - data['pre'], interpolation='bilinear', cmap='seismic', origin='lower', aspect='auto',
              vmin=-1 * np.max(np.abs(data['true'] - data['pre'])), vmax=np.max(np.abs(data['true'] - data['pre'])))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax, format=cbformat)
ax.axis('off')
ax.set_title(r'Error ${\Omega  - {\Omega ^*}}$ (SVR-POD)', fontsize=10)

# The 6th figure
cbformat = matplotlib.ticker.ScalarFormatter()
cbformat.set_powerlimits((-2, 2))
data = sio.loadmat('./data/cylinder/' + model_name[4] + '.mat')
ax = plt.subplot(gs0[1, 2:3])
h = ax.imshow(data['true'] - data['pre'], interpolation='bilinear', cmap='seismic', origin='lower', aspect='auto',
              vmin=-1 * np.max(np.abs(data['true'] - data['pre'])), vmax=np.max(np.abs(data['true'] - data['pre'])))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax, format=cbformat)
ax.axis('off')
ax.set_title(r'Error ${\Omega  - {\Omega ^*}}$ (MLP)', fontsize=10)

# The 7th figure
cbformat = matplotlib.ticker.ScalarFormatter()
cbformat.set_powerlimits((-2, 2))
data = sio.loadmat('./data/cylinder/' + model_name[5] + '.mat')
ax = plt.subplot(gs0[2, 0:1])
h = ax.imshow(data['true'] - data['pre'], interpolation='bilinear', cmap='seismic', origin='lower', aspect='auto',
              vmin=-1 * np.max(np.abs(data['true'] - data['pre'])), vmax=np.max(np.abs(data['true'] - data['pre'])))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax, format=cbformat)
ax.axis('off')
ax.set_title(r'Error ${\Omega  - {\Omega ^*}}$ (Gappy-MLP)', fontsize=10)

# The 8th figure
cbformat = matplotlib.ticker.ScalarFormatter()
cbformat.set_powerlimits((-2, 2))
data = sio.loadmat('./data/cylinder/' + model_name[6] + '.mat')
ax = plt.subplot(gs0[2, 1:2])
h = ax.imshow(data['true'] - data['pre'], interpolation='bilinear', cmap='seismic', origin='lower', aspect='auto',
              vmin=-1 * np.max(np.abs(data['true'] - data['pre'])), vmax=np.max(np.abs(data['true'] - data['pre'])))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax, format=cbformat)
ax.axis('off')
ax.set_title(r'Error ${\Omega  - {\Omega ^*}}$ (CNN)', fontsize=10)

# The 9th figure
cbformat = matplotlib.ticker.ScalarFormatter()
cbformat.set_powerlimits((-2, 2))
data = sio.loadmat('./data/cylinder/' + model_name[7] + '.mat')
ax = plt.subplot(gs0[2, 2:3])
h = ax.imshow(data['true'] - data['pre'], interpolation='bilinear', cmap='seismic', origin='lower', aspect='auto',
              vmin=-1 * np.max(np.abs(data['true'] - data['pre'])), vmax=np.max(np.abs(data['true'] - data['pre'])))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax, format=cbformat)
ax.axis('off')
ax.set_title(r'Error ${\Omega  - {\Omega ^*}}$ (Gappy-CNN)', fontsize=10)

# plt.show()
plt.savefig('{}.pdf'.format('./figure/cylinder'), bbox_inches='tight', pad_inches=0)
