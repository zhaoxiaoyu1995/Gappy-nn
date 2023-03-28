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

data_index = sio.loadmat('./data/in_airfoil/index3.mat')['data']
model_name = ['mlp_pod', 'mlp_graph', 'mlp_cnn']
data = sio.loadmat('./data/in_airfoil/' + model_name[0] + '_vx.mat')

fig = plt.figure(figsize=(15, 9.2))

# The 1th figure
gs0 = gridspec.GridSpec(3, 4)
ax = plt.subplot(gs0[0, :1])
h = ax.imshow(data['true'][:, data_index].reshape(200, 200), interpolation='bilinear', cmap='jet', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title('Referenced field $V_x$', fontsize=10)

# The 2th figure
data = sio.loadmat('./data/in_airfoil/' + model_name[0] + '_vy.mat')
ax = plt.subplot(gs0[1, :1])
h = ax.imshow(data['true'][:, data_index].reshape(200, 200), interpolation='bilinear', cmap='jet', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title('Referenced field $V_y$', fontsize=10)

# The 3rd figure
data = sio.loadmat('./data/in_airfoil/' + model_name[0] + '_p.mat')
ax = plt.subplot(gs0[2, :1])
h = ax.imshow(data['true'][:, data_index].reshape(200, 200) / 1000.0, interpolation='bilinear', cmap='jet', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title('Referenced field $P$', fontsize=10)

# The 4th figure
data = sio.loadmat('./data/in_airfoil/' + model_name[0] + '_vx.mat')
ax = plt.subplot(gs0[0, 1:2])
h = ax.imshow((data['true'] - data['pre'])[:, data_index].reshape(200, 200), interpolation='bilinear', cmap='jet', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${V_x  - {{V_x} ^*}}$ (Gappy-POD)', fontsize=10)

# The 5th figure
data = sio.loadmat('./data/in_airfoil/' + model_name[0] + '_vy.mat')
ax = plt.subplot(gs0[1, 1:2])
h = ax.imshow((data['true'] - data['pre'])[:, data_index].reshape(200, 200), interpolation='bilinear', cmap='jet', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${V_y  - {{V_y} ^*}}$ (Gappy-POD)', fontsize=10)

# The 6th figure
data = sio.loadmat('./data/in_airfoil/' + model_name[0] + '_p.mat')
ax = plt.subplot(gs0[2, 1:2])
h = ax.imshow((data['true'] - data['pre'])[:, data_index].reshape(200, 200) / 1000.0, interpolation='bilinear', cmap='jet', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${P  - {P ^*}}$ (Gappy-POD)', fontsize=10)

# The 7th figure
data = sio.loadmat('./data/in_airfoil/' + model_name[1] + '_vx.mat')
ax = plt.subplot(gs0[0, 2:3])
h = ax.imshow((data['true'] - data['pre'])[:, data_index].reshape(200, 200), interpolation='bilinear', cmap='jet', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${V_x  - {{V_x} ^*}}$ (CNN)', fontsize=10)

# The 8th figure
data = sio.loadmat('./data/in_airfoil/' + model_name[1] + '_vy.mat')
ax = plt.subplot(gs0[1, 2:3])
h = ax.imshow((data['true'] - data['pre'])[:, data_index].reshape(200, 200), interpolation='bilinear', cmap='jet', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${V_y  - {{V_y} ^*}}$ (CNN)', fontsize=10)

# The 9th figure
data = sio.loadmat('./data/in_airfoil/' + model_name[1] + '_p.mat')
ax = plt.subplot(gs0[2, 2:3])
h = ax.imshow((data['true'] - data['pre'])[:, data_index].reshape(200, 200) / 1000.0, interpolation='bilinear', cmap='jet', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${P  - {P ^*}}$ (CNN)', fontsize=10)

# The 10th figure
data = sio.loadmat('./data/in_airfoil/' + model_name[2] + '_vx.mat')
ax = plt.subplot(gs0[0, 3:4])
h = ax.imshow((data['true'] - data['pre'])[:, data_index].reshape(200, 200), interpolation='bilinear', cmap='jet', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${V_x  - {{V_x} ^*}}$ (Gappy-CNN)', fontsize=10)

# The 11th figure
data = sio.loadmat('./data/in_airfoil/' + model_name[2] + '_vy.mat')
ax = plt.subplot(gs0[1, 3:4])
h = ax.imshow((data['true'] - data['pre'])[:, data_index].reshape(200, 200), interpolation='bilinear', cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${V_y  - {{V_y} ^*}}$ (Gappy-CNN)', fontsize=10)

# The 12th figure
data = sio.loadmat('./data/in_airfoil/' + model_name[2] + '_p.mat')
ax = plt.subplot(gs0[2, 3:4])
h = ax.imshow((data['true'] - data['pre'])[:, data_index].reshape(200, 200) / 1000.0, interpolation='bilinear', cmap='jet', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${P  - {P ^*}}$ (Gappy-CNN)', fontsize=10)

plt.show()
# plt.savefig('{}.pdf'.format('./figure/airfoil'), bbox_inches='tight', pad_inches=0)
