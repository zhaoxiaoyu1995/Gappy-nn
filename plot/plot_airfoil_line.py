# -*- coding: utf-8 -*-
# @Time    : 2022/12/4 3:57
# @Author  : zhaoxiaoyu
# @File    : plot_cylinder.py
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import scienceplots

# The optional setting of style
# Style setting: 'science', 'ieee', 'nature', 'grid'
# Color style: 'bright', 'vibrant', 'muted', 'high-contrast', 'light', 'high-vis', 'retro'
plt.style.use(['science', 'ieee', 'grid', 'muted'])

data = sio.loadmat('./data/airfoil/gappy_p.mat')
x_axis = np.linspace(-0.5 + 96 / 256.0, -0.5 + 160 / 256.0, 64)

fig = plt.figure(figsize=(15, 3))
w, h = 256, 256
x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
x_coor, y_coor = np.meshgrid(x_coor, y_coor)
x_coor, y_coor = y_coor / 100.0, x_coor / 100.0
x_coor = x_coor[::-1, :]

# The 1th figure
gs0 = gridspec.GridSpec(1, 4)
ax = plt.subplot(gs0[0, :1])
h = ax.imshow(data['true'].T == 0, origin='lower')
ax.axis('off')
y = np.linspace(96, 160, 64)
x = np.ones_like(y) * 245
ax.plot(x, y, c='blue', linewidth=2, label='region for visualization')
ax.legend()
ax.set_title('Region for visualization', fontsize=10)

ax = plt.subplot(gs0[0, 1:2])
model_name = ['gappy', 'mlp_pod', 'rfr', 'cnn', 'gappy_cnn']
data = sio.loadmat('./data/airfoil/' + model_name[0] + '_vx.mat')
true = data['true'].T
pre = data['pre'].T
ax.plot(x_axis, true[96:160, 256 - 11], label='Exact')
ax.plot(x_axis, pre[96:160, 256 - 11], '--', label='Gappy')
data = sio.loadmat('./data/airfoil/' + model_name[1] + '_vx.mat')
pre = data['pre'].T
ax.plot(x_axis, pre[96:160, 256 - 11], '--', label='MLP-POD')
data = sio.loadmat('./data/airfoil/' + model_name[2] + '_vx.mat')
pre = data['pre'].T
ax.plot(x_axis, pre[96:160, 256 - 11], '--', label='RFR-POD')
data = sio.loadmat('./data/airfoil/' + model_name[3] + '_vx.mat')
pre = data['pre'].T
ax.plot(x_axis, pre[96:160, 256 - 11], '--', label='CNN')
data = sio.loadmat('./data/airfoil/' + model_name[4] + '_vx.mat')
pre = data['pre'].T
ax.plot(x_axis, pre[96:160, 256 - 11], '--', label='Gappy-CNN')
ax.legend()
ax.set_title('Velocity-x $V_x$', fontsize=10)

ax = plt.subplot(gs0[0, 2:3])
model_name = ['gappy', 'mlp_pod', 'rfr', 'cnn', 'gappy_cnn']
data = sio.loadmat('./data/airfoil/' + model_name[0] + '_vy.mat')
true = data['true'].T
pre = data['pre'].T
ax.plot(x_axis, true[96:160, 256 - 11], label='Exact')
ax.plot(x_axis, pre[96:160, 256 - 11], '--', label='Gappy')
data = sio.loadmat('./data/airfoil/' + model_name[1] + '_vy.mat')
pre = data['pre'].T
ax.plot(x_axis, pre[96:160, 256 - 11], '--', label='MLP-POD')
data = sio.loadmat('./data/airfoil/' + model_name[2] + '_vy.mat')
pre = data['pre'].T
ax.plot(x_axis, pre[96:160, 256 - 11], '--', label='RFR-POD')
data = sio.loadmat('./data/airfoil/' + model_name[3] + '_vy.mat')
pre = data['pre'].T
ax.plot(x_axis, pre[96:160, 256 - 11], '--', label='CNN')
data = sio.loadmat('./data/airfoil/' + model_name[4] + '_vy.mat')
pre = data['pre'].T
ax.plot(x_axis, pre[96:160, 256 - 11], '--', label='Gappy-CNN')
ax.legend()
ax.set_title('Velocity-y $V_y$', fontsize=10)

ax = plt.subplot(gs0[0, 3:4])
model_name = ['gappy', 'mlp_pod', 'rfr', 'cnn', 'gappy_cnn']
data = sio.loadmat('./data/airfoil/' + model_name[0] + '_p.mat')
true = data['true'].T
pre = data['pre'].T
ax.plot(x_axis, true[96:160, 256 - 11], label='Exact')
ax.plot(x_axis, pre[96:160, 256 - 11], '--', label='Gappy')
data = sio.loadmat('./data/airfoil/' + model_name[1] + '_p.mat')
pre = data['pre'].T
ax.plot(x_axis, pre[96:160, 256 - 11], '--', label='MLP-POD')
data = sio.loadmat('./data/airfoil/' + model_name[2] + '_p.mat')
pre = data['pre'].T
ax.plot(x_axis, pre[96:160, 256 - 11], '--', label='RFR-POD')
data = sio.loadmat('./data/airfoil/' + model_name[3] + '_p.mat')
pre = data['pre'].T
ax.plot(x_axis, pre[96:160, 256 - 11], '--', label='CNN')
data = sio.loadmat('./data/airfoil/' + model_name[4] + '_p.mat')
pre = data['pre'].T
ax.plot(x_axis, pre[96:160, 256 - 11], '--', label='Gappy-CNN')
ax.legend()
ax.set_title('Pressure $P$', fontsize=10)

# plt.show()
plt.savefig('{}.pdf'.format('./figure/airfoil_line'), bbox_inches='tight', pad_inches=0)
