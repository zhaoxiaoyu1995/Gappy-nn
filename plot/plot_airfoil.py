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

model_name = ['gappy', 'cnn', 'gappy_cnn']
data = sio.loadmat('./data/airfoil/' + model_name[0] + '_vx.mat')

fig = plt.figure(figsize=(15, 9.2))
w, h = 256, 256
x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
x_coor, y_coor = np.meshgrid(x_coor, y_coor)
x_coor, y_coor = y_coor / 100.0, x_coor / 100.0
x_coor = x_coor[::-1, :]

# The 1th figure
gs0 = gridspec.GridSpec(3, 4)
ax = plt.subplot(gs0[0, :1])
# h = ax.contourf(x_coor, y_coor, data['true'], interpolation='bilinear', cmap='jet', origin='lower', aspect='auto')
h = ax.contourf(x_coor, y_coor, data['true'], cmap='jet', origin='lower', levels=100)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
positions = np.array(
    [[65, 125], [90, 141], [191, 126], [90, 114], [65, 150], [90, 166], [115, 141], [65, 100]])
x, y = [], []
for i in range(positions.shape[0]):
    x.append(x_coor[positions[i, 0], positions[i, 1]])
    y.append(y_coor[positions[i, 0], positions[i, 1]])
ax.scatter(x, y, c='black', s=6, label='sensors')
ax.legend()
ax.set_title('Referenced field $V_x$', fontsize=10)

# The 2th figure
data = sio.loadmat('./data/airfoil/' + model_name[0] + '_vy.mat')
ax = plt.subplot(gs0[1, :1])
h = ax.contourf(x_coor, y_coor, data['true'], cmap='jet', origin='lower', levels=100)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
positions = np.array(
    [[63, 128], [38, 128], [48, 153], [47, 103], [13, 128], [23, 153], [22, 103], [27, 178]])
x, y = [], []
for i in range(positions.shape[0]):
    x.append(x_coor[positions[i, 0], positions[i, 1]])
    y.append(y_coor[positions[i, 0], positions[i, 1]])
ax.scatter(x, y, c='black', s=6, label='sensors')
ax.legend()
ax.set_title('Referenced field $V_y$', fontsize=10)

# The 3rd figure
data = sio.loadmat('./data/airfoil/' + model_name[0] + '_p.mat')
ax = plt.subplot(gs0[2, :1])
h = ax.contourf(x_coor, y_coor, data['true'], cmap='jet', origin='lower', levels=100)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
positions = np.array(
    [[65, 126], [90, 139], [65, 151], [90, 114], [90, 164], [115, 139], [65, 101], [115, 164]])
x, y = [], []
for i in range(positions.shape[0]):
    x.append(x_coor[positions[i, 0], positions[i, 1]])
    y.append(y_coor[positions[i, 0], positions[i, 1]])
ax.scatter(x, y, c='black', s=6, label='sensors')
ax.legend()
ax.set_title('Referenced field $P$', fontsize=10)

# The 4th figure
data = sio.loadmat('./data/airfoil/' + model_name[0] + '_vx.mat')
ax = plt.subplot(gs0[0, 1:2])
h = ax.imshow(data['true'].T - data['pre'].T, interpolation='bilinear', cmap='jet', origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${V_x  - {{V_x} ^*}}$ (Gappy-POD)', fontsize=10)

# The 5th figure
data = sio.loadmat('./data/airfoil/' + model_name[0] + '_vy.mat')
ax = plt.subplot(gs0[1, 1:2])
h = ax.imshow(data['true'].T - data['pre'].T, interpolation='bilinear', cmap='jet', origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${V_y  - {{V_y} ^*}}$ (Gappy-POD)', fontsize=10)

# The 6th figure
data = sio.loadmat('./data/airfoil/' + model_name[0] + '_p.mat')
ax = plt.subplot(gs0[2, 1:2])
h = ax.imshow(data['true'].T - data['pre'].T, interpolation='bilinear', cmap='jet', origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${P  - {P ^*}}$ (Gappy-POD)', fontsize=10)

# The 7th figure
data = sio.loadmat('./data/airfoil/' + model_name[1] + '_vx.mat')
ax = plt.subplot(gs0[0, 2:3])
h = ax.imshow(data['true'].T - data['pre'].T, interpolation='bilinear', cmap='jet', origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${V_x  - {{V_x} ^*}}$ (CNN)', fontsize=10)

# The 8th figure
data = sio.loadmat('./data/airfoil/' + model_name[1] + '_vy.mat')
ax = plt.subplot(gs0[1, 2:3])
h = ax.imshow(data['true'].T - data['pre'].T, interpolation='bilinear', cmap='jet', origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${V_y  - {{V_y} ^*}}$ (CNN)', fontsize=10)

# The 9th figure
data = sio.loadmat('./data/airfoil/' + model_name[1] + '_p.mat')
ax = plt.subplot(gs0[2, 2:3])
h = ax.imshow(data['true'].T - data['pre'].T, interpolation='bilinear', cmap='jet', origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${P  - {P ^*}}$ (CNN)', fontsize=10)

# The 10th figure
data = sio.loadmat('./data/airfoil/' + model_name[2] + '_vx.mat')
ax = plt.subplot(gs0[0, 3:4])
h = ax.imshow(data['true'].T - data['pre'].T, interpolation='bilinear', cmap='jet', origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${V_x  - {{V_x} ^*}}$ (Gappy-CNN)', fontsize=10)

# The 11th figure
data = sio.loadmat('./data/airfoil/' + model_name[2] + '_vy.mat')
ax = plt.subplot(gs0[1, 3:4])
h = ax.imshow(data['true'].T - data['pre'].T, interpolation='bilinear', cmap='jet', origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${V_y  - {{V_y} ^*}}$ (Gappy-CNN)', fontsize=10)

# The 12th figure
data = sio.loadmat('./data/airfoil/' + model_name[2] + '_p.mat')
ax = plt.subplot(gs0[2, 3:4])
h = ax.imshow(data['true'].T - data['pre'].T, interpolation='bilinear', cmap='jet', origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${P  - {P ^*}}$ (Gappy-CNN)', fontsize=10)

# plt.show()
plt.savefig('{}.pdf'.format('./figure/airfoil'), bbox_inches='tight', pad_inches=0)
