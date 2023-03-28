# -*- coding: utf-8 -*-
# @Time    : 2022/12/4 3:57
# @Author  : zhaoxiaoyu
# @File    : plot_cylinder.py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import scipy.io as sio

data = sio.loadmat('./data/in_airfoil/coor.mat')
x_coor, y_coor = data['x_coor'].reshape(-1), data['y_coor'].reshape(-1)
model_name = ['mlp_pod', 'mlp_graph', 'mlp_cnn']
data = sio.loadmat('./data/in_airfoil/' + model_name[0] + '_vx.mat')

fig = plt.figure(figsize=(15, 9.2))

# The 1th figure
gs0 = gridspec.GridSpec(3, 4)
ax = plt.subplot(gs0[0, :1])
h = ax.tricontour(x_coor, y_coor, data['true'].reshape(-1), levels=50, cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title('Referenced field $V_x$', fontsize=10)

# The 2th figure
data = sio.loadmat('./data/in_airfoil/' + model_name[0] + '_vy.mat')
ax = plt.subplot(gs0[1, :1])
h = ax.tricontour(x_coor, y_coor, data['true'].reshape(-1), levels=50, cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title('Referenced field $V_y$', fontsize=10)

# The 3rd figure
data = sio.loadmat('./data/in_airfoil/' + model_name[0] + '_p.mat')
ax = plt.subplot(gs0[2, :1])
h = ax.tricontour(x_coor, y_coor, data['true'].reshape(-1) / 1000.0, levels=50, cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title('Referenced field $P$', fontsize=10)

# The 4th figure
data = sio.loadmat('./data/in_airfoil/' + model_name[0] + '_vx.mat')
ax = plt.subplot(gs0[0, 1:2])
h = ax.tricontour(x_coor, y_coor, (data['true'] - data['pre']).reshape(-1), levels=50, cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${V_x  - {{V_x} ^*}}$ (MLP-POD)', fontsize=10)

# The 5th figure
data = sio.loadmat('./data/in_airfoil/' + model_name[0] + '_vy.mat')
ax = plt.subplot(gs0[1, 1:2])
h = ax.tricontour(x_coor, y_coor, (data['true'] - data['pre']).reshape(-1), levels=50, cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${V_y  - {{V_y} ^*}}$ (MLP-POD)', fontsize=10)

# The 6th figure
data = sio.loadmat('./data/in_airfoil/' + model_name[0] + '_p.mat')
ax = plt.subplot(gs0[2, 1:2])
h = ax.tricontour(x_coor, y_coor, (data['true'] - data['pre']).reshape(-1) / 1000.0, levels=50, cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${P  - {P ^*}}$ (MLP-POD)', fontsize=10)

# The 7th figure
data = sio.loadmat('./data/in_airfoil/' + model_name[1] + '_vx.mat')
ax = plt.subplot(gs0[0, 2:3])
h = ax.tricontour(x_coor, y_coor, (data['true'] - data['pre']).reshape(-1), levels=50, cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${V_x  - {{V_x} ^*}}$ (GraphSAGE)', fontsize=10)

# The 8th figure
data = sio.loadmat('./data/in_airfoil/' + model_name[1] + '_vy.mat')
ax = plt.subplot(gs0[1, 2:3])
h = ax.tricontour(x_coor, y_coor, (data['true'] - data['pre']).reshape(-1), levels=50, cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${V_y  - {{V_y} ^*}}$ (GraphSAGE)', fontsize=10)

# The 9th figure
data = sio.loadmat('./data/in_airfoil/' + model_name[1] + '_p.mat')
ax = plt.subplot(gs0[2, 2:3])
h = ax.tricontour(x_coor, y_coor, (data['true'] - data['pre']).reshape(-1) / 1000.0, levels=50, cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${P  - {P ^*}}$ (GraphSAGE)', fontsize=10)

# The 10th figure
data = sio.loadmat('./data/in_airfoil/' + model_name[2] + '_vx.mat')
ax = plt.subplot(gs0[0, 3:4])
h = ax.tricontour(x_coor, y_coor, (data['true'] - data['pre']).reshape(-1), levels=50, cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${V_x  - {{V_x} ^*}}$ (Gappy-CNN)', fontsize=10)

# The 11th figure
data = sio.loadmat('./data/in_airfoil/' + model_name[2] + '_vy.mat')
ax = plt.subplot(gs0[1, 3:4])
h = ax.tricontour(x_coor, y_coor, (data['true'] - data['pre']).reshape(-1), levels=50, cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${V_y  - {{V_y} ^*}}$ (Gappy-CNN)', fontsize=10)

# The 12th figure
data = sio.loadmat('./data/in_airfoil/' + model_name[2] + '_p.mat')
ax = plt.subplot(gs0[2, 3:4])
h = ax.tricontour(x_coor, y_coor, (data['true'] - data['pre']).reshape(-1) / 1000.0, levels=50, cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.axis('off')
ax.set_title(r'Error ${P  - {P ^*}}$ (Gappy-CNN)', fontsize=10)

# plt.show()
plt.savefig('{}.pdf'.format('./figure/in_airfoil'), bbox_inches='tight', pad_inches=0)
