# -*- coding: utf-8 -*-
# @Time    : 2022/6/30 0:02
# @Author  : zhaoxiaoyu
# @File    : plot_results.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

# The optional setting of style
# Style setting: 'science', 'ieee', 'nature', 'grid'
# Color style: 'bright', 'vibrant', 'muted', 'high-contrast', 'light', 'high-vis', 'retro'
plt.style.use(['science', 'ieee', 'grid', 'bright'])


def plot_line_chart(df, ylim=None, save_path=None, xlabel=None, ylabel=None):
    data = df.to_numpy()

    index = df.index
    columns = df.columns
    x = np.array([i + 1 for i in range(len(columns))])

    markers = ['>', '<', '^', 'v', 'd', 'p', 'o', 's', 'h', 'H', 'D']
    for i in range(data.shape[0]):
        plt.plot(x, data[i, :], '--', label=index[i], marker=markers[i], markersize=5,
                 markerfacecolor='none')

    # Set the legend
    plt.legend(ncol=3, prop={'size': 6}, frameon=False)

    # Set the axis
    plt.xticks(x, columns)
    plt.yscale('log')
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    if xlabel is None:
        plt.xlabel("number of sensors")
    else:
        plt.xlabel(xlabel)
    if ylabel is None:
        plt.ylabel("MAE")
    else:
        plt.ylabel(ylabel)

    # Save the figure
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()


if __name__ == '__main__':
    # Load data
    df = pd.read_excel('./data/airfoil_compressible.xlsx', engine='openpyxl', index_col=0, sheet_name='MAE_vy')
    # plot_line_chart(df, ylim=(9e-4, 8e-1), ylabel='MAE')
    plot_line_chart(df, ylim=(1.4e-1, 3.2e0), save_path='./figure/airfoil_com_vy_mae.pdf', ylabel='MAE')
