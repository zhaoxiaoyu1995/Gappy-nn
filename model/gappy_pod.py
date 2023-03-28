# -*- coding: utf-8 -*-
# @Time    : 2022/10/5 0:10
# @Author  : zhaoxiaoyu
# @File    : gappy_pod.py
import torch
import numpy as np
from sklearn.decomposition import PCA


class GappyPod():
    def __init__(self, data, map_size=(112, 192), n_components=50,
                 positions=np.array([[50, 25], [62, 30]])):
        self.data = data
        self.pca = PCA(n_components=n_components)
        self.pca.fit(self.data.reshape(data.shape[0], -1))

        self.positions = positions
        self.map_size = map_size

        components = self.pca.components_
        means = self.pca.mean_

        component = components.reshape(-1, self.map_size[0], self.map_size[1])
        mean = means.reshape(-1, self.map_size[0], self.map_size[1])

        component_mask, mean_mask = [], []
        for i in range(self.positions.shape[0]):
            component_mask.append(component[:, self.positions[i, 0], :][:, self.positions[i, 1]].reshape(-1, 1))
            mean_mask.append(mean[:, self.positions[i, 0], :][:, self.positions[i, 1]].reshape(-1, 1))

        self.component_mask = torch.from_numpy(np.concatenate(component_mask, axis=-1)).float().cuda().unsqueeze(
            0)
        self.mean_mask = torch.from_numpy(np.concatenate(mean_mask, axis=-1).T).float().cuda()
        self.components_ = torch.from_numpy(components).float().cuda()
        self.mean_ = torch.from_numpy(means).reshape(1, -1).float().cuda()

    def reconstruct(self, observations):
        component_mask = self.component_mask.repeat(observations.shape[0], 1, 1)
        observe = (observations.T - self.mean_mask).unsqueeze(dim=1).permute(2, 0, 1)
        coff_pre = torch.linalg.inv(component_mask @ component_mask.permute(0, 2, 1)) @ component_mask @ observe
        coff_pre = coff_pre.squeeze(dim=-1)
        recons = self.inverse_transform(coff_pre)
        recons = recons.reshape(recons.shape[0], 1, self.map_size[0], self.map_size[1])
        return recons

    def inverse_transform(self, coff):
        return coff @ self.components_ + self.mean_


class GappyPodWeight():
    def __init__(self, data, map_size=(112, 192), n_components=50,
                 positions=np.array([[50, 25], [62, 30]]), observe_weight=50):
        self.data = data
        self.pca = PCA(n_components=n_components)
        self.pca.fit(self.data.reshape(data.shape[0], -1))

        self.positions_observe = positions
        self.positions_pre = np.array([[i, j] for i in range(map_size[0]) for j in range(map_size[1])])
        self.map_size = map_size
        self.observe_weight = observe_weight

        components = self.pca.components_
        means = self.pca.mean_

        component = components.reshape(-1, self.map_size[0], self.map_size[1])
        mean = means.reshape(-1, self.map_size[0], self.map_size[1])

        component_mask, mean_mask = [], []
        for i in range(self.positions_pre.shape[0]):
            component_mask.append(component[:, self.positions_pre[i, 0], :][:, self.positions_pre[i, 1]].reshape(-1, 1))
            mean_mask.append(mean[:, self.positions_pre[i, 0], :][:, self.positions_pre[i, 1]].reshape(-1, 1))
        for i in range(self.positions_observe.shape[0]):
            component_mask.append(
                self.observe_weight * component[:, self.positions_observe[i, 0], :][:, self.positions_observe[i, 1]]
                .reshape(-1, 1)
            )
            mean_mask.append(
                self.observe_weight * mean[:, self.positions_observe[i, 0], :][:, self.positions_observe[i, 1]]
                .reshape(-1, 1)
            )

        self.component_mask = torch.from_numpy(
            np.concatenate(component_mask, axis=-1)).float().cuda().unsqueeze(0)
        self.mean_mask = torch.from_numpy(np.concatenate(mean_mask, axis=-1).T).float().cuda()
        self.components_ = torch.from_numpy(components).float().cuda()
        self.mean_ = torch.from_numpy(means).reshape(1, -1).float().cuda()

    def reconstruct(self, pres, inputs, weight):
        component_mask = self.component_mask.repeat(inputs.shape[0], 1, 1)
        observe = torch.cat([(weight * pres).flatten(1), self.observe_weight * inputs], dim=-1)
        mask_temp = torch.cat([weight.flatten(1), torch.ones(inputs.shape[0], self.positions_observe.shape[0]).cuda()],
                              dim=1)
        observe = (observe.T - self.mean_mask * mask_temp.T).unsqueeze(dim=1).permute(2, 0, 1)
        component_mask_temp = component_mask * mask_temp.unsqueeze(dim=1)
        coff_pre = torch.linalg.inv(
            component_mask_temp @ component_mask_temp.permute(0, 2, 1)) @ component_mask_temp @ observe
        coff_pre = coff_pre.squeeze(dim=-1)
        recons = self.inverse_transform(coff_pre)
        recons = recons.reshape(recons.shape[0], 1, self.map_size[0], self.map_size[1])
        return recons

    def inverse_transform(self, coff):
        return coff @ self.components_ + self.mean_


class GappyPodWeight2():
    def __init__(self, data, map_size=(112, 192), n_components=50):
        self.data = data
        self.pca = PCA(n_components=n_components)
        self.pca.fit(self.data.reshape(data.shape[0], -1))

        self.positions_pre = np.array([[i, j] for i in range(map_size[0]) for j in range(map_size[1])])
        self.map_size = map_size

        components = self.pca.components_
        means = self.pca.mean_

        component = components.reshape(-1, self.map_size[0], self.map_size[1])
        mean = means.reshape(-1, self.map_size[0], self.map_size[1])

        component_mask, mean_mask = [], []
        for i in range(self.positions_pre.shape[0]):
            component_mask.append(component[:, self.positions_pre[i, 0], :][:, self.positions_pre[i, 1]].reshape(-1, 1))
            mean_mask.append(mean[:, self.positions_pre[i, 0], :][:, self.positions_pre[i, 1]].reshape(-1, 1))

        self.component_mask = torch.from_numpy(
            np.concatenate(component_mask, axis=-1)).float().cuda().unsqueeze(0)
        self.mean_mask = torch.from_numpy(np.concatenate(mean_mask, axis=-1).T).float().cuda()
        self.components_ = torch.from_numpy(components).float().cuda()
        self.mean_ = torch.from_numpy(means).reshape(1, -1).float().cuda()

    def reconstruct(self, pres, weight):
        component_mask = self.component_mask.repeat(pres.shape[0], 1, 1)
        observe = (weight * pres).flatten(1)
        mask_temp = weight.flatten(1)
        observe = (observe.T - self.mean_mask * mask_temp.T).unsqueeze(dim=1).permute(2, 0, 1)
        component_mask_temp = component_mask * mask_temp.unsqueeze(dim=1)
        coff_pre = torch.linalg.inv(
            component_mask_temp @ component_mask_temp.permute(0, 2, 1)) @ component_mask_temp @ observe
        coff_pre = coff_pre.squeeze(dim=-1)
        recons = self.inverse_transform(coff_pre)
        recons = recons.reshape(recons.shape[0], self.map_size[0], self.map_size[1])
        return recons

    def inverse_transform(self, coff):
        return coff @ self.components_ + self.mean_


class GappyPod1D():
    def __init__(self, data, map_size=16339, n_components=50,
                 positions=np.array([50, 25, 62, 30])):
        self.data = data
        self.pca = PCA(n_components=n_components)
        self.pca.fit(self.data.reshape(data.shape[0], -1))

        self.positions = positions
        self.map_size = map_size

        components = self.pca.components_
        means = self.pca.mean_

        component = components.reshape(-1, map_size)
        mean = means.reshape(-1, map_size)

        component_mask, mean_mask = [], []
        for i in range(self.positions.shape[0]):
            component_mask.append(component[:, self.positions[i]].reshape(-1, 1))
            mean_mask.append(mean[:, self.positions[i]].reshape(-1, 1))

        self.component_mask = torch.from_numpy(np.concatenate(component_mask, axis=-1)).float().cuda().unsqueeze(
            0)
        self.mean_mask = torch.from_numpy(np.concatenate(mean_mask, axis=-1).T).float().cuda()
        self.components_ = torch.from_numpy(components).float().cuda()
        self.mean_ = torch.from_numpy(means).reshape(1, -1).float().cuda()

    def reconstruct(self, observations):
        component_mask = self.component_mask.repeat(observations.shape[0], 1, 1)
        observe = (observations.T - self.mean_mask).unsqueeze(dim=1).permute(2, 0, 1)
        coff_pre = torch.linalg.inv(
            component_mask @ component_mask.permute(0, 2, 1)) @ component_mask @ observe
        coff_pre = coff_pre.squeeze(dim=-1)
        recons = self.inverse_transform(coff_pre)
        recons = recons.reshape(recons.shape[0], self.map_size)
        return recons

    def inverse_transform(self, coff):
        return coff @ self.components_ + self.mean_


class GappyPodWeight1D():
    def __init__(self, data, map_size=16339, n_components=50,
                 positions=np.array([50, 25, 62, 30]), observe_weight=50):
        self.data = data
        self.pca = PCA(n_components=n_components)
        self.pca.fit(self.data.reshape(data.shape[0], -1))

        self.positions_observe = positions
        self.positions_pre = np.array([i for i in range(map_size)])
        self.map_size = map_size
        self.observe_weight = observe_weight

        components = self.pca.components_
        means = self.pca.mean_
        component, mean = components.reshape(-1, self.map_size), means.reshape(-1, self.map_size)

        component_mask, mean_mask = [], []
        for i in range(self.positions_pre.shape[0]):
            component_mask.append(component[:, self.positions_pre[i]].reshape(-1, 1))
            mean_mask.append(mean[:, self.positions_pre[i]].reshape(-1, 1))
        for i in range(self.positions_observe.shape[0]):
            component_mask.append(
                self.observe_weight * component[:, self.positions_observe[i]].reshape(-1, 1)
            )
            mean_mask.append(
                self.observe_weight * mean[:, self.positions_observe[i]].reshape(-1, 1)
            )

        self.component_mask = torch.from_numpy(
            np.concatenate(component_mask, axis=-1)).float().cuda().unsqueeze(0)
        self.mean_mask = torch.from_numpy(np.concatenate(mean_mask, axis=-1).T).float().cuda()
        self.components_ = torch.from_numpy(components).float().cuda()
        self.mean_ = torch.from_numpy(means).reshape(1, -1).float().cuda()

    def reconstruct(self, pres, inputs, weight):
        component_mask = self.component_mask.repeat(inputs.shape[0], 1, 1)
        observe = torch.cat([(weight * pres).flatten(1), self.observe_weight * inputs], dim=-1)
        mask_temp = torch.cat([weight.flatten(1), torch.ones(inputs.shape[0], self.positions_observe.shape[0]).cuda()],
                              dim=1)
        observe = (observe.T - self.mean_mask * mask_temp.T).unsqueeze(dim=1).permute(2, 0, 1)
        component_mask_temp = component_mask * mask_temp.unsqueeze(dim=1)
        coff_pre = torch.linalg.inv(
            component_mask_temp @ component_mask_temp.permute(0, 2, 1)) @ component_mask_temp @ observe
        coff_pre = coff_pre.squeeze(dim=-1)
        recons = self.inverse_transform(coff_pre)
        return recons

    def inverse_transform(self, coff):
        return coff @ self.components_ + self.mean_


class GappyPodWeightLayout():
    def __init__(self, data, map_size=(112, 192), n_components=50):
        self.data = data
        self.pca = PCA(n_components=n_components)
        self.pca.fit(self.data.reshape(data.shape[0], -1))

        self.positions_pre = np.array([[i, j] for i in range(map_size[0]) for j in range(map_size[1])])
        self.map_size = map_size

        components = self.pca.components_
        means = self.pca.mean_

        component = components.reshape(-1, self.map_size[0], self.map_size[1])
        mean = means.reshape(-1, self.map_size[0], self.map_size[1])

        component_mask, mean_mask = [], []
        for i in range(self.positions_pre.shape[0]):
            component_mask.append(component[:, self.positions_pre[i, 0], :][:, self.positions_pre[i, 1]].reshape(-1, 1))
            mean_mask.append(mean[:, self.positions_pre[i, 0], :][:, self.positions_pre[i, 1]].reshape(-1, 1))

        self.component_mask = torch.from_numpy(
            np.concatenate(component_mask, axis=-1)).float().cuda().unsqueeze(0)
        self.mean_mask = torch.from_numpy(np.concatenate(mean_mask, axis=-1).T).float().cuda()
        self.components_ = torch.from_numpy(components).float().cuda()
        self.mean_ = torch.from_numpy(means).reshape(1, -1).float().cuda()

    def reconstruct(self, pres, weight):
        component_mask = self.component_mask.repeat(pres.shape[0], 1, 1)
        observe = (weight * pres).flatten(1)
        mask_temp = weight.flatten(1)
        observe = (observe.T - self.mean_mask * mask_temp.T).unsqueeze(dim=1).permute(2, 0, 1)
        component_mask_temp = component_mask * mask_temp.unsqueeze(dim=1)
        coff_pre = torch.linalg.inv(
            component_mask_temp @ component_mask_temp.permute(0, 2, 1)) @ component_mask_temp @ observe
        coff_pre = coff_pre.squeeze(dim=-1)
        recons = self.inverse_transform(coff_pre)
        recons = recons.reshape(recons.shape[0], 1, self.map_size[0], self.map_size[1])
        return recons

    def inverse_transform(self, coff):
        return coff @ self.components_ + self.mean_
