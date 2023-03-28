# -*- coding: utf-8 -*-
# @Time    : 2022/10/8 2:08
# @Author  : zhaoxiaoyu
# @File    : sub_airfoil_pod_ml.py
from sklearn.multioutput import MultiOutputRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, DotProduct, Matern
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import numpy as np
import h5py
from tqdm import tqdm
from sklearn import linear_model


def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res


class PODAirfoil:
    def __init__(self, n_components, positions, type='vx'):
        self.positions = positions
        f = h5py.File('/mnt/jfs/zhaoxiaoyu/Gappy_POD/subsonicairfoil/airfoil_train.h5', 'r')
        pca_data = f['p'][:]
        self.min = np.min(f[type][:], axis=0, keepdims=True)
        self.max = np.max(f[type][:], axis=0, keepdims=True)
        f.close()

        self.n_components = n_components
        self.type = type
        self.pca = PCA(n_components=n_components)
        X_t = self.pca.fit_transform(pca_data)
        self.max_coeff, self.min_coeff = np.max(X_t, axis=0).reshape(1, -1), np.min(X_t, axis=0).reshape(1, -1)
        self.train_data_coff = (X_t - self.min_coeff) / (self.max_coeff - self.min_coeff + 1e-10)

    def prepare_data(self, data_path='/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_train.h5'):
        f = h5py.File(data_path)
        data = f['p'][:]
        data_norm = (f['pre'][:] - self.min) / (self.max - self.min + 1e-10)
        data_norm = f['pre'][:]
        f.close()

        # Normalize the coefficients
        data_coff = self.pca.transform(data)
        data_coff = (data_coff - self.min_coeff) / (self.max_coeff - self.min_coeff + 1e-10)

        sparse_data = []
        for i in range(self.positions.shape[0]):
            sparse_data.append(data_norm[:, self.positions[i]].reshape(-1, 1))
        observe = np.concatenate(sparse_data, axis=-1)
        return observe, data_coff, data

    def inverse_transform(self, coff):
        inverse_coff = coff * (self.max_coeff - self.min_coeff) + self.min_coeff
        return self.pca.inverse_transform(inverse_coff).reshape(coff.shape[0], -1)

    def sample(self, n=1000):
        data_coff = []
        for _ in tqdm(range(n)):
            while True:
                coff = np.random.rand(1, self.n_components)
                similar = get_cos_similar_matrix(coff, self.train_data_coff)
                if np.sum(similar > 0.900) > 1:
                    data_coff.append(coff)
                    break
        data_coff = np.concatenate(data_coff, axis=0)
        inverse_coff = data_coff * (self.max_coeff - self.min_coeff) + self.min_coeff
        observe = []
        for i in tqdm(range(int(n / 100.0))):
            data = self.pca.inverse_transform(inverse_coff[i * 100:i * 100 + 100]).reshape(100, -1)
            data = (data - self.min) / (self.max - self.min + 1e-10)
            sparse_data = []
            for i in range(self.positions.shape[0]):
                sparse_data.append(data[:, self.positions[i]].reshape(-1, 1))
            sparse_data = np.concatenate(sparse_data, axis=-1)
            observe.append(sparse_data)
        observe = np.concatenate(observe, axis=0)
        return observe, data_coff


# SVR Regression
def svr_regression(train_inputs, train_outputs, val_inputs, val_outputs):
    X = np.concatenate([train_inputs, val_inputs], axis=0)
    y = np.concatenate([train_outputs, val_outputs], axis=0)
    ps = PredefinedSplit(
        test_fold=np.concatenate([-1 * np.ones((train_inputs.shape[0], 1)), np.zeros((val_inputs.shape[0], 1))],
                                 axis=0)
    )
    tuned_parameter = [
        {'estimator__kernel': ['rbf'],
         'estimator__epsilon': [0.00001, 0.0002, 0.0005, 0.0001, 0.005, 0.001]
         }
    ]
    svr = MultiOutputRegressor(svm.SVR())
    svr_c = GridSearchCV(estimator=svr, param_grid=tuned_parameter, cv=ps, n_jobs=4)
    svr_c.fit(X, y)
    kernel, epsilon = svr_c.best_params_['estimator__kernel'], svr_c.best_params_['estimator__epsilon']
    print('The optimal parameters are: \n kernel:', kernel, 'epsilon:', epsilon)

    svr = MultiOutputRegressor(svm.SVR(kernel=kernel, epsilon=epsilon))
    svr.fit(train_inputs, train_outputs)
    return svr


# Random Forest Regression
def rf_regression(train_inputs, train_outputs, val_inputs, val_outputs):
    X = np.concatenate([train_inputs, val_inputs], axis=0)
    y = np.concatenate([train_outputs, val_outputs], axis=0)
    ps = PredefinedSplit(
        test_fold=np.concatenate([-1 * np.ones((train_inputs.shape[0], 1)), np.zeros((val_inputs.shape[0], 1))],
                                 axis=0)
    )
    tuned_parameter = [{'n_estimators': [100, 300, 500, 1000]}]
    rfr = RandomForestRegressor()
    rfr_c = GridSearchCV(estimator=rfr, param_grid=tuned_parameter, cv=ps, n_jobs=4)
    rfr_c.fit(X, y)
    n_estimators = rfr_c.best_params_['n_estimators']
    print('The optimal parameters are: \n n_estimators:', n_estimators)

    rfr = RandomForestRegressor(n_estimators=n_estimators)
    rfr.fit(train_inputs, train_outputs)
    return rfr


# Gaussian Process Regression
def gp_regression(train_inputs, train_outputs, val_inputs, val_outputs):
    X = np.concatenate([train_inputs, val_inputs], axis=0)
    y = np.concatenate([train_outputs, val_outputs], axis=0)
    ps = PredefinedSplit(
        test_fold=np.concatenate([-1 * np.ones((train_inputs.shape[0], 1)), np.zeros((val_inputs.shape[0], 1))],
                                 axis=0)
    )
    tuned_parameter = [{'kernel': [RationalQuadratic(length_scale_bounds=(1e-5, 1e5), alpha_bounds=(1e-5, 1e5)),
                                   RBF(length_scale_bounds=(1e-5, 1e5)),
                                   DotProduct(sigma_0_bounds=(1e-5, 1e5)),
                                   Matern(length_scale_bounds=(1e-5, 1e5))]}]
    gpr = GaussianProcessRegressor()
    gpr_c = GridSearchCV(estimator=gpr, param_grid=tuned_parameter, cv=ps, n_jobs=4)
    gpr_c.fit(X, y)
    kernel = gpr_c.best_params_['kernel']
    print('The optimal parameters are: \n kernel:', kernel)

    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(train_inputs, train_outputs)
    return gpr


# Gaussian Process Regression
def ridge_regression(train_inputs, train_outputs, val_inputs, val_outputs):
    X = np.concatenate([train_inputs, val_inputs], axis=0)
    y = np.concatenate([train_outputs, val_outputs], axis=0)
    ps = PredefinedSplit(
        test_fold=np.concatenate([-1 * np.ones((train_inputs.shape[0], 1)), np.zeros((val_inputs.shape[0], 1))],
                                 axis=0)
    )
    tuned_parameter = [{'alpha': [100]}]
    ridge = linear_model.Ridge()
    gpr_c = GridSearchCV(estimator=ridge, param_grid=tuned_parameter, cv=ps, n_jobs=4)
    gpr_c.fit(X, y)
    alpha = gpr_c.best_params_['alpha']
    print('The optimal parameters are: \n alpha:', alpha)

    ridge = linear_model.Ridge(alpha=alpha)
    ridge.fit(train_inputs, train_outputs)
    return ridge


if __name__ == '__main__':
    n_components = 100

    positions = [int(16339 * np.random.rand(1)) for i in range(500)]
    # positions = positions[:1000]
    # positions = [320 * i for i in range(50)]

    # Training
    pod_airfoil = PODAirfoil(n_components, positions=np.array(positions), type='p')
    train_inputs, train_outputs, _ = pod_airfoil.prepare_data(data_path='/mnt/jfs/zhaoxiaoyu/Gappy_POD/subsonicairfoil/airfoil_train.h5')
    val_inputs, val_outputs, _ = pod_airfoil.prepare_data(data_path='/mnt/jfs/zhaoxiaoyu/Gappy_POD/subsonicairfoil/airfoil_val.h5')

    # aug_inputs, aug_outputs = pod_airfoil.sample(200)
    # train_inputs = np.concatenate([train_inputs, aug_inputs], axis=0)
    # train_outputs = np.concatenate([train_outputs, aug_outputs], axis=0)

    # model = svr_regression(train_inputs, train_outputs, val_inputs, val_outputs)
    # model = rf_regression(train_inputs, train_outputs, val_inputs, val_outputs)
    # model = gp_regression(train_inputs, train_outputs, val_inputs, val_outputs)
    model = ridge_regression(train_inputs, train_outputs, val_inputs, val_outputs)

    # Testing
    test_inputs, test_outputs, test_data = pod_airfoil.prepare_data(data_path='/mnt/jfs/zhaoxiaoyu/Gappy_POD/subsonicairfoil/airfoil_train.h5')
    test_pres = model.predict(test_inputs)
    test_maps = pod_airfoil.inverse_transform(coff=test_pres)
    print("MAE of coefficients", np.mean(np.abs(test_pres - test_outputs)))
    print("MAE of reconstruct fields", np.mean(np.abs(test_maps - test_data)))
    print("Max-AE of reconstruct fields",
          np.mean(np.max(np.abs(test_maps - test_data).reshape(test_maps.shape[0], -1), axis=1)))
