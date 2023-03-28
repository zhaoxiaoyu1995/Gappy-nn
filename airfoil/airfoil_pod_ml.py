# -*- coding: utf-8 -*-
# @Time    : 2022/10/8 2:08
# @Author  : zhaoxiaoyu
# @File    : airfoil_pod_ml.py
from sklearn.multioutput import MultiOutputRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, DotProduct, Matern
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import numpy as np
import scipy.io as sio


class PODAirfoil:
    def __init__(self, n_components, positions, pod_index, type='vx'):
        self.positions = positions
        f = sio.loadmat('/mnt/jfs/zhaoxiaoyu/Gappy_POD/airfoil_incompressible.mat')
        pca_data = f[type][pod_index, :, :, :]

        self.type = type
        self.pca = PCA(n_components=n_components)
        X_t = self.pca.fit_transform(pca_data.reshape(len(pod_index), -1))
        self.max_coeff, self.min_coeff = np.max(X_t, axis=0).reshape(1, -1), np.min(X_t, axis=0).reshape(1, -1)
        self.size = pca_data.shape[-3:]

    def prepare_data(self, data_index):
        f = sio.loadmat('/mnt/jfs/zhaoxiaoyu/Gappy_POD/airfoil_incompressible.mat')
        data = f[self.type][data_index, :, :, :]

        # Normalize the coefficients
        data_coff = self.pca.transform(data.reshape(len(data_index), -1))
        data_coff = (data_coff - self.min_coeff) / (self.max_coeff - self.min_coeff)

        sparse_data = []
        for i in range(self.positions.shape[0]):
            sparse_data.append(data[:, 0, self.positions[i, 0], :][:, self.positions[i, 1]].reshape(-1, 1))
        observe = np.concatenate(sparse_data, axis=-1)
        return observe, data_coff, data

    def inverse_transform(self, coff):
        inverse_coff = coff * (self.max_coeff - self.min_coeff) + self.min_coeff
        return self.pca.inverse_transform(inverse_coff).reshape(coff.shape[0], self.size[0], self.size[1], self.size[2])


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


if __name__ == '__main__':
    n_components = 20

    positions = [[65, 126], [90, 139], [65, 151], [90, 114], [90, 164], [115, 139], [65, 101], [115, 164]]
    positions = positions[:8]
    print(positions)
    pod_index = [i for i in range(700)]
    train_index = [i for i in range(500)]
    val_index = [i for i in range(500, 700)]
    test_index = [i for i in range(999, 1000)]

    # Training
    pod_airfoil = PODAirfoil(n_components, positions=np.array(positions), pod_index=pod_index, type='p')
    train_inputs, train_outputs, _ = pod_airfoil.prepare_data(train_index)
    val_inputs, val_outputs, _ = pod_airfoil.prepare_data(val_index)

    # model = svr_regression(train_inputs, train_outputs, val_inputs, val_outputs)
    model = rf_regression(train_inputs, train_outputs, val_inputs, val_outputs)
    # model = gp_regression(train_inputs, train_outputs, val_inputs, val_outputs)

    # Testing
    test_inputs, test_outputs, test_data = pod_airfoil.prepare_data(test_index)
    test_pres = model.predict(test_inputs)
    test_maps = pod_airfoil.inverse_transform(coff=test_pres)
    print("MAE of coefficients", np.mean(np.abs(test_pres - test_outputs)))
    print("MAE of reconstruct fields", np.mean(np.abs(test_maps - test_data)))
    print("Max-AE of reconstruct fields",
          np.mean(np.max(np.abs(test_maps - test_data).reshape(test_maps.shape[0], -1), axis=1)))

    import scipy.io as sio

    sio.savemat('rfr_p.mat', {
        'true': test_data[-1, 0, :, :],
        'pre': test_maps[-1, 0, :, :]
    })
