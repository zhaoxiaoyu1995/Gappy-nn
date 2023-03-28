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
import pickle


class PODCylinder:
    def __init__(self, n_components, positions, pod_index):
        self.positions = positions
        df = open('/home/ubuntu/zhaoxiaoyu/data/cylinder/Cy_Taira.pickle', 'rb')
        pca_data = np.transpose(pickle.load(df), (0, 3, 1, 2))[pod_index, :, :, :]
        df.close()

        self.pca = PCA(n_components=n_components)
        X_t = self.pca.fit_transform(pca_data.reshape(len(pod_index), -1))
        self.max_coeff, self.min_coeff = np.max(X_t, axis=0).reshape(1, -1), np.min(X_t, axis=0).reshape(1, -1)
        self.size = pca_data.shape[-3:]

    def prepare_data(self, data_index):
        df = open('/home/ubuntu/zhaoxiaoyu/data/cylinder/Cy_Taira.pickle', 'rb')
        data = np.transpose(pickle.load(df), (0, 3, 1, 2))[data_index, :, :, :]
        df.close()
        # Normalize the coefficients
        data_coff = self.pca.transform(data.reshape(len(data_index), -1))
        data_coff = (data_coff - self.min_coeff) / (self.max_coeff - self.min_coeff)

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
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
    n_components = 50
    positions = np.array([[50, 25], [62, 30], [50, 35], [62, 40], [50, 45], [62, 50], [50, 55], [62, 60]])
    pod_index = [i for i in range(4250)]
    train_index = [i for i in range(3500)]
    val_index = [i for i in range(3500, 4250)]
    test_index = [i for i in range(4900, 4901)]

    # Training
    pod_cylinder = PODCylinder(n_components, positions, pod_index)
    train_inputs, train_outputs, _ = pod_cylinder.prepare_data(train_index)
    val_inputs, val_outputs, _ = pod_cylinder.prepare_data(val_index)

    model = svr_regression(train_inputs, train_outputs, val_inputs, val_outputs)
    # model = rf_regression(train_inputs, train_outputs, val_inputs, val_outputs)
    # model = gp_regression(train_inputs, train_outputs, val_inputs, val_outputs)

    # Testing
    test_inputs, test_outputs, test_data = pod_cylinder.prepare_data(test_index)
    test_pres = model.predict(test_inputs)
    test_maps = pod_cylinder.inverse_transform(coff=test_pres)
    print("MAE of coefficients", np.mean(np.abs(test_pres - test_outputs)))
    print("MAE of reconstruct fields", np.mean(np.abs(test_maps - test_data)))
    print("Max-AE of reconstruct fields",
          np.mean(np.max(np.abs(test_maps - test_data).reshape(test_maps.shape[0], -1), axis=1)))

    import scipy.io as sio

    sio.savemat('svr.mat', {
        'true': test_data[-1, 0, :, :],
        'pre': test_maps[-1, 0, :, :]
    })
