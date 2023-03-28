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


class PODAirfoil:
    def __init__(self, n_components, positions, type='vx'):
        self.positions = positions
        f = h5py.File('/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_train.h5', 'r')
        pca_data = f[type][:]
        self.min = np.min(f[type][:], axis=0, keepdims=True)
        self.max = np.max(f[type][:], axis=0, keepdims=True)
        f.close()

        self.type = type
        self.pca = PCA(n_components=n_components)
        X_t = self.pca.fit_transform(pca_data)
        self.max_coeff, self.min_coeff = np.max(X_t, axis=0).reshape(1, -1), np.min(X_t, axis=0).reshape(1, -1)

    def prepare_data(self, data_path='/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_train.h5'):
        f = h5py.File(data_path, 'r')
        data = f[self.type][:]
        data_norm = (f[self.type][:] - self.min) / (self.max - self.min + 1e-10)
        f.close()

        # Normalize the coefficients
        data_coff = self.pca.transform(data)
        data_coff = (data_coff - self.min_coeff) / (self.max_coeff - self.min_coeff)

        sparse_data = []
        for i in range(self.positions.shape[0]):
            sparse_data.append(data_norm[:, self.positions[i]].reshape(-1, 1))
        observe = np.concatenate(sparse_data, axis=-1)
        return observe, data_coff, data

    def inverse_transform(self, coff):
        inverse_coff = coff * (self.max_coeff - self.min_coeff) + self.min_coeff
        return self.pca.inverse_transform(inverse_coff).reshape(coff.shape[0], -1)


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
    n_components = 35

    positions = [16097, 16016, 16087, 15986, 15987, 15978, 15981, 15980, 16018, 16046, 16061, 16069, 16075, 15657,
                 15641, 15633, 15665, 15605, 15598, 15603, 15604, 15654, 15684, 15699, 15707, 15713, 15141, 15124,
                 15135, 15113, 15150, 15157, 15149, 15152, 15146, 15179, 15194, 15202, 15208, 14569, 14590, 14574,
                 14489, 14468, 14482, 14471, 14466, 14470, 14510, 14525, 14533, 14539, 13473, 13584, 13554, 13557,
                 13611, 13580, 13703, 13612, 13603, 13645, 13660, 13668, 13674, 11998, 12094, 11981, 12144, 11978,
                 12106, 11997, 12097, 11996, 12017, 12032, 12040, 12046, 8636, 8506, 8634, 8674, 9277, 9971, 9821,
                 9273, 8676, 8558, 8573, 8581, 8587, 4792, 4638, 4734, 4743, 4719, 4715, 4714, 4720, 4620, 4672,
                 4687, 4695, 4701, 2949, 3067, 2891, 2944, 2947, 3034, 3061, 3018, 3039, 2981, 2996, 3004, 3010,
                 2056, 2012, 1920, 2004, 2013, 1998, 2008, 2006, 2005, 1953, 1968, 1976, 1982, 1356, 1245, 1350,
                 1362, 1328, 1322, 1326, 1321, 1324, 1274, 1289, 1297, 1303, 830, 838, 842, 815, 803, 741, 798, 735,
                 808, 761, 776, 784, 790, 374, 459, 437, 364, 362, 360, 363, 365, 439, 395, 410, 418, 424]
    # positions = positions[:32]
    # positions = [320 * i for i in range(50)]

    # Training
    pod_airfoil = PODAirfoil(n_components, positions=np.array(positions), type='vx')
    train_inputs, train_outputs, _ = pod_airfoil.prepare_data('/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_train.h5')
    val_inputs, val_outputs, _ = pod_airfoil.prepare_data('/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_val.h5')

    # model = svr_regression(train_inputs, train_outputs, val_inputs, val_outputs)
    model = rf_regression(train_inputs, train_outputs, val_inputs, val_outputs)
    # model = gp_regression(train_inputs, train_outputs, val_inputs, val_outputs)

    # Testing
    test_inputs, test_outputs, test_data = pod_airfoil.prepare_data('/mnt/jfs/zhaoxiaoyu/data/airfoil/naca0012_test.h5')
    test_pres = model.predict(test_inputs)
    test_maps = pod_airfoil.inverse_transform(coff=test_pres)
    print("MAE of coefficients", np.mean(np.abs(test_pres - test_outputs)))
    print("MAE of reconstruct fields", np.mean(np.abs(test_maps - test_data)))
    print("Max-AE of reconstruct fields",
          np.mean(np.max(np.abs(test_maps - test_data).reshape(test_maps.shape[0], -1), axis=1)))
