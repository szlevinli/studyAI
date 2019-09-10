"""一对多分类器
"""

import numpy as np
from scipy.optimize import minimize
from logistic_regression import cost_function_reg, gradient_reg


def one_vs_all(X, y, num_labels, lbd):
    """实现多分类器
    `num_labels`表示类别数量

    Arguments:
        X {matrix (m, n)} -- 特征数据
        y {vector (m,)} -- 实际值
        num_labels {int} -- 分类数量
        lbd {float} -- lambda 超参,用于正则化

    Return:
        {matrix (c, n)} -- 每行表示一个分类的 theta 值, theta 值共有 n+1 列, 与 X 的列数一致
    """
    m = X.shape[0]
    n = X.shape[1]
    all_theta = np.zeros((num_labels, n+1))
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    for i in range(num_labels):
        init_theta = np.zeros((n+1))
        R = minimize(cost_function_reg, init_theta, args=(X, (y % num_labels == i), lbd), jac=gradient_reg)
        if not R.success:
            raise Exception(f'minimize optimize fail, message is {R.message}')
        all_theta[i, :] = R.x.T

    return all_theta
