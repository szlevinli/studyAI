"""逻辑回归算法实现
"""

import numpy as np


def sigmoid(z):
    """返回 sigmoid 函数计算结果

    Arguments:
        z {float | vector | matrix} -- sigmoid 函数的自变量。可以是 scalar, vector, matrix

    Return:
        {float | vector | matrix} -- sigmoid 函数的计算结果。可以是 scalar, vector, matrix
    """
    return 1 / (1 + np.exp(-z))


def cost_function(theta, X, y):
    """计算成本函数

    Arguments:
        theta {ndarray 1D} -- theta系数 vector
        X {ndarray 2D} -- 特征数据 matrix
        y {ndarray 1D} -- 目标数据 vector

    Returns:
        float -- 成本函数计算结果 J scalar
        ndarray 1D -- 梯度值 vector
    """
    # *X.shape = (m, n) is 2D
    # *theta.shape = (n,) is 1D
    # *y.shape = (m,) is 1D
    # H_theta_x.shape = (m,) is 1D
    H_theta_x = sigmoid(X.dot(theta))
    # temp1.shape = (m,) is 1D
    temp1 = -1 * (y * np.log(H_theta_x))
    # temp2.shape = (m,) is 1D
    temp2 = (1 - y) * np.log(1 - H_theta_x)
    # J is scalar
    J = (temp1 - temp2).mean()
    # X is (m, n) X^T is (n, m) dot (H_theta_x - y) is (m,)
    # gradient is (n,) because (n, m) dot (m,) is (n,)
    # 目前 gradient 中放置的是每个特征列的总误差，共 n 列
    gradient = X.transpose().dot(H_theta_x - y)
    # 求每个特征列的平均误差
    gradient = gradient * (1 / y.shape[0])

    return J, gradient
