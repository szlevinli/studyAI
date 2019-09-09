"""逻辑回归算法实现
"""

import numpy as np
from decorators import debug


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
    J = cost_function_(theta, X, y)
    g = gradient(theta, X, y)

    return J, g


# @debug
def cost_function_(theta, X, y):
    """计算成本函数

    Arguments:
        theta {ndarray 1D} -- theta系数 vector
        X {ndarray 2D} -- 特征数据 matrix
        y {ndarray 1D} -- 目标数据 vector

    Returns:
        float -- 成本函数计算结果 J scalar
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

    return J


def gradient(theta, X, y):
    """计算梯度

    Arguments:
        theta {ndarray 1D} -- theta系数 vector
        X {ndarray 2D} -- 特征数据 matrix
        y {ndarray 1D} -- 目标数据 vector

    Returns:
        ndarray 1D -- 梯度值 vector
    """
    # *X.shape = (m, n) is 2D
    # *theta.shape = (n,) is 1D
    # *y.shape = (m,) is 1D
    # H_theta_x.shape = (m,) is 1D
    H_theta_x = sigmoid(X.dot(theta))
    # X is (m, n) X^T is (n, m) dot (H_theta_x - y) is (m,)
    # gradient is (n,) because (n, m) dot (m,) is (n,)
    # 目前 gradient 中放置的是每个特征列的总误差，共 n 列
    gradient = X.transpose().dot(H_theta_x - y)
    # 求每个特征列的平均误差
    gradient = gradient * (1 / y.shape[0])

    return gradient


def map_featrue(X1, X2, degree):
    """特征工程 多项式方法
    将给定的两个特征进行多项式算法, 得到新的特征从而可以更好的拟合数据
    算法为: 假设 degree=i, 则返回的结果为
    X1^i * X2^0, X1^(i-1) * X2^1, X1^(i-2) * X2^2, ..., X^0 * X2^i

    Arguments:
        X1 {ndarray(n, 1) 2D} -- 特征 X1
        X2 {ndarray(n, 1) 2D} -- 特征 X2
        degree {int} -- 多项式的 degree

    Returns:
        ndarray(n, m) -- 生成的新的特征, 其中 m=1+2+...+degree+degree;
        比如: degree = 6, 则 m=1+2+3+4+5+6+6=27
    """
    out = np.ones((X1.shape[0], 1))
    for i in range(1, degree+1):
        for j in range(i+1):
            temp = (X1 ** (i - j)) * (X2 ** j)
            out = np.concatenate((out, temp), axis=1)
    return out


def cost_function_reg(theta, X, y, l):
    """计算成本函数使用正则化方式

    Arguments:
        theta {ndarray 1D} -- theta系数 vector
        X {ndarray 2D} -- 特征数据 matrix
        y {ndarray 1D} -- 目标数据 vector
        l {float scalar} -- lambda 值

    Returns:
        float -- 成本函数计算结果 J scalar
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
    # calculate correction
    thetaT = np.copy(theta)
    thetaT[0] = 0
    correction = (thetaT ** 2).sum() * l / 2 * X.shape[0]
    # J is scalar
    J = (temp1 - temp2).mean() + correction

    return J


def gradient_reg(theta, X, y, l):
    """计算梯度使用正则化方式

    Arguments:
        theta {ndarray 1D} -- theta系数 vector
        X {ndarray 2D} -- 特征数据 matrix
        y {ndarray 1D} -- 目标数据 vector
        l {float scalar} -- lambda 值

    Returns:
        ndarray 1D -- 梯度值 vector
    """
    # *X.shape = (m, n) is 2D
    # *theta.shape = (n,) is 1D
    # *y.shape = (m,) is 1D
    # H_theta_x.shape = (m,) is 1D
    H_theta_x = sigmoid(X.dot(theta))
    # correction correction
    thetaT = np.copy(theta)
    thetaT[0] = 0
    correction = l * thetaT / X.shape[0]
    # X is (m, n) X^T is (n, m) dot (H_theta_x - y) is (m,)
    # gradient is (n,) because (n, m) dot (m,) is (n,)
    # 目前 gradient 中放置的是每个特征列的总误差，共 n 列
    gradient = X.transpose().dot(H_theta_x - y)
    # 求每个特征列的平均误差
    gradient = gradient * (1 / y.shape[0])
    # regular gradient
    gradient = gradient + correction

    return gradient
