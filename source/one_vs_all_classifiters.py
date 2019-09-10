"""一对多分类器
"""

import numpy as np


def sigmoid(z):
    """sigmoid函数
    
    Arguments:
        z {scalar | vector | matrix} -- sigmoid函数自变量
    
    Returns:
        scalar | vector | matrix -- sigmoid函数计算结果，值域在[0, 1]
    """
    return 1.0 / (1.0 + np.exp(-1 * z))


def cost_function(theta, X, y, l):
    h_theta_x = sigmoid(X.dot(theta))
    temp1 = -1 * y * np.log(h_theta_x)
    temp2 = (1 - y) * np.log(1 - h_theta_x)
    thetaT = np.copy(theta)
    thetaT[0] = 0
    correction = l * (thetaT ** 2).sum() / 2 * X.shape[0]
    J = np.mean((temp1 - temp2))

    return J


def gradient(theta, X, y, l):
    pass
