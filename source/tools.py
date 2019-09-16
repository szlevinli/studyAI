import numpy as np


def sigmoid(z):
    """返回 sigmoid 函数计算结果

    Arguments:
        z {float | vector | matrix} -- sigmoid 函数的自变量。可以是 scalar, vector, matrix

    Return:
        {float | vector | matrix} -- sigmoid 函数的计算结果。可以是 scalar, vector, matrix
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_gradient(z):
    """返回 sigmoid 函数的导数计算结果

    Arguments:
        z {float | vector | matrix} -- sigmoid 函数的自变量。可以是 scalar, vector, matrix

    Returns:
        {float | vector | matrix} -- sigmoid 函数的计算结果。可以是 scalar, vector, matrix
    """
    s = sigmoid(z)

    return s * (1 - s)


def normalization(arr):
    '''
    正则化
    =====
    采用 （current_value - mean） / std
    std: standard deviation 标准偏差
    按列计算 std mean
    特别注意算法中有除法运算, 需要考虑除零异常问题, 这里采用方法是如果被除数为零, 则将该被除数置为 1,
    返回结果将为 0, 并不会影响结果

    Parameters
    ----------
    s: numpy.ndarry[float]
       需正则化的数据, 支持二维数组

    Returns
    -------
    numpy.ndarry[float]
          正则化后的数据
    '''
    arr_std = np.apply_along_axis(np.std, 0, arr)
    arr_mean = np.apply_along_axis(np.mean, 0, arr)
    # 防止发生除零异常
    arr_std[arr_std == 0] = 1

    return (arr - arr_mean) / arr_std


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


# def map_thetas(nn_weights, nn_layers):
#     start_size = 0
#     thetas = np.array([])
#     for layer_sizes in nn_layers:
#         theta = np.reshape(nn_weights[start_size:layer_sizes[0] * (
#         layer_sizes[1] + 1)], (layer_sizes[0], layer_sizes[1] + 1))
#         thetas = np.concatenate((theta, thetas))
