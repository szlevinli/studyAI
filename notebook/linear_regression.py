import numpy as np


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


def verify_parameters(thetas, features, targets=None):
    '''
    参数验证
    =======
    对需要操作的 thetas features targets 进行维度上的校验
    校验不通过则抛出异常

    Parameters
    ----------
    thetas:    numpy.ndarray[float] 2D
               线性假设函数的系数数组
    features : numpy.ndarray[float] 2D
               特征字段数组
    targets :  numpy.ndarray[float] 2D
               实际值/目标值

    Returns
    -------
          None
    '''
    import inspect
    # 处理None参数
    if targets is None:
        targets = features
    # 使用 inspect 包获取函数参数信息
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    # 所有参数必须是二维的
    for key in args:
        if values[key].ndim != 2:
            raise Exception(
                f'{key} dimensions must be 2D'
            )
    # 参数间维度判读
    if thetas.shape[0] != features.shape[1]:
        raise Exception(
            f'thetas rows is {thetas.shape[0]} '
            f'not equals features columns {features.shape[1]}')
    if features.shape[0] != targets.shape[0]:
        raise Exception(
            f'features rows {features.shape[0]} '
            f'not equals targets rows {targets.shape[0]}')


def calc_J_value(thetas, features, targets):
    '''
    计算成本函数
    ============
    采用 MSE 作为成本函数

    Parameters
    ----------
    thetas:    numpy.ndarray[float]
               线性假设函数的系数数组
    features : numpy.ndarray[float]
               特征字段数组
    targets :  numpy.ndarray[float]
               实际值/目标值

    Returns
    -------
    float
         成本函数计算结果值
    '''
    # 如果thetas的维度与features的维度不同，则抛出异常
    verify_parameters(thetas, features, targets)

    p = features.dot(thetas)
    d = p - targets
    c = d ** 2 / 2
    return c.mean()


def calc_J_partial_derivative(thetas, features, targets):
    '''
    计算成本函数的偏导数
    =============

    Parameters
    ----------
    thetas:    numpy.ndarray[float]
               线性假设函数的系数数组
    features : numpy.ndarray[float]
               特征字段数组
    targets :  numpy.ndarray[float]
               实际值/目标值

    Returns
    -------
    numpy.ndarray[float]
         成本函数的偏导数
    '''
    # 如果thetas的维度与features的维度不同，则抛出异常
    verify_parameters(thetas, features, targets)

    # 预测值
    p = features.dot(thetas)
    # 预测值与实际值的差
    d = p - targets
    # 将预测值 reshape 为 (n, 1) 即 n 行 1 列
    # 默认情况下 [1,2,3] 的 shape 为 (3,)
    # 需要将其 reshape 称为 (3, 1)
    # 此时 [1,2,3] 变为 [[1], [2], [3]]
    # 目的是保证可以进行矩阵乘法(注意不是矩阵点积)
    d = np.reshape(d, (d.shape[0], 1))
    # 进行矩阵乘法, 注意不是矩阵点积操作
    d_by_features = d * features

    # 将矩阵按列计算平均值
    d_mean = d_by_features.mean(axis=0)
    return np.reshape(d_mean, (d_mean.shape[0], 1))
    # 这里说明一下矩阵的规模
    # thetas: (3, 1)
    # features: (5, 3)
    # targets: (5, 1)
    # p: (5, 1), features dot thetas, (5, 3) dot (3, 1) = (5, 1)
    # d: (5, 1), p - targets, (5, 1) - (5, 1) = (5, 1)
    # d_by_features: (5, 3), d * features, (5, 1) * (5, 3) = (5, 3)
    # d_by_features.mean(axis=0): (3,), d_by_features是 3 列 按列计算平均值后的 (3, 1)


def calc_new_thetas(thetas, features, targets, learning_rate):
    '''
    计算新线性假设函数的系数
    ========
    根据梯度下降算法规则, 计算新的线性假设函数的系数, 公式为:
    thetas = thetas - 学习率 * 成本函数偏导值

    Parameters
    ----------
    thetas :        numpy.ndarray[float]
                    线性假设函数的系数数组
    features :      numpy.ndarray[float]
                    特征字段数组
    targets :       numpy.ndarray[float]
                    实际值/目标值
    learning_rate : float
                    学习率

    Returns
    -------
    numpy.ndarray[float]
         新线性假设函数的系数
    '''
    # 如果thetas的维度与features的维度不同，则抛出异常
    verify_parameters(thetas, features, targets)

    return (thetas
            - learning_rate
            * calc_J_partial_derivative(thetas, features, targets))


def gradient_descent(thetas, features, targets, learning_rate, iterate_num):
    """梯度下降

    Arguments:
        thetas {numpy.ndarray<float>} -- 线性假设函数的系数数组
        features {numpy.ndarray<float>} -- 特征数据
        targets {numpy.ndarray<float>} -- 目标值/真实值
        learning_rate {float} -- 学习率
        iterate_num {int} -- 迭代次数

    Returns:
        {float} -- 迭代完成后最终的成本函数结果
        {numpy.ndarray<float>} -- 迭代完成后最终的线性假设函数的系数数组，也就是我们最终需要的结果
        {numpy.ndarray<float>} -- 每次迭代的成本函数结果
        {numpy.ndarray<float>} -- 每次迭代的线性假设函数的系数，2D数组，每行表示某次迭代后计算出来的线性假设函数的系数数组

    """
    # 如果thetas的维度与features的维度不同，则抛出异常
    verify_parameters(thetas, features, targets)

    plt_J_thetas = []
    plt_thetas = []

    for _ in range(iterate_num):
        # *** 以下是核心算法 ***
        J_theta = calc_J_value(thetas, features, targets)
        thetas_old = thetas
        thetas = calc_new_thetas(thetas, features, targets, learning_rate)
        # ***END
        plt_J_thetas.append(J_theta)
        plt_thetas.append(list(thetas_old))

    return J_theta, thetas, np.array(plt_J_thetas), np.array(plt_thetas)


def normal_equation(features, targets):
    """正规方程方式计算线性假设函数系数

    Arguments:
        features {numpy.ndarray<float>} -- 特征数据
        targets {numpy.ndarray<float>} -- 目标值/真实值

    Returns:
        {numpy.ndarray<float>} -- 迭代完成后最终的线性假设函数的系数数组，也就是我们最终需要的结果
    """
    if (features.shape[0] != targets.shape[0]):
        raise Exception(
            f'features rows {features.shape[0]} '
            f'not equals targets rows {targets.shape[0]}')

    X = features
    y = targets
    left = np.linalg.inv(X.T.dot(X))
    right = X.T.dot(y)

    return left.dot(right)


if __name__ == "__main__":
    pass
