import pandas as pd
import numpy as np


def normalization(arr):
    '''
    正则化
    =====
    采用 current_value - mean / (max - min) 算法使数值控制在[-1, 1]之间
    其中聚合函数 max min mean 均按列处理, 比如:
    arr = [[1, 2, 3],
           [4, 5, 6]]
    max = [4, 5, 6]
    min = [1, 2, 3]
    mean = [2.5, 3.5, 4.5]
    返回的结果:
    [[-0.5, -0.5, -0.5],
     [ 0.5,  0.5,  0.5]]
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
    arr_max = np.apply_along_axis(np.max, 0, arr)
    arr_min = np.apply_along_axis(np.min, 0, arr)
    arr_mean = np.apply_along_axis(np.mean, 0, arr)
    arr_diff = np.array(arr_max - arr_min)
    # 防止发生除零异常
    arr_diff[arr_diff == 0] = 1

    return (arr - arr_mean) / arr_diff


def verify_parameters(thetas, features, targets=None):
    if targets is None:
        targets = features
    if thetas.shape[0] != features.shape[1] or features.shape[0] != targets.shape[0]:
        raise Exception(
            f'thetas rows is {thetas.shape[0]} '
            f'not equals features columns {features.shape[1]} '
            f'or features rows {features.shape[0]} '
            f'not equals targets rows {targets.shape[0]}')


def calc_prediction_price(thetas, features):
    '''
    计算预测结果
    =============
    根据给定的系数theta和特征features，通过线性假设函数计算预测结果

    Parameters
    ----------
    thetas :   numpy.ndarray[float]
               线性假设函数的系数数组
    features : numpy.ndarray[float]
               特征字段数组

    Returns
    -------
    float
         预测结果
    '''
    # 如果thetas的维度与features的维度不同，则抛出异常
    verify_parameters(thetas, features)

    return features.dot(thetas).sum()


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
    return d_by_features.mean(axis=0)
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
    verify_parameters(thetas, features, targets)

    return (thetas
            - learning_rate
            * calc_J_partial_derivative(thetas, features, targets))


def normal_equation(dataset, targets):
    X = dataset.values
    y = targets.values
    left = np.linalg.inv(X.T.dot(X))
    right = X.T.dot(y)

    return left.dot(right)


def gradient_descent(thetas, features, targets, learning_rate, iterate_num):
    plt_J_thetas = []
    plt_thetas = []

    for _ in range(iterate_num):
        # *** 以下是核心算法 ***
        J_theta = calc_J_value(thetas, features, targets)
        thetas = calc_new_thetas(thetas, features, targets, learning_rate)
        # ***END
        plt_J_thetas.append(J_theta)
        plt_thetas.append(thetas)

    return J_theta, thetas, plt_J_thetas, plt_thetas


def gradient_descent2():
    import pandas as pd

    df = pd.read_csv('../data/kc_house_data.csv')
    train_set = df.sample(frac=0.2, random_state=1)

    virtual_field_name = 'field_0'
    train_set[virtual_field_name] = pd.Series(
        np.full(len(train_set.index), 1),
        index=train_set.index
    )

    fields = [virtual_field_name, 'sqft_lot', 'sqft_living',
              'sqft_above'
              ]
    thetas = np.full((len(fields), ), 0)
    targets = train_set['price'].values
    features = train_set[fields].values

    # normalization
    targets = normalization(targets)
    features = normalization(features)

    J_theta = 100
    learning_rate = 0.1

    plt_J_thetas = []
    plt_thetas = []

    iterate_num = 15000
    J_theta_diff = 1
    for i in range(iterate_num):
        J_theta_old = J_theta
        # print(f'NO.{i}')
        # print(f'J_theta = {J_theta}')
        # print(f'thetas = {thetas}')
        # print(f'J_theta_diff = {J_theta_diff}')
        if J_theta_diff <= 0:
            iterate_num = i
            print('*'*10)
            print(f'在运行了 {i} 次后, J_theta_diff = {J_theta_diff}')
            print('*'*10)
            break

        # *** 以下是核心算法 ***
        J_theta = calc_J_value(thetas, features, targets)
        thetas = calc_new_thetas(thetas, features, targets, learning_rate)
        # ***END

        J_theta_diff = J_theta_old-J_theta

        # for plot
        plt_J_thetas.append(J_theta)
        plt_thetas.append(thetas[1])

    print(f'特征字段的选择为 {fields}')
    print(f'迭代 {iterate_num} 次后')
    print(f'成本函数 J_theta 的值 {J_theta}')
    print(f'成本函数差值为 {J_theta_diff}')
    print(f'thetas {thetas}')

    # plot
    # import matplotlib.pyplot as plt
    # plt.plot(plt_thetas, plt_J_thetas)
    # plt.show()


if __name__ == "__main__":
    import io
    import sys
    import pandas as pd

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

    df = pd.read_csv('../data/kc_house_data.csv')
    train_set = df.sample(frac=0.2, random_state=1)

    fields = ['sqft_lot', #'sqft_living', 'sqft_above'
    ]
    thetas = np.full((len(fields), ), 0)
    targets = train_set['price'].values
    features = train_set[fields].values
    learning_rate = 1
    iterate_num = 10000

    # normalization
    targets = targets / 100000
    features = normalization(features)

    # add theta_0 and x_0
    thetas = np.insert(thetas, 0, 0)
    features = np.insert(features, 0, 1, axis=1)

    J_theta, thetas, plt_J_thetas, plt_thetas = gradient_descent(
        thetas, features, targets, learning_rate, iterate_num)

    print(f'特征字段的选择为 {fields}')
    print(f'迭代 {iterate_num} 次后')
    print(f'成本函数 J_theta 的值 {J_theta}')
    print(f'成本函数差值为 {plt_J_thetas[-2] - plt_J_thetas[-1]}')
    print(f'thetas {thetas}')

    # 运用测试集进行验证
    test_set = df.sample(frac=0.3, random_state=2)
    test_set_arr = test_set[fields].values
    test_set_targets = test_set['price'].values

    # add x_0
    test_set_arr = np.insert(test_set_arr, 0, 1, axis=1)

    # normalization
    # test_set_targets = test_set_targets / 100000
    # test_set_arr = normalization(test_set_arr)

    predicts = test_set_arr.dot(thetas)
    print(f'test_set_arr = {test_set_arr[:10]}')

    result = np.array([test_set_targets, predicts])
    print(f'result is {result[:10]}')
