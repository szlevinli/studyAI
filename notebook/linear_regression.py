import pandas as pd
import numpy as np
import pandas as pd


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
        thetas_old = thetas
        thetas = calc_new_thetas(thetas, features, targets, learning_rate)
        # ***END
        plt_J_thetas.append(J_theta)
        plt_thetas.append(list(thetas_old))

    return J_theta, thetas, np.array(plt_J_thetas), np.array(plt_thetas)


def gd_2():
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


def gd_3():
    df = pd.read_csv('../data/kc_house_data.csv')
    train_set = df.sample(frac=0.2, random_state=1)

    fields = ['sqft_lot',  # 'sqft_living', 'sqft_above'
              ]
    thetas = np.full((len(fields), ), 0)
    targets = train_set['price'].values
    features = train_set[fields].values
    learning_rate = 1
    iterate_num = 10000

    # normalization
    # targets = targets / 100000
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


def gd_4():
    thetas = np.array([0, 0])
    features = np.array([[1, -0.867188766],
                         [1, 0.026727652],
                         [1, 1.575359235],
                         [1, -1.232167203],
                         [1, 0.497269083]])
    targets = np.array([
        221900,
        538000,
        180000,
        604000,
        510000
    ])
    learning_rate = 0.1
    iterate_num = 999

    J_theta, thetas, plt_J_thetas, plt_thetas = gradient_descent(
        thetas, features, targets, learning_rate, iterate_num)

    # print(f'plt_J_thetas = {plt_J_thetas}')
    # print(f'plt_thetas = {plt_thetas}')
    print(f'J_theta = {J_theta}')

def gd_5():
    df = pd.read_csv('../data/andrew/exedata1.csv', header=None)

    thetas = np.array([0, 0])
    features = df[[0]].values
    features = np.insert(features, 0, 1, axis=1)
    targets = df[1].values
    learning_rate = 0.01
    iterate_num = 1500

    J_theta, thetas, plt_J_thetas, plt_thetas = gradient_descent(
        thetas, features, targets, learning_rate, iterate_num)

    print(f'J_theta={J_theta}')
    print(f'thetas={thetas}')
    print(f'plt_thetas={plt_thetas}')


def plot_surface3D():
    # import matplotlib.pyplot as plt
    # from matplotlib import cm

    df = pd.read_csv('./data/andrew/exedata1.csv', header=None)

    thetas = np.array([-3.63029144, 1.16636235])
    features = df[[0]].values
    features = np.insert(features, 0, 1, axis=1)
    targets = df[1].values
    targets = np.reshape(targets, (targets.shape[0], 1))

    X = np.linspace(0, thetas[0], 50)
    Y = np.linspace(0, thetas[1], 50)
    X, Y = np.meshgrid(X, Y)
    X_1 = X.flatten()
    Y_1 = Y.flatten()
    X_1 = np.reshape(X_1, (1, X_1.shape[0]))
    Y_1 = np.reshape(Y_1, (1, Y_1.shape[0]))
    Thetas = np.append(X_1, Y_1, axis=0)
    print('')

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')

    # # Plot the surface.
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
    #                     linewidth=0, antialiased=False)

    # plt.show()


if __name__ == "__main__":
    import io
    import sys

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

    plot_surface3D()
    # gd_5()
