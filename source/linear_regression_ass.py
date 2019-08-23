"""线性回归算法的应用
"""
import pandas as pd
import numpy as np
import pandas as pd
import linear_regression as LR


def get_data_from_csv(file_name, feature_fields, targets_fields, frac, **kwargs):
    """从csv文件中获取数据构建 thetas-系数数组，features-特征数据，targets-真实值

    Arguments:
        file_name {string} -- 文件名
        feature_fields {numpy.ndarray<string>(n,)} -- 特征字段名称
        targets_fields {numpy.ndarray<string>(n,)} -- 真实值字段名称
        frac {float} -- 从数据集中提取的样本比率(0, 1]

    Returns:
        numpy.ndarray<float>(n, 1) -- thetas-系数数组
        numpy.ndarray<float>(m, n) -- features-特征数据
        numpy.ndarray<float>(m, 1) -- targets-真实值
    """
    # get data from file
    header = kwargs['header'] if 'header' in kwargs else 0
    df = pd.read_csv(file_name, header=header)
    # get samples
    train_set = df.sample(frac=frac, random_state=1)
    # get features
    fields = feature_fields
    features = train_set[fields].values
    # get targets
    targets = train_set[targets_fields].values
    # set thetas
    # * 根据features的column数来构建
    thetas = np.full((features.shape[1], 1), 0)

    return thetas, features, targets


def gd_1():
    fields = ['sqft_lot', 'sqft_living', 'sqft_above']
    thetas, features, targets = get_data_from_csv(
        './data/kc_house_data.csv',
        fields,
        ['price'],
        0.2
    )
    # normalization
    features = LR.normalization(features)
    # add theta_0 and feature_0
    thetas = np.insert(thetas, 0, 0, axis=0)
    features = np.insert(features, 0, 1, axis=1)
    # set learning rate
    learning_rate = 0.01
    # set iterators numbers
    iterate_num = 5000
    # !梯度下降
    J_theta, thetas, plt_J_thetas, plt_thetas = LR.gradient_descent(
        thetas, features, targets, learning_rate, iterate_num)
    # 打印输出结果
    print(f'特征字段的选择为 {fields}')
    print(f'迭代 {iterate_num} 次后')
    print(f'成本函数 J_theta 的值 {J_theta}')
    print(f'成本函数差值为 {plt_J_thetas[-2] - plt_J_thetas[-1]}')
    print(f'thetas {thetas}')


def gd_2():
    fields = [0]
    thetas, features, targets = get_data_from_csv(
        './data/andrew/exedata1.csv',
        fields,
        [1],
        1,
        header=None
    )
    # normalization
    features = LR.normalization(features)
    # add theta_0 and feature_0
    thetas = np.insert(thetas, 0, 0, axis=0)
    features = np.insert(features, 0, 1, axis=1)
    # set learning rate
    learning_rate = 0.01
    # set iterators numbers
    iterate_num = 1500
    # !梯度下降
    J_theta, thetas, plt_J_thetas, plt_thetas = LR.gradient_descent(
        thetas, features, targets, learning_rate, iterate_num)
    # 打印输出结果
    print(f'特征字段的选择为 {fields}')
    print(f'迭代 {iterate_num} 次后')
    print(f'成本函数 J_theta 的值 {J_theta}')
    print(f'成本函数差值为 {plt_J_thetas[-2] - plt_J_thetas[-1]}')
    print(f'thetas {thetas}')


def plot_surface3D(theta_0, theta_1, features, targets, split_num):
    """画 3D surface 和 contour 图

    Arguments:
        theta_0 {float} -- theta_0的值
        theta_1 {float} -- theta_1的值
        features {numpy.ndarry<float> (m, 2)} -- 特征数据 2D
        targets {numpy.ndarry<float> (m, 1)} -- 真实值 2D
        split_num {int} -- theta_0和theta_1的切片数
    """
    # x 坐标 theta_0的切片数据
    X = np.linspace(0, theta_0, split_num)
    # x 坐标 theta_1的切片数据
    Y = np.linspace(0, theta_1, split_num)
    # X 和 Y 的笛卡尔积转换
    # X 和 Y 转换前的 shape (n,)
    # X 和 Y 笛卡尔积转换后的 shape (n, n)
    X, Y = np.meshgrid(X, Y)
    # 打平 X
    X_1 = X.flatten()
    # 打平 Y
    Y_1 = Y.flatten()
    # 构建 Thetas 其 shape (2, n*n)
    X_1 = np.reshape(X_1, (1, X_1.shape[0]))
    Y_1 = np.reshape(Y_1, (1, Y_1.shape[0]))
    Thetas = np.append(X_1, Y_1, axis=0)
    # *计算预测值 H_theta
    # features.shape (m, 2) Thetas.shape (2, n)
    # H.shape (m, n)
    H = features.dot(Thetas)
    H = H - targets
    H = H ** 2 / 2
    # *计算成本值 J
    # J.shape (m, n)
    J = np.mean(H, axis=0)
    # reshape J
    J = np.reshape(J, (50, 50))
    # !plot 3D surface and contour
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    # Plot the surface
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, J, cmap=cm.coolwarm,
                     linewidth=0, antialiased=False)
    # plot the contour
    ax2 = fig.add_subplot(122)
    ax2.contour(X, Y, J)

    plt.show()


def plot_surface3D_1():
    fields = [0]
    _, features, targets = get_data_from_csv(
        './data/andrew/exedata1.csv',
        fields,
        [1],
        1,
        header=None
    )
    # add theta_0 and feature_0
    features = np.insert(features, 0, 1, axis=1)
    targets = np.reshape(targets, (targets.shape[0], 1))
    theta_0 = 5.8391334
    theta_1 = 4.59303983
    # plot
    plot_surface3D(theta_0, theta_1, features, targets, 50)


def plot_surface3D_2():
    fields = ['sqft_living']
    _, features, targets = get_data_from_csv(
        './data/kc_house_data.csv',
        fields,
        ['price'],
        0.2
    )
    # add theta_0 and feature_0
    features = np.insert(features, 0, 1, axis=1)
    targets = np.reshape(targets, (targets.shape[0], 1))
    theta_0 = 5.8391334
    theta_1 = 4.59303983
    # plot
    plot_surface3D(theta_0, theta_1, features, targets, 50)


if __name__ == "__main__":
    plot_surface3D_2()
