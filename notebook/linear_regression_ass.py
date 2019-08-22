"""线性回归算法的应用
"""
import pandas as pd
import numpy as np
import pandas as pd
import linear_regression as LR


def gd_1():
    # get data from file
    df = pd.read_csv('./data/kc_house_data.csv')
    # get samples
    train_set = df.sample(frac=0.2, random_state=1)
    # get features
    fields = ['sqft_lot', 'sqft_living', 'sqft_above']
    features = train_set[fields].values
    # get targets
    targets = train_set['price'].values
    # set thetas
    # * 根据features的column数来构建
    thetas = np.full(features.shape[1], 0)
    # normalization
    features = LR.normalization(features)
    # add theta_0 and feature_0
    thetas = np.insert(thetas, 0, 0)
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
    # get data from file
    df = pd.read_csv('./data/andrew/exedata1.csv', header=None)
    # get samples
    train_set = df
    # get features
    fields = [0]
    features = train_set[fields].values
    # get targets
    targets = train_set[[1]].values
    # set thetas
    # * 根据features的column数来构建
    thetas = np.full((features.shape[1], 1), 0)
    # normalization
    features = LR.normalization(features)
    # add theta_0 and feature_0
    thetas = np.insert(thetas, 0, 0, axis=0)
    features = np.insert(features, 0, 1, axis=1)
    # set learning rate
    learning_rate = 0.01
    # set iterators numbers
    iterate_num = 5
    # !梯度下降
    J_theta, thetas, plt_J_thetas, plt_thetas = LR.gradient_descent(
        thetas, features, targets, learning_rate, iterate_num)
    # 打印输出结果
    print(f'特征字段的选择为 {fields}')
    print(f'迭代 {iterate_num} 次后')
    print(f'成本函数 J_theta 的值 {J_theta}')
    print(f'成本函数差值为 {plt_J_thetas[-2] - plt_J_thetas[-1]}')
    print(f'thetas {thetas}')



def plot_surface3D():
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

    # features (97,2)
    # Thetas (2, 2500)
    # targets (97, 1)
    print(f'features {features.shape}')
    print(f'Thetas {Thetas.shape}')
    print(f'targets {targets.shape}')

    H = features.dot(Thetas)
    H = H - targets
    H = H ** 2 / 2
    print(f'H {H.shape}')
    J = np.mean(H, axis=0)
    J = np.reshape(J, (50, 50))

    print(f'X {X.shape}')
    print(f'Y {Y.shape}')
    print(f'J {J.shape}')

    print(f'X first row is {X[0,:5]}')
    print(f'Y first row is {Y[0,:5]}')
    print(f'J first row is {J[0,:5]}')

    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface
    surf = ax.plot_surface(X, Y, J, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # plot the contour
    # plt.contour(X, Y, J)

    plt.show()


if __name__ == "__main__":
    # import io
    # import sys

    # sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

    # plot_surface3D()
    gd_2()
