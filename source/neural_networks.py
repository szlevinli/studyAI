"""神经网络算法实现
"""

import numpy as np
from logistic_regression import sigmoid


def predict(theta1, theta2, X):
    """通过神经网络（3层）计算预测结果

    Arguments:
        theta1 {vector} -- 第1层映射到第2层的权重向量
        theta2 {vector} -- 第2层映射到第3层的权重向量
        X {matrix} -- 输入层的数据

    Returns:
        vector -- 预测结果
    """
    #! 第1层映射到第2层
    # add bias unit
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    # X shape is (m, n)
    # theta1 shape is (l, n)
    # so (X @ theta1.T) shape is (m, l)
    layer2_z = X @ theta1.T  # shape is (m, l)
    layer2_a = sigmoid(layer2_z)  # shape is (m, l)
    #! 第2层映射到第3层
    # add bias unit
    layer2_a = np.concatenate(
        (np.ones((layer2_a.shape[0], 1)), layer2_a), axis=1)
    # layer3_z shape is (m, l) * (r, l)
    layer3_z = layer2_a @ theta2.T
    layer3_a = sigmoid(layer3_z)
    #! 返回结果
    # 因为真实值是1-10，而返回的最大值索引是0-9，因此需要加1
    return np.argmax(layer3_a, axis=1) + 1


def nn_cost_function(nn_weights, input_layer_size, hidden_layer_size, num_labels, X, y, lbd):
    """计算神经网络成本值

    Arguments:
        nn_weights {vector} -- 权重向量
        input_layer_size {int} -- 输入层单元个数
        hidden_layer_size {int} -- 隐藏层单元个数
        num_labels {int} -- 分类标签数，同时也是输出层单元个数
        X {matrix} -- 输入值/特征值
        y {vector} -- 真实值 shape 需要用 2D 形式
        lbd {float} -- 惩罚项系数

    Returns:
        float -- 残值 residual
    """
    #! 需要用到的数据
    # 输入数据的记录数（行数）
    m = X.shape[0]
    # 将真实值 y 映射成 vector
    # y 的值域是 [1, 10]
    # y = 1 map to [True, False, False, False, False, False, False, False,False, False]
    # y = 5 map to [False, False, False, False,  True, False, False, False, False, False]
    # y = 10 map to [False, False, False, False, False, False, False, False, False, True]
    y_ = y == np.arange(1, num_labels + 1)
    # Reshape nn_weights to theta1 and theta2
    theta1 = np.reshape(nn_weights[:hidden_layer_size * (
        input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1))
    theta2 = np.reshape(nn_weights[hidden_layer_size * (
        input_layer_size + 1):], (num_labels, hidden_layer_size + 1))

    #! from layer 1 map to layer2
    # add bias
    a1 = np.concatenate((np.ones((m, 1)), X), axis=1)
    z2 = a1 @ theta1.T
    a2 = sigmoid(z2)
    #! from layer 2 map to layer3
    # add bias
    a2 = np.concatenate((np.ones((m, 1)), a2), axis=1)
    z3 = a2 @ theta2.T
    a3 = sigmoid(z3)

    H_theta_x = a3

    #! compute residual
    # y_ shape is (m, num_labels)
    # H_theta_x shape is (m, num_labels)
    temp1 = -1 * y_ * np.log(H_theta_x)
    temp2 = (1 - y_) * np.log(1 - H_theta_x)
    J_ = temp1 - temp2
    J_ = J_.sum(axis=1)
    J = J_.mean()

    #! compute regularization
    # theta 的 第一列即 偏置单元 的 theta 不参与, 因此需要置为 0
    T1 = np.copy(theta1)
    T1[:, 0] = 0
    T2 = np.copy(theta2)
    T2[:, 0] = 0
    # computer correction
    temp1 = T1 ** 2
    temp1 = temp1.sum()
    temp2 = T2 ** 2
    temp2 = temp2.sum()
    correction = (lbd / (2 * m)) * (temp1 + temp2)

    return J + correction
